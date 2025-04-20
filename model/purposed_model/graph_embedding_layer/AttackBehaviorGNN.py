import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report
from purposed_model.graph_embedding_layer.AttackGraphGenerator import AttackGraphGenerator
from purposed_model.graph_embedding_layer.GNNModel import GNNModel


class AttackBehaviorGNN:
    def __init__(self,
                 num_features=768,
                 hidden_channels=256,
                 feature_sim_threshold=0.7,
                 time_threshold=60,
                 k_nearest=5,
                 device='cuda'):

        """
        Initialize Attack Behavior GNN with integrated graph generation

        Args:
            num_features: Number of input features
            hidden_channels: Size of hidden layers
            feature_sim_threshold: Similarity threshold for graph edges
            time_threshold: Time window for connecting nodes (seconds)
            k_nearest: Number of fallback connections
            device: Computation device ('cuda' or 'cpu')
        """
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize graph generator
        self.graph_generator = AttackGraphGenerator(
            feature_sim_threshold=feature_sim_threshold,
            time_threshold=time_threshold,
            k_nearest=k_nearest,
            device=device
        )
        self.optimizer = None
        self.num_classes = None
        self.graph_data = None

        # Model will be initialized during first training
        self.model = None
        self.scaler = torch.cuda.amp.GradScaler()
        self.embeddings = None  # Thêm biến lưu trữ embedding

    # ... (giữ nguyên các phương thức hiện có)
    def prepare_data(self, df, feature_cols=None, timestamp_col='timestamp', label_col='label'):
        """
        Prepare graph data from DataFrame

        Args:
            df: Input DataFrame
            feature_cols: List of feature columns (if None, use first num_features columns)
            timestamp_col: Name of timestamp column
            label_col: Name of label column
        """
        if feature_cols is None:
            features = df.iloc[:, :self.num_features].values
        else:
            features = df[feature_cols].values

        timestamps = df[timestamp_col].values
        labels = df[label_col].values

        # Generate graph
        self.graph_data = self.graph_generator.generate_graph(
            features=features,
            timestamps=timestamps,
            labels=labels
        )

        return self.graph_data

    def init_model(self):
        """Initialize the GNN model"""
        if self.graph_data is None:
            raise ValueError("No graph data available. Call prepare_data() first.")

        self.num_classes = len(self.graph_generator.label_encoder.classes_)

        self.model = GNNModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
            hidden_channels=self.hidden_channels
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=5e-4
        )

    def train(self, epochs=100, eval_every=10):
        """
        Train the GNN model

        Args:
            epochs: Number of training epochs
            eval_every: Evaluate model every n epochs
        """
        if self.model is None:
            self.init_model()

        if self.graph_data is None:
            raise ValueError("No data prepared. Call prepare_data() first.")

        data = self.graph_data.to(self.device)
        history = {'loss': [], 'accuracy': []}

        for epoch in tqdm(range(1, epochs+1)):
            self.model.train()
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = self.model(data.x, data.edge_index, data.edge_attr)
                loss = F.nll_loss(out, data.y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            history['loss'].append(loss.item())

            if epoch % eval_every == 0:
                acc = self.evaluate()
                history['accuracy'].append(acc)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

        return history

    def evaluate(self):
        """Evaluate model performance"""
        if self.model is None or self.graph_data is None:
            raise ValueError("Model or data not initialized")

        self.model.eval()
        data = self.graph_data.to(self.device)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_attr)
            pred = out.argmax(dim=1)
            correct = (pred == data.y).sum().item()
            acc = correct / len(data.y)

        return acc

    def classification_report(self):
        """Generate detailed classification report"""
        if self.model is None or self.graph_data is None:
            raise ValueError("Model or data not initialized")

        self.model.eval()
        data = self.graph_data.to(self.device)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_attr)
            pred = out.argmax(dim=1)

        return classification_report(
            data.y.cpu().numpy(),
            pred.cpu().numpy(),
            target_names=self.graph_generator.label_encoder.classes_
        )

    def predict(self, new_df):
        """
        Make predictions on new data

        Args:
            new_df: DataFrame with same structure as training data

        Returns:
            Predicted labels (original class names)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Generate graph for new data
        new_graph = self.graph_generator.generate_graph(
            features=new_df.iloc[:, :self.num_features].values,
            timestamps=new_df['timestamp'].values,
            labels=None  # No labels for prediction
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(new_graph.x, new_graph.edge_index, new_graph.edge_attr)
            pred = out.argmax(dim=1)

        return self.graph_generator.label_encoder.inverse_transform(pred.cpu().numpy())
    def get_embeddings(self, data=None, layer: str = 'last'):
        """
        Trích xuất embedding từ các layer của GNN

        Args:
            data: PyG Data object (nếu None sẽ dùng graph_data)
            layer: Layer để trích xuất ('conv1', 'conv2', 'conv3', 'last')

        Returns:
            Numpy array chứa các embedding
        """
        if self.model is None:
            raise ValueError("Model chưa được khởi tạo. Hãy train model trước.")

        if data is None:
            if self.graph_data is None:
                raise ValueError("Không có dữ liệu đồ thị. Hãy chuẩn bị dữ liệu trước.")
            data = self.graph_data.to(self.device)

        self.model.eval()
        with torch.no_grad():
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            # Lấy embedding từ các layer khác nhau
            if layer == 'conv1':
                x = self.model.conv1(x, edge_index, edge_attr=edge_attr)
                embeddings = F.leaky_relu(x).cpu().numpy()
            elif layer == 'conv2':
                x = self.model.conv1(x, edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)
                x = self.model.conv2(x, edge_index, edge_attr=edge_attr)
                embeddings = F.leaky_relu(x).cpu().numpy()
            elif layer == 'conv3':
                x = self.model.conv1(x, edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)
                x = self.model.conv2(x, edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)
                x = self.model.conv3(x, edge_index, edge_attr=edge_attr)
                embeddings = F.leaky_relu(x).cpu().numpy()
            else:  # 'last' - layer trước classification
                x = self.model.conv1(x, edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)
                x = self.model.conv2(x, edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)
                x = self.model.conv3(x, edge_index, edge_attr=edge_attr)
                x = F.leaky_relu(x)
                embeddings = self.model.lin1(x).cpu().numpy()

        self.embeddings = embeddings
        return embeddings

    def visualize_embeddings(self, embeddings=None, labels=None, method='umap', **kwargs):
        """
        Trực quan hóa embedding bằng UMAP hoặc t-SNE

        Args:
            embeddings: Embedding để visualize (nếu None dùng embedding hiện có)
            labels: Nhãn để hiển thị (nếu None dùng nhãn từ graph_data)
            method: 'umap' hoặc 'tsne'
            **kwargs: Các tham số cho UMAP/t-SNE
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("Không có embedding nào được lưu. Hãy gọi get_embeddings() trước.")
            embeddings = self.embeddings

        if labels is None:
            if self.graph_data is None or not hasattr(self.graph_data, 'y'):
                raise ValueError("Không có nhãn để hiển thị")
            labels = self.graph_data.y.cpu().numpy()
            label_names = self.graph_generator.label_encoder.inverse_transform(labels)
        else:
            label_names = labels

        # Giảm chiều embedding
        if method == 'umap':
            reducer = umap.UMAP(**kwargs)
        elif method == 'tsne':
            reducer = TSNE(**kwargs)
        else:
            raise ValueError("Phương thức phải là 'umap' hoặc 'tsne'")

        embedding_2d = reducer.fit_transform(embeddings)

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embedding_2d[:, 0], y=embedding_2d[:, 1],
            hue=label_names, palette='viridis', alpha=0.8
        )
        plt.title(f'Embedding visualization ({method.upper()})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def save_embeddings(self, file_path, embeddings=None):
        """
        Lưu embedding ra file

        Args:
            file_path: Đường dẫn file để lưu (.npy hoặc .pkl)
            embeddings: Embedding để lưu (nếu None dùng embedding hiện có)
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("Không có embedding nào được lưu. Hãy gọi get_embeddings() trước.")
            embeddings = self.embeddings

        if file_path.endswith('.npy'):
            np.save(file_path, embeddings)
        elif file_path.endswith('.pkl'):
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(embeddings, f)
        else:
            raise ValueError("Định dạng file phải là .npy hoặc .pkl")

        print(f"Embedding đã được lưu tại {file_path}")

    def load_embeddings(self, file_path):
        """
        Tải embedding từ file

        Args:
            file_path: Đường dẫn file chứa embedding
        """
        if file_path.endswith('.npy'):
            embeddings = np.load(file_path)
        elif file_path.endswith('.pkl'):
            import pickle
            with open(file_path, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            raise ValueError("Định dạng file phải là .npy hoặc .pkl")

        self.embeddings = embeddings
        return embeddings