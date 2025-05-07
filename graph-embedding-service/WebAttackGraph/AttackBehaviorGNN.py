import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report

from WebAttackGraph.AttackGraphGenerator import AttackGraphGenerator
from WebAttackGraph.GNNModel import GNNModel
from core.config import GraphConfig


class AttackBehaviorGNN:
    def __init__(self,
                 num_features=768,
                 hidden_channels=256,
                 feature_sim_threshold=0.7,
                 time_threshold=60,
                 k_nearest=5,
                 seq_len=10,
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
        self.seq_len = seq_len

        # Model will be initialized during first training
        self.model = None
        self.scaler = torch.cuda.amp.GradScaler()
        self.embeddings = None  # Thêm biến lưu trữ embedding

    def prepare_data(self, df, feature_cols=None, timestamp_col='timestamps', label_col='label'):
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
        self.num_classes = len(np.unique(labels))

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

    def get_embeddings(self, data=None, layer: str = 'last'):
        """
        Trích xuất embedding từ các layer của GNN

        Args:
            data: PyG Data object (nếu None sẽ dùng graph_data)
            layer: Layer để trích xuất ('conv1', 'conv2', 'conv3', 'last')

        Returns:
            Numpy array chứa các embedding

        Raises:
            ValueError: Nếu model, data không được khởi tạo hoặc layer không hợp lệ
        """
        if self.model is None:
            raise ValueError("Model chưa được khởi tạo. Hãy train model trước.")

        # Sử dụng graph_data nếu data không được cung cấp
        if data is None:
            if self.graph_data is None:
                raise ValueError("Không có dữ liệu đồ thị. Hãy chuẩn bị dữ liệu trước.")
            data = self.graph_data

        # Kiểm tra dữ liệu đầu vào
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'edge_attr'):
            raise ValueError("Dữ liệu đầu vào phải chứa x, edge_index và edge_attr.")

        # Chuyển dữ liệu sang thiết bị phù hợp
        data = data.to(self.device)

        self.model.eval()
        with torch.no_grad():
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            # Tùy chọn layer
            if layer == 'last':
                # Sử dụng hàm forward để lấy embedding trực tiếp
                _, embeddings = self.model(x, edge_index, edge_attr, return_embeddings=True)
                embeddings = embeddings.cpu().numpy()
            else:
                # Xử lý các layer convolution
                if layer == 'conv1':
                    x = self.model.conv1(x, edge_index, edge_attr=edge_attr)
                elif layer == 'conv2':
                    x = self.model.conv1(x, edge_index, edge_attr=edge_attr)
                    x = F.leaky_relu(x)
                    x = self.model.conv2(x, edge_index, edge_attr=edge_attr)
                elif layer == 'conv3':
                    x = self.model.conv1(x, edge_index, edge_attr=edge_attr)
                    x = F.leaky_relu(x)
                    x = self.model.conv2(x, edge_index, edge_attr=edge_attr)
                    x = F.leaky_relu(x)
                    x = self.model.conv3(x, edge_index, edge_attr=edge_attr)
                else:
                    raise ValueError("Layer phải là 'conv1', 'conv2', 'conv3' hoặc 'last'")
                embeddings = F.leaky_relu(x).cpu().numpy()

        self.embeddings = embeddings
        return embeddings


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

    def refresh(self):
        """
        Reset the model and related attributes to their initial state
        """
        # Reset model and optimizer
        self.model = None
        self.optimizer = None

        # Reset graph data and embeddings
        self.graph_data = None
        self.embeddings = None

        # Reinitialize graph generator with initial parameters
        self.graph_generator = AttackGraphGenerator(
            feature_sim_threshold=self.graph_generator.feature_sim_threshold,
            time_threshold=self.graph_generator.time_threshold,
            k_nearest=self.graph_generator.k_nearest,
            device= self.device
        )

        # Reset scaler
        self.scaler = torch.cuda.amp.GradScaler()

        print("Model has been reset to initial state")


graphParameter = GraphConfig()
model = AttackBehaviorGNN(**graphParameter.creator.__dict__)
