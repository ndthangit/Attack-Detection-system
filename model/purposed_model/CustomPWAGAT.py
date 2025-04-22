import pandas as pd
import torch
import torch.nn as nn
from purposed_model.classification_layer.CustomGAN import CustomGAN
from purposed_model.classification_layer.CustomRL import CustomRL
from sklearn.preprocessing import LabelEncoder
from purposed_model.graph_embedding_layer.AttackBehaviorGNN import AttackBehaviorGNN

class CustomPWAGAT:

    def __init__(self, gnn_system: AttackBehaviorGNN = None, gan_system: CustomGAN =None, rl_system: CustomRL = None, seq_len= 10):

        self.gnn_system = gnn_system
        self.model_name = "Custom_PWAGAT"
        self.num_classes = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gan_system = gan_system
        self.rl_system = rl_system
        self.label_encoder = LabelEncoder()
        self.seq_len = seq_len


        # Theo dõi quá trình huấn luyện
        self.training_history = {
            'gan_loss': [],
            'rl_rewards': [],
            'val_metrics': []
        }

    # graph embedding layer

    def create_graph_layer(self, **kwargs):
         self.gnn_system = AttackBehaviorGNN(**kwargs)

    def get_graph_layer(self):
        return self.gnn_system

    "classification layer"

    def create_gan_model(self, **kwargs):
        self.gan_system = CustomGAN(**kwargs)
        self.gan_system.info()

    def create_rl_agent(self, **kwargs):
        self.rl_system = CustomRL(self.gan_system.generator, self.num_classes, **kwargs)

    def fit(self, data_train_x: pd.DataFrame, data_timestamps: pd.DataFrame,
            data_train_y: pd.DataFrame, val_data=None):
        """
        Huấn luyện toàn bộ mô hình bao gồm GAN và RL agent

        Args:
            data_train_x: Dữ liệu huấn luyện (văn bản)
            data_timestamps: Timestamp tương ứng
            data_train_y: Nhãn huấn luyện
            val_data: Tuple (val_x, val_timestamps, val_y) cho validation
        """
        if self.gan_system is None:
            raise ValueError("GAN system chưa được khởi tạo. Gọi create_gan_model() trước.")
        self.num_classes = len(data_train_y.unique())
        # 1. Tiền xử lý dữ liệu
        features, labels, delta_t = self.gan_system.preprocess_data(
            data_train_x, data_timestamps, data_train_y
        )

        # 2. Tạo dataset và dataloader
        train_loader = self.gan_system.create_dataloader(features, labels, delta_t)

        # 3. Huấn luyện GAN
        self.gan_system.train(train_loader)

        # 4. Huấn luyện RL agent
        # self.rl_system.train(train_loader)

        # 5. Validation (nếu có dữ liệu validation)
        if val_data is not None:
            val_x, val_timestamps, val_y = val_data
            val_features, val_labels, val_delta_t = self.gan_system.preprocess_data(
                val_x, val_timestamps, val_y
            )
            val_loader = self.gan_system.create_dataloader(val_features, val_labels, val_delta_t)
            self.gan_system.validate(val_loader)

    def _validate(self, val_loader):
        """
        Thực hiện đánh giá trên tập validation
        """
        self.gan_system.generator.eval()
        self.gan_system.discriminator.eval()

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch_features, batch_delta_t, batch_labels in val_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_delta_t = batch_delta_t.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)

                # Tính toán dữ liệu giả
                fake_data, _ = self.gan_system.generator(batch_features, batch_delta_t)

                # Tính loss
                reconstruction_loss = nn.MSELoss()(fake_data, batch_features)
                total_loss += reconstruction_loss.item() * batch_features.size(0)
                total_samples += batch_features.size(0)

        avg_loss = total_loss / total_samples
        print(f"Validation Loss: {avg_loss:.4f}")


    def predict(self, data_extract: pd.DataFrame):
        # Implement the prediction logic here
        pass

    def load_model(self, model_path):
        # Implement the logic to load the model
        pass