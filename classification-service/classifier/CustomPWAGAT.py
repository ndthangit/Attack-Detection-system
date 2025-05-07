import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from classifier.CustomGAN.CustomGAN import CustomGAN
from classifier.CustomRL.CustomRL import CustomRL


class CustomPWAGAT:

    def __init__(self, gan_system: CustomGAN = None, rl_system: CustomRL = None, seq_len=10):
        self.model_name = "Custom_PWAGAT"
        self.num_classes = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan_system = gan_system
        self.rl_system = rl_system
        self.seq_len = seq_len

        # Theo dõi quá trình huấn luyện
        self.training_history = {
            'gan_loss': [],
            'rl_rewards': [],
            'val_metrics': []
        }

    # classification layer
    def create_gan_model(self, **kwargs):
        self.gan_system = CustomGAN(**kwargs)

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
        self.rl_system.num_classes = self.num_classes
        self.rl_system.info()
        # 1. Tiền xử lý dữ liệu
        features, labels, delta_t = self.gan_system.preprocess_data(
            data_train_x, data_timestamps, data_train_y
        )

        # 2. Tạo dataset và dataloader
        train_loader = self.gan_system.create_dataloader(features, labels, delta_t)

        # 3. Huấn luyện GAN
        self.gan_system.train(train_loader)
        # 4. Huấn luyện RL agent
        self.rl_system.train(train_loader)

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

    def predict(self, data_extract: pd.DataFrame, timestamps: pd.DataFrame = None):
        """
        Dự đoán nhãn cho dữ liệu đầu vào sử dụng mô hình CustomPWAGAT.

        Args:
            data_extract (pd.DataFrame): Dữ liệu đầu vào (đặc trưng hoặc văn bản).
            timestamps (pd.DataFrame, optional): Dữ liệu timestamps tương ứng. Nếu None, giả lập delta_t = 1.

        Returns:
            np.ndarray: Nhãn dự đoán.
        """
        if self.gan_system is None or self.rl_system is None:
            raise ValueError(
                "GAN hoặc RL system chưa được khởi tạo. Gọi create_gan_model() và create_rl_agent() trước.")

        # 1. Tiền xử lý dữ liệu
        # Giả lập timestamps nếu không được cung cấp
        if timestamps is None:
            timestamps = pd.DataFrame(np.ones(len(data_extract)), columns=['Timestamps'])

        # Giả lập nhãn tạm thời (sẽ không dùng trong dự đoán)
        dummy_labels = pd.Series(np.zeros(len(data_extract)), name='Label')

        # Tiền xử lý dữ liệu với hàm của GAN system
        features, _, delta_t = self.gan_system.preprocess_data(data_extract, timestamps, dummy_labels)

        # 2. Tạo dataset và dataloader
        dataset = self.gan_system.create_dataset(features, delta_t, torch.zeros(len(features), dtype=torch.long),
                                                 self.seq_len)
        dataloader = self.gan_system.create_dataloader(dataset, batch_size=32, shuffle=False)

        # 3. Dự đoán
        self.gan_system.generator.eval()
        self.rl_system.policy_net.eval()
        all_predictions = []

        with torch.no_grad():
            for batch_features, batch_delta_t, _ in dataloader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_delta_t = batch_delta_t.to(self.device, non_blocking=True)

                # Trích xuất đặc trưng từ generator
                _, latent = self.gan_system.generator(batch_features, batch_delta_t)

                # Sử dụng RL agent để dự đoán
                q_values = self.rl_system.policy_net(latent)
                predicted_labels = q_values.argmax(dim=1).cpu().numpy()

                all_predictions.extend(predicted_labels)

        return np.array(all_predictions)

    def load_model(self, model_path: str):
        """
        Tải trọng số mô hình từ file đã lưu.

        Args:
            model_path (str): Đường dẫn đến file chứa trọng số mô hình (định dạng .pth).

        Raises:
            FileNotFoundError: Nếu file mô hình không tồn tại.
            ValueError: Nếu GAN hoặc RL system chưa được khởi tạo.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")

        if self.gan_system is None or self.rl_system is None:
            raise ValueError("GAN hoặc RL system chưa được khởi tạo. Gọi create_gan_model() và create_rl_agent() trước.")

        # Tải checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Tải trọng số cho generator và discriminator của GAN
        self.gan_system.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan_system.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # Tải trọng số cho RL agent
        self.rl_system.policy_net.load_state_dict(checkpoint['rl_policy_state_dict'])

        # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
        self.gan_system.generator.eval()
        self.gan_system.discriminator.eval()
        self.rl_system.policy_net.eval()

        print(f"Đã tải mô hình thành công từ: {model_path}")

    def save_model(self, model_path: str):
        """
        Lưu trọng số mô hình vào file.

        Args:
            model_path (str): Đường dẫn để lưu file mô hình.
        """
        checkpoint = {
            'generator_state_dict': self.gan_system.generator.state_dict(),
            'discriminator_state_dict': self.gan_system.discriminator.state_dict(),
            'rl_policy_state_dict': self.rl_system.policy_net.state_dict(),
        }
        torch.save(checkpoint, model_path)
        print(f"Đã lưu mô hình tại: {model_path}")