import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from purposed_model.classification_layer.CustomCallback import CustomCallback
from purposed_model.classification_layer.TLSTMGenerator import TLSTMGenerator
from purposed_model.classification_layer.Discriminator import Discriminator
from sklearn.preprocessing import LabelEncoder
from purposed_model.classification_layer.SequenceDataset import SequenceDataset
from stable_baselines3 import DQN
from purposed_model.classification_layer.CustomGymEnv import CustomGymEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader


class custom_PWAGAT:

    def __init__(self,input_size = 768, hidden_size_g = 31, hidden_size_d = 42, dropout_rate_g = 0.42,
                 dropout_rate_d = 0.32,learning_rate_g = 0.008,learning_rate_d = 0.009,num_epochs_g = 25,
                 hidden_size_mlp = 53 ,dropout_rate_mlp = 0.31,learning_rate_mlp = 0.005,num_episodes = 25,
                 batch_size_mlp = 74, seq_len = 10):
        self.model_name = "custom_PWAGAT_model"
        self.num_classes = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size_g = hidden_size_g
        self.hidden_size_d = hidden_size_d
        self.latent_size = 128
        self.output_size = 768
        self.dropout_rate_g = dropout_rate_g
        self.dropout_rate_d = dropout_rate_d
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.num_epochs_g = num_epochs_g  # 152


        self.hidden_size_mlp = hidden_size_mlp
        self.dropout_rate_mlp = dropout_rate_mlp
        self.learning_rate_mlp = learning_rate_mlp
        self.num_episodes = num_episodes  # 245
        self.batch_size_mlp = batch_size_mlp  # Đồng bộ với giá trị đã xác định trước đó
        self.batch_size_rl = self.batch_size_mlp  # Đồng bộ batch_size_rl với batch_size_mlp

        # Khởi tạo các mô hình
        self._init_models()

        # Theo dõi quá trình huấn luyện
        self.training_history = {
            'gan_loss': [],
            'rl_rewards': [],
            'val_metrics': []
        }

        self.seq_len = seq_len
    def _init_models(self):
        """Khởi tạo các mô hình và chuyển lên device"""
        self.generator = TLSTMGenerator(
            self.input_size, self.hidden_size_g, self.latent_size,
            self.output_size, dropout_rate=self.dropout_rate_g
        ).to(self.device)

        self.discriminator = Discriminator(
            self.input_size, self.hidden_size_d,
            dropout_rate=self.dropout_rate_d
        ).to(self.device)

        self.rl_agent = None  # Sẽ được khởi tạo trong quá trình huấn luyện

        self.label_encoder = LabelEncoder()

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
        self.num_classes = len(data_train_y.unique())
        # 1. Tiền xử lý dữ liệu
        features, labels, delta_t = self._preprocess_data(
            data_train_x, data_timestamps, data_train_y
        )

        # 2. Tạo dataset và dataloader
        train_dataset = SequenceDataset(features, labels, delta_t, self.seq_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size_mlp,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        # 3. Huấn luyện GAN
        self._train_gan(train_loader)

        # 4. Huấn luyện RL agent
        self._train_rl_agent(train_loader)

        # 5. Validation (nếu có dữ liệu validation)
        if val_data is not None:
            val_x, val_timestamps, val_y = val_data
            val_features, val_labels, val_delta_t = self._preprocess_data(
                val_x, val_timestamps, val_y
            )
            val_dataset = SequenceDataset(val_features, val_labels, val_delta_t, self.seq_len)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_mlp,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            self._validate(val_loader)

    def _validate(self, val_loader):
        """
        Thực hiện đánh giá trên tập validation
        """
        self.generator.eval()
        self.discriminator.eval()

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch_features, batch_delta_t, batch_labels in val_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_delta_t = batch_delta_t.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)

                # Tính toán dữ liệu giả
                fake_data, _ = self.generator(batch_features, batch_delta_t)

                # Tính loss
                reconstruction_loss = nn.MSELoss()(fake_data, batch_features)
                total_loss += reconstruction_loss.item() * batch_features.size(0)
                total_samples += batch_features.size(0)

        avg_loss = total_loss / total_samples
        print(f"Validation Loss: {avg_loss:.4f}")

    def _preprocess_data(self, data_x, timestamps, labels):
        """Tiền xử lý dữ liệu đầu vào"""
        # Encode nhãn
        label_encoder = LabelEncoder()
        if not hasattr(label_encoder, 'classes_'):
            label_encoder.fit(np.unique(labels))
        labels = label_encoder.transform(labels)

        # Tính delta time
        timestamps = timestamps.astype(float)
        delta_t = np.diff(timestamps, prepend=timestamps[0])

        # Chuyển đổi thành tensor
        if not isinstance(data_x, torch.Tensor):
            data_x = torch.tensor(data_x, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        if not isinstance(delta_t, torch.Tensor):
            delta_t = torch.tensor(delta_t, dtype=torch.float32).unsqueeze(-1)

        return data_x, labels, delta_t

    def _train_gan(self, dataloader):
        """Huấn luyện GAN"""
        # Khởi tạo optimizer và loss
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate_g,
            betas=(0.5, 0.999)
        )

        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate_d,
            betas=(0.5, 0.999)
        )

        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        print("Bắt đầu huấn luyện GAN...")
        for epoch in range(self.num_epochs_g):
            self.generator.train()
            self.discriminator.train()

            total_d_loss = 0
            total_g_loss = 0
            num_batches = 0

            for batch_features, batch_delta_t, _ in dataloader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_delta_t = batch_delta_t.to(self.device, non_blocking=True)
                current_batch_size = batch_features.size(0)

                real_label = torch.ones(current_batch_size, 1, device=self.device)
                fake_label = torch.zeros(current_batch_size, 1, device=self.device)

                # Huấn luyện Discriminator
                with torch.cuda.amp.autocast():
                    d_optimizer.zero_grad(set_to_none=True)

                    # Dữ liệu thật
                    real_output = self.discriminator(batch_features)
                    d_real_loss = criterion(real_output, real_label)

                    # Dữ liệu giả
                    fake_data, _ = self.generator(batch_features, batch_delta_t)
                    fake_output = self.discriminator(fake_data.detach())
                    d_fake_loss = criterion(fake_output, fake_label)

                    d_loss = (d_real_loss + d_fake_loss) / 2

                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)

                # Huấn luyện Generator
                with torch.cuda.amp.autocast():
                    g_optimizer.zero_grad(set_to_none=True)

                    fake_output = self.discriminator(fake_data)
                    g_loss = criterion(fake_output, real_label)
                    reconstruction_loss = nn.MSELoss()(fake_data, batch_features)
                    g_total_loss = g_loss + 0.5 * reconstruction_loss  # Thêm hệ số cân bằng

                scaler.scale(g_total_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()

                total_d_loss += d_loss.item()
                total_g_loss += g_total_loss.item()
                num_batches += 1

            avg_d_loss = total_d_loss / num_batches
            avg_g_loss = total_g_loss / num_batches
            self.training_history['gan_loss'].append((avg_d_loss, avg_g_loss))

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs_g}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

    def _train_rl_agent(self, dataloader):
        """Huấn luyện RL agent"""
        print("Bắt đầu huấn luyện RL agent...")

        # Tạo môi trường
        env = CustomGymEnv(self.generator, dataloader, self.device,self.latent_size, self.num_classes)
        env = DummyVecEnv([lambda: env])

        # Tính toán total timesteps
        total_timesteps = self.num_episodes * len(dataloader) * self.batch_size_mlp

        # Khởi tạo DQN
        self.rl_agent = DQN(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate_mlp,
            buffer_size=10000,
            learning_starts=self.batch_size_mlp,
            batch_size=self.batch_size_mlp,
            tau=1.0,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            exploration_fraction=0.3,
            verbose=0,
            device=self.device
        )

        # Callback để theo dõi quá trình huấn luyện
        callbacks = CustomCallback()

        # Huấn luyện
        self.rl_agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10
        )

        # Lưu lại kết quả huấn luyện
        self.training_history['rl_rewards'].extend(callbacks.episode_rewards)

        # In kết quả
        final_accuracy = np.mean(callbacks.episode_accuracies) if callbacks.episode_accuracies else 0
        print(f"Final Avg Reward: {np.mean(callbacks.episode_rewards):.4f}")
        print(f"Final Accuracy: {final_accuracy:.4f}")

    def predict(self, data_extract: pd.DataFrame):
        # Implement the prediction logic here
        pass

    def load_model(self, model_path):
        # Implement the logic to load the model
        pass