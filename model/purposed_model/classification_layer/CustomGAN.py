import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

from purposed_model.classification_layer.Discriminator import Discriminator
from purposed_model.classification_layer.SequenceDataset import SequenceDataset
from purposed_model.classification_layer.TLSTMGenerator import TLSTMGenerator


class CustomGAN:
    def __init__(self, input_size=768, hidden_size_g=31, hidden_size_d=42, dropout_rate_g=0.42,
                 dropout_rate_d=0.32, learning_rate_g=0.008, learning_rate_d=0.009, num_epochs_g=25,
                 batch_size=74, seq_len=10):

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
        self.num_epochs_g = num_epochs_g
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.generator = TLSTMGenerator(
            self.input_size, self.hidden_size_g, self.latent_size,
            self.output_size, dropout_rate=self.dropout_rate_g
        ).to(self.device)

        self.discriminator = Discriminator(
            self.input_size, self.hidden_size_d,
            dropout_rate=self.dropout_rate_d
        ).to(self.device)

        self.label_encoder = LabelEncoder()
        self.training_history = {'gan_loss': []}

    def train(self, dataloader):
        """Huấn luyện GAN"""
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

    def validate(self, val_loader):
        """Thực hiện đánh giá trên tập validation"""
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
        return avg_loss

    def preprocess_data(self, data_x, timestamps, labels):
        """Tiền xử lý dữ liệu đầu vào"""
        # Encode nhãn
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(np.unique(labels))
        labels = self.label_encoder.transform(labels)

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

    def create_dataloader(self, features, labels, delta_t):
        """Tạo dataloader từ dữ liệu đã tiền xử lý"""
        dataset = SequenceDataset(features, labels, delta_t, self.seq_len)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )