import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
from stable_baselines3 import DQN
from purposed_model.classification_layer.CustomGymEnv import CustomGymEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from purposed_model.classification_layer.CustomCallback import CustomCallback
from torch.utils.data import Dataset, DataLoader, Subset
from purposed_model.classification_layer.TLSTMGenerator import TLSTMGenerator
from purposed_model.classification_layer.Discriminator import Discriminator

def train_with_sb3(generator, dataloader_mlp, num_episodes, device, batch_size_mlp, learning_rate_mlp):
    generator.eval()

    # Tạo môi trường
    env = CustomGymEnv(generator, dataloader_mlp, device)
    env = DummyVecEnv([lambda: env])

    # Tính toán total timesteps
    total_timesteps = num_episodes * len(dataloader_mlp) * batch_size_mlp

    # Khởi tạo DQN với các tham số
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate_mlp,
        buffer_size=10000,
        learning_starts=batch_size_mlp,
        batch_size=batch_size_mlp,
        tau=1.0,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.3,
        verbose=0,
        device=device
    )

    # Callback
    callbacks = CustomCallback()

    # Huấn luyện
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10
    )

    # Tính toán accuracy cuối cùng
    final_accuracy = np.mean(callbacks.episode_accuracies) if callbacks.episode_accuracies else 0

    # In kết quả
    print(f"Final Avg Reward: {np.mean(callbacks.episode_rewards):.4f}")
    print(f"Final Accuracy: {final_accuracy:.4f}")

    return model


def evaluate_metrics(generator, sb3_model, dataloader, device):
    """
    Đánh giá model DQN của Stable-Baselines3 với generator

    Args:
        generator: Mô hình generator đã được huấn luyện
        sb3_model: Model DQN từ Stable-Baselines3
        dataloader: DataLoader chứa dữ liệu test
        device: Thiết bị tính toán (cuda/cpu)

    Returns:
        Dictionary chứa các metrics và dự đoán
    """
    generator.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Xử lý batch data
            if len(batch) == 3:
                features, delta_t, labels = batch
                features = features.to(device)
                delta_t = delta_t.to(device)
                labels = labels.to(device)
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                delta_t = None

            # Tạo latent representations
            latent = generator(features, delta_t)[1] if delta_t is not None else generator(features)

            # Chuyển latent sang numpy array cho SB3
            obs = latent.cpu().numpy()

            # Dự đoán với SB3 model
            batch_preds = []
            for obs_i in obs:
                action, _ = sb3_model.predict(obs_i, deterministic=True)
                batch_preds.append(action)

            predictions.extend(batch_preds)
            true_labels.extend(labels[:, -1].cpu().numpy() if labels.dim() > 1 else labels.cpu().numpy())

    # Tính toán metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(true_labels, predictions)

    # return {
    #     'accuracy': float(accuracy),
    #     'precision': float(precision),
    #     'recall': float(recall),
    #     'f1': float(f1),
    #     'predictions': np.array(predictions),
    #     'true_labels': np.array(true_labels)
    # }
    return float(accuracy), float(precision), float(recall), float(f1)

def train_gan(generator: TLSTMGenerator , discriminator : Discriminator, dataloader_g: DataLoader, g_optimizer, d_optimizer,
                       criterion, scaler, device):
    """
        Train GAN for one epoch

        Args:
            generator: Generator model
            discriminator: Discriminator model
            dataloader_g: DataLoader for generator training
            g_optimizer: Optimizer for generator
            d_optimizer: Optimizer for discriminator
            criterion: Loss function
            scaler: GradScaler for AMP
            device: Device to train on

        Returns:
            avg_d_loss: Average discriminator loss for the epoch
            avg_g_loss: Average generator loss for the epoch
        """
    generator.train()
    discriminator.train()
    total_d_loss = 0
    total_g_loss = 0
    num_batches = 0

    for batch_features, batch_delta_t, _ in dataloader_g:
        # Chuyển dữ liệu lên GPU
        batch_features = batch_features.to(device, non_blocking=True)
        batch_delta_t = batch_delta_t.to(device, non_blocking=True)
        current_batch_size = batch_features.size(0)

        real_label = torch.ones(current_batch_size, 1, device=device)
        fake_label = torch.zeros(current_batch_size, 1, device=device)

        # Sử dụng AMP với cú pháp mới
        with torch.amp.autocast('cuda'):
            # Huấn luyện Discriminator
            d_optimizer.zero_grad(set_to_none=True)
            real_output = discriminator(batch_features)
            d_real_loss = criterion(real_output, real_label)

            fake_data, _ = generator(batch_features, batch_delta_t)
            fake_output = discriminator(fake_data.detach())
            d_fake_loss = criterion(fake_output, fake_label)

            d_loss = d_real_loss + d_fake_loss

        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)
        scaler.update()

        # Huấn luyện Generator
        with torch.amp.autocast('cuda'):
            g_optimizer.zero_grad(set_to_none=True)
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_label)
            reconstruction_loss = nn.MSELoss()(fake_data, batch_features)
            g_total_loss = g_loss + reconstruction_loss

        scaler.scale(g_total_loss).backward()
        scaler.step(g_optimizer)
        scaler.update()

        # Chuyển loss về CPU để tính trung bình
        total_d_loss += d_loss.item()
        total_g_loss += g_total_loss.item()
        num_batches += 1

    avg_d_loss = total_d_loss / num_batches
    avg_g_loss = total_g_loss / num_batches

    return avg_d_loss, avg_g_loss