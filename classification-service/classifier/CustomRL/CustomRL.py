import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader

from classifier.CustomRL.CustomCallback import CustomCallback
from classifier.CustomRL.CustomGymEnv import CustomGymEnv
from modelData.SequenceDataset import SequenceDataset


class CustomRL:
    def __init__(self, generator, num_classes=None, latent_size=128,
                 learning_rate_mlp=0.005, num_episodes=25, batch_size=74):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.learning_rate_mlp = learning_rate_mlp
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.rl_agent = None
        self.training_history = {'rl_rewards': []}
        self.info()

    def info(self):
        """Display model information"""
        print("===== RL Model  =====")
        print(f"Generator: {self.generator}")
        print(f"Latent Size: {self.latent_size}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"MLP Learning Rate: {self.learning_rate_mlp}")
        print(f"Number of Episodes: {self.num_episodes}")
        print(f"Batch Size: {self.batch_size}")
        print("===============================")

    def train(self, dataloader):
        """Huấn luyện RL agent"""
        print("Bắt đầu huấn luyện RL agent...")

        # Tạo môi trường
        env = CustomGymEnv(self.generator, dataloader, self.device, self.latent_size, self.num_classes)
        env = DummyVecEnv([lambda: env])

        # Tính toán total timesteps
        total_timesteps = self.num_episodes * len(dataloader) * self.batch_size

        # Khởi tạo DQN
        self.rl_agent = DQN(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate_mlp,
            buffer_size=10000,
            learning_starts=self.batch_size,
            batch_size=self.batch_size,
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

    def predict(self, states):
        """
        Dự đoán hành động từ trạng thái sử dụng RL agent.

        Args:
            states (torch.Tensor): Trạng thái đầu vào (latent features từ generator).

        Returns:
            np.ndarray: Hành động dự đoán (nhãn lớp).
        """
        if self.rl_agent is None:
            raise ValueError("RL agent chưa được huấn luyện. Gọi train() trước.")

        states = states.to(self.device)
        actions, _ = self.rl_agent.predict(states.cpu().numpy(), deterministic=True)
        return actions

    def create_dataset(self, features, delta_t, labels, seq_len):
        """
        Tạo dataset cho RL từ dữ liệu đã tiền xử lý.

        Args:
            features (torch.Tensor): Đặc trưng đầu vào.
            delta_t (torch.Tensor): Delta time.
            labels (torch.Tensor): Nhãn.
            seq_len (int): Độ dài chuỗi.

        Returns:
            SequenceDataset: Dataset cho RL.
        """
        return SequenceDataset(features, labels, delta_t, seq_len)

    def create_dataloader(self, dataset, batch_size=None, shuffle=True):
        """
        Tạo DataLoader từ dataset.

        Args:
            dataset (SequenceDataset): Dataset đã tạo.
            batch_size (int, optional): Kích thước batch. Nếu None, sử dụng self.batch_size.
            shuffle (bool): Có xáo trộn dữ liệu không.

        Returns:
            DataLoader: DataLoader cho RL.
        """
        batch_size = batch_size or self.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True
        )

    @property
    def policy_net(self):
        """Trả về policy network của DQN."""
        if self.rl_agent is None:
            raise ValueError("RL agent chưa được huấn luyện.")
        return self.rl_agent.policy