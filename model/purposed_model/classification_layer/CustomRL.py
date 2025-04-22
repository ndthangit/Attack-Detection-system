import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from purposed_model.classification_layer.CustomCallback import CustomCallback
from purposed_model.classification_layer.CustomGymEnv import CustomGymEnv


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