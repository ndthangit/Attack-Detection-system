from typing import Any, SupportsFloat

import gymnasium as gym
import torch
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete

class CustomGymEnv(gym.Env):
    """Môi trường tùy chỉnh cho bài toán của bạn"""

    def __init__(self, generator, dataloader, device, latent_size, num_classes):
        super().__init__()
        self.current_labels = None
        self.current_latent = None
        self.generator = generator
        self.dataloader = dataloader
        self.device = device
        self.iterator = iter(dataloader)
        self.current_batch = None
        self.current_idx = 0

        # Định nghĩa observation và action space
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                            shape=(latent_size,), dtype=np.float32)
        self.action_space = Discrete(num_classes)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        # Initialize the random generator if seed is provided
        super().reset(seed=seed)

        try:
            self.current_batch = next(self.iterator)
            batch_features, batch_delta_t, batch_labels = self.current_batch
            batch_features = batch_features.to(self.device)
            batch_delta_t = batch_delta_t.to(self.device)

            with torch.no_grad():
                _, latent = self.generator(batch_features, batch_delta_t)

            self.current_latent = latent
            self.current_labels = batch_labels[:, -1].cpu().numpy()
            self.current_idx = 0

            observation = latent[0].cpu().numpy()
            info = self._get_info()
            return observation, info
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return self.reset(seed=seed, options=options)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Calculate reward
        reward = 1.0 if action == self.current_labels[self.current_idx] else -0.5
        terminated = True  # Mỗi step là một episode ngắn
        truncated = False  # No early termination in this case

        info = {
            "true_label": self.current_labels[self.current_idx],
            **self._get_info()
        }

        self.current_idx += 1
        if self.current_idx >= len(self.current_latent):
            next_observation, next_info = self.reset()
        else:
            next_observation = self.current_latent[self.current_idx].cpu().numpy()
            next_info = info  # Can be updated if needed

        return next_observation, reward, terminated, truncated, next_info

    def _get_info(self) :
        """Lấy thông tin phụ trợ về trạng thái hiện tại"""
        return {
            "current_idx": self.current_idx,
            "batch_size": len(self.current_latent) if self.current_latent is not None else 0
        }