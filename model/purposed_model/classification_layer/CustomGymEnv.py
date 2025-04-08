import gym
from gym import spaces
import torch


class CustomGymEnv(gym.Env):
    """Môi trường tùy chỉnh cho bài toán của bạn"""

    def __init__(self, generator, dataloader, device):
        super().__init__()
        self.generator = generator
        self.dataloader = dataloader
        self.device = device
        self.iterator = iter(dataloader)
        self.current_batch = None
        self.current_idx = 0

        # Định nghĩa observation và action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(latent_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_classes)

    def reset(self):
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

            return latent[0].cpu().numpy()
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return self.reset()

    def step(self, action):
        reward = 1.0 if action == self.current_labels[self.current_idx] else -0.5
        done = True  # Mỗi step là một episode ngắn

        info = {"true_label": self.current_labels[self.current_idx]}

        self.current_idx += 1
        if self.current_idx >= len(self.current_latent):
            next_state = self.reset()
        else:
            next_state = self.current_latent[self.current_idx].cpu().numpy()

        return next_state, reward, done, info