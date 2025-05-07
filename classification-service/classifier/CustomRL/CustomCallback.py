from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):
    """Callback tùy chỉnh để theo dõi quá trình huấn luyện"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_accuracies = []

    def _on_step(self) -> bool:
        # Lấy thông tin từ info dict
        if "true_label" in self.locals.get('infos', [{}])[0]:
            action = self.locals['actions'][0]
            true_label = self.locals['infos'][0]["true_label"]
            accuracy = float(action == true_label)
            self.episode_accuracies.append(accuracy)

        return True

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()

        if len(self.model.ep_info_buffer) > 0:
            avg_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.episode_rewards.append(avg_reward)

    def get_last_accuracy(self) -> float:
        """Trả về độ chính xác của episode gần nhất"""
        return self.episode_accuracies[-1] if len(self.episode_accuracies) > 0 else 0.0

    def get_last_reward(self) -> float:
        """Trả về reward trung bình của episode gần nhất"""
        return self.episode_rewards[-1] if len(self.episode_rewards) > 0 else 0.0

