import random
from state import SystemState

class ExplorationController:
    """Randomly samples continuous actions to build a robust RL dataset."""

    def __init__(self, action_interval_frames: int = 5):
        self.action_interval_frames = action_interval_frames
        self.frames_since_last_action = 0

        # Current continuous action values
        self.current_imgsz = 640
        self.current_skip_rate = 0.0

    def should_update(self) -> bool:
        """Check if it's time to sample a new action."""
        if self.frames_since_last_action >= self.action_interval_frames:
            self.frames_since_last_action = 0
            return True
        self.frames_since_last_action += 1
        return False

    def sample_action(self) -> dict:
        """Samples a new random continuous action."""
        # Random float between -1.0 and 1.0
        action_0 = random.uniform(-1.0, 1.0)
        action_1 = random.uniform(-1.0, 1.0)

        # Map action_0 to imgsz [256, 640] rounded to nearest 32
        # -1.0 -> 256, 1.0 -> 640
        normalized_0 = (action_0 + 1.0) / 2.0  # [0.0, 1.0]
        mapped_imgsz = 256 + (normalized_0 * (640 - 256))
        self.current_imgsz = int(round(mapped_imgsz / 32) * 32)

        # Map action_1 to skip probability [0.0, 1.0]
        # Or to an integer frame skip [0, 5]. Let's do probability.
        self.current_skip_rate = (action_1 + 1.0) / 2.0

        return {
            "action_imgsz_raw": action_0,
            "action_skip_raw": action_1,
            "imgsz": self.current_imgsz,
            "skip_prob": self.current_skip_rate
        }

    def compute_reward(self, prev_state: SystemState, action: dict, next_state: SystemState, proxy_accuracy: float) -> float:
        """Computes the reward for the transition.
        R = w1*(Accuracy) + w2*(FPS) - w3*(Latency) - ThermalPenalty
        """
        w_acc = 10.0
        w_fps = 0.5
        w_lat = 0.1

        # Reward for high accuracy and high FPS
        reward = (w_acc * proxy_accuracy) + (w_fps * next_state.fps)
        
        # Penalize high latency
        reward -= (w_lat * next_state.latency_ms)

        # Thermal Penalty (If Temp goes above 80C)
        if next_state.temperature > 80.0:
            reward -= 50.0  # Heavy penalty for overheating
            
        # CPU Penalty (If CPU maxes out)
        if next_state.cpu_percent > 90.0:
            reward -= 10.0

        return reward
