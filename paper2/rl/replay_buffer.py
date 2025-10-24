"""
Replay Buffer for RL-PyramidKD

Stores and manages experience for PPO training.
"""

import torch
from typing import Dict, List, Tuple
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for PPO

    Stores trajectories: (state, action, reward, value, log_prob, done)
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.reset()

    def reset(self):
        """Reset buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool
    ):
        """
        Store a transition

        Args:
            state: State tensor
            action: Action tensor
            reward: Reward value
            value: Value estimate
            log_prob: Log probability of action
            done: Done flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

        # Remove oldest if exceeding max size
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored data

        Returns:
            data: Dictionary of tensors
                - states: [T, state_dim]
                - actions: [T, num_layers]
                - rewards: [T]
                - values: [T]
                - log_probs: [T]
                - dones: [T]
        """
        data = {
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'values': torch.stack(self.values).squeeze(-1),
            'log_probs': torch.stack(self.log_probs),
            'dones': torch.tensor(self.dones, dtype=torch.float32)
        }
        return data

    def size(self) -> int:
        """Get current buffer size"""
        return len(self.states)

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.states) == 0


class RolloutBuffer:
    """
    Rollout buffer for storing complete episodes

    More efficient for on-policy algorithms like PPO
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        num_layers: int,
        num_envs: int = 1,
        device: str = "cpu"
    ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.num_envs = num_envs
        self.device = device

        # Pre-allocate tensors
        self.states = torch.zeros(
            (buffer_size, num_envs, state_dim),
            dtype=torch.float32,
            device=device
        )
        self.actions = torch.zeros(
            (buffer_size, num_envs, num_layers),
            dtype=torch.float32,
            device=device
        )
        self.rewards = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.values = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.log_probs = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.dones = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )

        self.pos = 0
        self.full = False

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor
    ):
        """
        Add a transition to buffer

        Args:
            state: [num_envs, state_dim]
            action: [num_envs, num_layers]
            reward: [num_envs]
            value: [num_envs]
            log_prob: [num_envs]
            done: [num_envs]
        """
        self.states[self.pos] = state.clone()
        self.actions[self.pos] = action.clone()
        self.rewards[self.pos] = reward.clone()
        self.values[self.pos] = value.clone()
        self.log_probs[self.pos] = log_prob.clone()
        self.dones[self.pos] = done.clone()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored data

        Returns:
            data: Dictionary of tensors
        """
        # Get valid data
        if self.full:
            indices = range(self.buffer_size)
        else:
            indices = range(self.pos)

        data = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'dones': self.dones[indices]
        }

        return data

    def reset(self):
        """Reset buffer"""
        self.pos = 0
        self.full = False

    def size(self) -> int:
        """Get current size"""
        if self.full:
            return self.buffer_size
        else:
            return self.pos


# Example usage
if __name__ == "__main__":
    # Test ReplayBuffer
    buffer = ReplayBuffer(max_size=100)

    for i in range(10):
        state = torch.randn(1, 1542)
        action = torch.randint(0, 2, (1, 4)).float()
        reward = float(i)
        value = torch.tensor([[float(i)]])
        log_prob = torch.tensor([0.1])
        done = (i == 9)

        buffer.store(state, action, reward, value, log_prob, done)

    data = buffer.get()
    print("ReplayBuffer test:")
    print(f"  States shape: {data['states'].shape}")
    print(f"  Actions shape: {data['actions'].shape}")
    print(f"  Rewards shape: {data['rewards'].shape}")
    print(f"  Size: {buffer.size()}")

    # Test RolloutBuffer
    rollout_buffer = RolloutBuffer(
        buffer_size=100,
        state_dim=1542,
        num_layers=4,
        num_envs=8
    )

    for i in range(50):
        state = torch.randn(8, 1542)
        action = torch.randint(0, 2, (8, 4)).float()
        reward = torch.randn(8)
        value = torch.randn(8)
        log_prob = torch.randn(8)
        done = torch.zeros(8)

        rollout_buffer.add(state, action, reward, value, log_prob, done)

    data = rollout_buffer.get()
    print("\nRolloutBuffer test:")
    print(f"  States shape: {data['states'].shape}")
    print(f"  Actions shape: {data['actions'].shape}")
    print(f"  Size: {rollout_buffer.size()}")

    print("\nReplay buffer tests passed!")
