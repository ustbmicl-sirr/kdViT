"""
PPO Trainer for RL-PyramidKD

Implements the Proximal Policy Optimization algorithm for training
the layer selection policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) Trainer

    Args:
        policy: Policy network
        lr: Learning rate (default: 3e-4)
        clip_epsilon: PPO clip parameter (default: 0.2)
        value_coef: Value loss coefficient (default: 0.5)
        entropy_coef: Entropy bonus coefficient (default: 0.01)
        gamma: Discount factor (default: 0.99)
        gae_lambda: GAE lambda parameter (default: 0.95)
        max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
        ppo_epochs: Number of PPO update epochs (default: 4)
        mini_batch_size: Mini-batch size for PPO updates (default: 64)
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64
    ):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

        # Statistics
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'clip_fraction': []
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: Rewards [T]
            values: Value estimates [T]
            dones: Done flags [T]

        Returns:
            advantages: GAE advantages [T]
            returns: Discounted returns [T]
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_value = 0

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0
                gae = 0

            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + γλ*δ_{t+1} + (γλ)^2*δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae

            advantages[t] = gae
            next_value = values[t]

        # Returns = Advantages + Values
        returns = advantages + values

        return advantages, returns

    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """
        PPO update step

        Args:
            states: States [N, state_dim]
            actions: Actions [N, num_layers]
            old_log_probs: Old log probabilities [N]
            returns: Returns [N]
            advantages: Advantages [N]

        Returns:
            losses: Dictionary of loss values
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Storage for statistics
        policy_losses = []
        value_losses = []
        entropies = []
        total_losses = []
        clip_fractions = []

        # Number of samples
        N = states.size(0)

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Random permutation for mini-batches
            indices = torch.randperm(N)

            # Mini-batch updates
            for start in range(0, N, self.mini_batch_size):
                end = min(start + self.mini_batch_size, N)
                mb_indices = indices[start:end]

                # Get mini-batch
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Evaluate actions
                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    mb_states, mb_actions
                )

                # Ratio: π_new / π_old
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()

                # Statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                total_losses.append(total_loss.item())

                # Clip fraction (for monitoring)
                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                clip_fractions.append(clip_fraction.item())

        # Average losses
        losses = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'total_loss': np.mean(total_losses),
            'clip_fraction': np.mean(clip_fractions)
        }

        # Update statistics
        for key, value in losses.items():
            self.stats[key].append(value)

        return losses

    def train_step(
        self,
        rollout_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step with rollout data

        Args:
            rollout_data: Dictionary containing:
                - 'states': [T, B, state_dim]
                - 'actions': [T, B, num_layers]
                - 'rewards': [T, B]
                - 'values': [T, B]
                - 'log_probs': [T, B]
                - 'dones': [T, B]

        Returns:
            losses: Dictionary of loss values
        """
        # Extract data
        states = rollout_data['states']  # [T, B, state_dim]
        actions = rollout_data['actions']  # [T, B, num_layers]
        rewards = rollout_data['rewards']  # [T, B]
        values = rollout_data['values']  # [T, B]
        log_probs = rollout_data['log_probs']  # [T, B]
        dones = rollout_data['dones']  # [T, B]

        T, B = rewards.shape

        # Flatten batch dimension
        states = states.view(T * B, -1)  # [T*B, state_dim]
        actions = actions.view(T * B, -1)  # [T*B, num_layers]
        log_probs = log_probs.view(T * B)  # [T*B]

        # Compute GAE for each trajectory
        all_advantages = []
        all_returns = []

        for b in range(B):
            advantages, returns = self.compute_gae(
                rewards[:, b],
                values[:, b],
                dones[:, b]
            )
            all_advantages.append(advantages)
            all_returns.append(returns)

        advantages = torch.stack(all_advantages, dim=1).view(T * B)  # [T*B]
        returns = torch.stack(all_returns, dim=1).view(T * B)  # [T*B]

        # PPO update
        losses = self.ppo_update(states, actions, log_probs, returns, advantages)

        return losses

    def get_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.stats

    def reset_stats(self):
        """Reset training statistics"""
        for key in self.stats:
            self.stats[key] = []


import torch.nn.functional as F


class ReplayBuffer:
    """
    Replay buffer for storing rollout data

    Stores trajectories for PPO training
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
        """Store a transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored data

        Returns:
            data: Dictionary of tensors
        """
        data = {
            'states': torch.stack(self.states),  # [T, state_dim]
            'actions': torch.stack(self.actions),  # [T, num_layers]
            'rewards': torch.tensor(self.rewards),  # [T]
            'values': torch.stack(self.values).squeeze(-1),  # [T]
            'log_probs': torch.stack(self.log_probs),  # [T]
            'dones': torch.tensor(self.dones, dtype=torch.float32)  # [T]
        }
        return data

    def size(self) -> int:
        """Get buffer size"""
        return len(self.states)


# Example usage
if __name__ == "__main__":
    from .policy import PolicyNetwork

    # Initialize policy and trainer
    policy = PolicyNetwork(state_dim=1542, hidden_dim=256)
    trainer = PPOTrainer(policy, lr=3e-4)

    # Mock rollout data
    T, B = 10, 8  # 10 steps, 8 parallel environments
    state_dim = 1542
    num_layers = 4

    rollout_data = {
        'states': torch.randn(T, B, state_dim),
        'actions': (torch.rand(T, B, num_layers) > 0.5).float(),
        'rewards': torch.randn(T, B),
        'values': torch.randn(T, B),
        'log_probs': torch.randn(T, B),
        'dones': torch.zeros(T, B)
    }

    # Training step
    losses = trainer.train_step(rollout_data)

    print("Training losses:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")

    print("\nPPOTrainer test passed!")
