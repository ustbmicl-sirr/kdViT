"""
Distillation Environment for RL-PyramidKD

Implements the MDP formulation of knowledge distillation with pyramid features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class DistillationEnvironment:
    """
    Knowledge Distillation Environment (MDP)

    State: [global_feat, pyramid_feat, distill_loss, selected_layers, budget_remain]
    Action: Binary vector [a_P2, a_P3, a_P4, a_P5]
    Reward: ΔL_distill + λ * Budget_saved

    Args:
        teacher: Teacher model with pyramid features
        student: Student model with pyramid features
        lambda_tradeoff: Quality-efficiency trade-off parameter (default: 0.5)
        max_steps: Maximum steps per episode (default: 4, one per layer)
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        lambda_tradeoff: float = 0.5,
        max_steps: int = 4
    ):
        self.teacher = teacher
        self.student = student
        self.lambda_tradeoff = lambda_tradeoff
        self.max_steps = max_steps

        # Layer costs (relative FLOPs)
        self.layer_costs = {
            'P2': 4.0,   # 56×56, most expensive
            'P3': 2.0,   # 28×28
            'P4': 1.0,   # 14×14
            'P5': 0.5    # 7×7, cheapest
        }
        self.total_cost = sum(self.layer_costs.values())
        self.layer_names = ['P2', 'P3', 'P4', 'P5']

        # Episode state
        self.current_step = 0
        self.selected_layers = []
        self.sample = None
        self.teacher_feats = None
        self.student_feats = None
        self.initial_loss = 0.0

    def reset(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Reset environment for a new sample

        Args:
            sample: Input sample (image or image-text pair)

        Returns:
            state: Initial state vector
        """
        self.sample = sample
        self.current_step = 0
        self.selected_layers = []

        # Extract pyramid features
        with torch.no_grad():
            self.teacher_feats = self.teacher.extract_pyramid(sample)
            self.student_feats = self.student.extract_pyramid(sample)

        # Compute initial loss (no layers selected)
        self.initial_loss = self.compute_distill_loss([])

        # Construct initial state
        state = self.get_state()

        return state

    def get_state(self) -> torch.Tensor:
        """
        Construct state vector from current environment state

        State components:
            - global_feat: Global sample features [D_global]
            - pyramid_feat: Pyramid layer features [4 × D_pyramid]
            - distill_loss: Current distillation loss [1]
            - selected_layers: Binary indicator of selected layers [4]
            - budget_remain: Remaining computational budget [1]

        Returns:
            state: State vector [state_dim]
        """
        # Global features
        global_feat = self.student_feats.get('global',
                                              self.student_feats['P5'].mean(dim=[2, 3]))

        # Pyramid features (global pooling)
        pyramid_feats = []
        for layer in self.layer_names:
            feat = self.student_feats[layer]
            # Global average pooling
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            pyramid_feats.append(pooled)
        pyramid_feat = torch.cat(pyramid_feats, dim=-1)

        # Current distillation loss
        current_loss = self.compute_distill_loss(self.selected_layers)
        loss_tensor = torch.tensor([[current_loss]], device=global_feat.device)

        # Selected layers (binary indicator)
        selected = torch.zeros(1, 4, device=global_feat.device)
        for layer in self.selected_layers:
            layer_idx = self.layer_names.index(layer)
            selected[0, layer_idx] = 1

        # Remaining budget
        used_cost = sum([self.layer_costs[l] for l in self.selected_layers])
        budget_remain = (self.total_cost - used_cost) / self.total_cost
        budget_tensor = torch.tensor([[budget_remain]], device=global_feat.device)

        # Concatenate all components
        state = torch.cat([
            global_feat,        # [B, D_global]
            pyramid_feat,       # [B, 4 * D_pyramid]
            loss_tensor,        # [B, 1]
            selected,           # [B, 4]
            budget_tensor       # [B, 1]
        ], dim=-1)

        return state

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute action and transition to next state

        Args:
            action: Binary action vector [a_P2, a_P3, a_P4, a_P5]

        Returns:
            next_state: Next state vector
            reward: Immediate reward
            done: Whether episode is terminated
            info: Additional information
        """
        # Decode action to layer names
        action_np = action.cpu().numpy().flatten()
        selected_this_step = [
            self.layer_names[i] for i, a in enumerate(action_np) if a > 0.5
        ]

        # Update selected layers (accumulate)
        prev_selected = self.selected_layers.copy()
        self.selected_layers.extend(selected_this_step)
        self.selected_layers = list(set(self.selected_layers))  # Remove duplicates

        # Compute distillation loss change
        prev_loss = self.compute_distill_loss(prev_selected)
        new_loss = self.compute_distill_loss(self.selected_layers)

        # Quality reward: loss improvement
        delta_loss = prev_loss - new_loss
        r_quality = delta_loss

        # Efficiency reward: saved cost
        used_cost = sum([self.layer_costs[l] for l in self.selected_layers])
        saved_cost = self.total_cost - used_cost
        r_efficiency = saved_cost / self.total_cost

        # Total reward (normalized)
        reward = (r_quality + self.lambda_tradeoff * r_efficiency) / (1 + self.lambda_tradeoff)

        # Update step counter
        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        # Next state
        next_state = self.get_state()

        # Additional info
        info = {
            'selected_layers': self.selected_layers.copy(),
            'num_layers': len(self.selected_layers),
            'distill_loss': new_loss,
            'used_cost': used_cost,
            'saved_cost': saved_cost,
            'delta_loss': delta_loss,
            'r_quality': r_quality,
            'r_efficiency': r_efficiency
        }

        return next_state, reward, done, info

    def compute_distill_loss(self, selected_layers: List[str]) -> float:
        """
        Compute distillation loss for selected layers

        Args:
            selected_layers: List of selected layer names

        Returns:
            loss: Distillation loss value
        """
        if len(selected_layers) == 0:
            return 0.0

        total_loss = 0.0
        for layer in selected_layers:
            student_feat = self.student_feats[layer]
            teacher_feat = self.teacher_feats[layer]

            # MSE loss between features
            loss = F.mse_loss(student_feat, teacher_feat.detach())
            total_loss += loss.item()

        return total_loss

    def render(self) -> str:
        """
        Render current environment state (for debugging)

        Returns:
            state_str: String representation of state
        """
        lines = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Selected layers: {self.selected_layers}",
            f"Distill loss: {self.compute_distill_loss(self.selected_layers):.4f}",
            f"Budget used: {sum([self.layer_costs[l] for l in self.selected_layers]):.1f}/{self.total_cost:.1f}"
        ]
        return "\n".join(lines)


class ParallelDistillationEnv:
    """
    Parallel version of DistillationEnvironment for efficient training

    Manages multiple environments in parallel for batch processing
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_envs: int = 8,
        lambda_tradeoff: float = 0.5,
        max_steps: int = 4
    ):
        self.num_envs = num_envs
        self.envs = [
            DistillationEnvironment(teacher, student, lambda_tradeoff, max_steps)
            for _ in range(num_envs)
        ]

    def reset(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Reset all environments

        Args:
            samples: Batch of samples [num_envs, ...]

        Returns:
            states: Batch of initial states [num_envs, state_dim]
        """
        states = []
        for i, env in enumerate(self.envs):
            state = env.reset(samples[i:i+1])
            states.append(state)

        return torch.cat(states, dim=0)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Step all environments

        Args:
            actions: Batch of actions [num_envs, num_layers]

        Returns:
            next_states: [num_envs, state_dim]
            rewards: [num_envs]
            dones: [num_envs]
            infos: List of info dicts
        """
        next_states = []
        rewards = []
        dones = []
        infos = []

        for i, env in enumerate(self.envs):
            next_state, reward, done, info = env.step(actions[i:i+1])
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        next_states = torch.cat(next_states, dim=0)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        return next_states, rewards, dones, infos


# Example usage
if __name__ == "__main__":
    # Mock teacher and student models
    class MockModel(nn.Module):
        def extract_pyramid(self, x):
            B = x.size(0)
            return {
                'P2': torch.randn(B, 256, 56, 56),
                'P3': torch.randn(B, 256, 28, 28),
                'P4': torch.randn(B, 256, 14, 14),
                'P5': torch.randn(B, 256, 7, 7),
                'global': torch.randn(B, 512)
            }

    teacher = MockModel()
    student = MockModel()

    # Test single environment
    env = DistillationEnvironment(teacher, student, lambda_tradeoff=0.5)

    sample = torch.randn(1, 3, 224, 224)
    state = env.reset(sample)
    print(f"Initial state shape: {state.shape}")

    # Take action
    action = torch.tensor([[1, 0, 1, 1]])  # Select P2, P4, P5
    next_state, reward, done, info = env.step(action)

    print(f"Next state shape: {next_state.shape}")
    print(f"Reward: {reward:.4f}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    print(env.render())

    print("\nDistillationEnvironment test passed!")
