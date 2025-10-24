"""
Meta-Learning for RL-PyramidKD

Implements MAML (Model-Agnostic Meta-Learning) for fast adaptation
to new tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import copy


class MAMLTrainer:
    """
    MAML (Model-Agnostic Meta-Learning) Trainer

    Enables fast adaptation of the RL policy to new tasks with minimal
    fine-tuning.

    Args:
        meta_policy: Policy network to meta-train
        inner_lr: Learning rate for inner loop (task adaptation)
        outer_lr: Learning rate for outer loop (meta-update)
        num_inner_steps: Number of gradient steps in inner loop
        first_order: Use first-order approximation (faster, less accurate)
    """

    def __init__(
        self,
        meta_policy: nn.Module,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-4,
        num_inner_steps: int = 5,
        first_order: bool = False
    ):
        self.meta_policy = meta_policy
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

        # Meta-optimizer (for outer loop)
        self.meta_optimizer = optim.Adam(
            meta_policy.parameters(),
            lr=outer_lr
        )

    def inner_update(
        self,
        task_data: Dict,
        create_graph: bool = True
    ) -> Tuple[nn.Module, List[torch.Tensor]]:
        """
        Inner loop: Adapt policy to a specific task

        Args:
            task_data: Task-specific data (support set)
            create_graph: Whether to create computation graph (for 2nd order)

        Returns:
            adapted_policy: Policy adapted to task
            inner_losses: Losses during adaptation
        """
        # Clone policy for task-specific adaptation
        adapted_policy = copy.deepcopy(self.meta_policy)

        # Inner loop optimizer (SGD for simplicity)
        inner_optimizer = optim.SGD(
            adapted_policy.parameters(),
            lr=self.inner_lr
        )

        inner_losses = []

        # Adapt to task
        for step in range(self.num_inner_steps):
            # Compute task loss
            loss = self.compute_task_loss(adapted_policy, task_data)
            inner_losses.append(loss.item())

            # Gradient step
            inner_optimizer.zero_grad()
            loss.backward(create_graph=create_graph and not self.first_order)
            inner_optimizer.step()

        return adapted_policy, inner_losses

    def meta_update(
        self,
        tasks: List[Dict]
    ) -> Dict[str, float]:
        """
        Meta-update: Update meta-policy based on multiple tasks

        Args:
            tasks: List of task dictionaries, each containing:
                - support: Support set (for adaptation)
                - query: Query set (for evaluation)

        Returns:
            meta_losses: Dictionary of meta-learning losses
        """
        meta_loss = 0.0
        task_losses = []

        for task in tasks:
            # Inner loop: Adapt to task using support set
            adapted_policy, inner_losses = self.inner_update(
                task['support'],
                create_graph=not self.first_order
            )

            # Evaluate adapted policy on query set
            query_loss = self.compute_task_loss(adapted_policy, task['query'])
            meta_loss += query_loss
            task_losses.append(query_loss.item())

        # Average meta-loss
        meta_loss = meta_loss / len(tasks)

        # Meta-gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return {
            'meta_loss': meta_loss.item(),
            'avg_task_loss': sum(task_losses) / len(task_losses),
            'min_task_loss': min(task_losses),
            'max_task_loss': max(task_losses)
        }

    def compute_task_loss(
        self,
        policy: nn.Module,
        task_data: Dict
    ) -> torch.Tensor:
        """
        Compute loss for a task

        Args:
            policy: Policy network
            task_data: Task data (states, actions, etc.)

        Returns:
            loss: Task loss
        """
        # Extract data
        states = task_data['states']
        actions = task_data['actions']
        advantages = task_data.get('advantages', None)
        returns = task_data.get('returns', None)

        # Evaluate actions
        log_probs, values, _ = policy.evaluate_actions(states, actions)

        # Policy loss
        if advantages is not None:
            policy_loss = -(log_probs * advantages).mean()
        else:
            policy_loss = -log_probs.mean()

        # Value loss (optional)
        if returns is not None:
            value_loss = (values.squeeze(-1) - returns).pow(2).mean()
            total_loss = policy_loss + 0.5 * value_loss
        else:
            total_loss = policy_loss

        return total_loss

    def fast_adapt(
        self,
        task_data: Dict,
        num_steps: int = 5
    ) -> nn.Module:
        """
        Fast adaptation to a new task (inference)

        Args:
            task_data: Support set for new task
            num_steps: Number of adaptation steps

        Returns:
            adapted_policy: Adapted policy
        """
        # Clone policy
        adapted_policy = copy.deepcopy(self.meta_policy)

        # Optimizer
        optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)

        # Adapt
        for step in range(num_steps):
            loss = self.compute_task_loss(adapted_policy, task_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_policy


class Task:
    """
    Task wrapper for meta-learning

    Represents a task with support and query sets
    """

    def __init__(
        self,
        name: str,
        support_data: Dict,
        query_data: Dict
    ):
        self.name = name
        self.support = support_data
        self.query = query_data

    def sample_support(self, n_shots: int = None):
        """Sample from support set"""
        if n_shots is None:
            return self.support

        # Random sampling
        indices = torch.randperm(len(self.support['states']))[:n_shots]
        return {
            'states': self.support['states'][indices],
            'actions': self.support['actions'][indices],
            'advantages': self.support.get('advantages', None),
            'returns': self.support.get('returns', None)
        }

    def sample_query(self, n_shots: int = None):
        """Sample from query set"""
        if n_shots is None:
            return self.query

        # Random sampling
        indices = torch.randperm(len(self.query['states']))[:n_shots]
        return {
            'states': self.query['states'][indices],
            'actions': self.query['actions'][indices],
            'advantages': self.query.get('advantages', None),
            'returns': self.query.get('returns', None)
        }


# Example usage
if __name__ == "__main__":
    from .policy import PolicyNetwork

    # Initialize policy
    policy = PolicyNetwork(state_dim=1542, hidden_dim=256)

    # Initialize MAML trainer
    maml = MAMLTrainer(
        meta_policy=policy,
        inner_lr=1e-3,
        outer_lr=1e-4,
        num_inner_steps=5
    )

    # Create mock tasks
    tasks = []
    for i in range(3):
        # Support set
        support_data = {
            'states': torch.randn(50, 1542),
            'actions': (torch.rand(50, 4) > 0.5).float(),
            'advantages': torch.randn(50),
            'returns': torch.randn(50)
        }

        # Query set
        query_data = {
            'states': torch.randn(20, 1542),
            'actions': (torch.rand(20, 4) > 0.5).float(),
            'advantages': torch.randn(20),
            'returns': torch.randn(20)
        }

        task = Task(
            name=f"task_{i}",
            support_data=support_data,
            query_data=query_data
        )
        tasks.append({'support': support_data, 'query': query_data})

    # Meta-update
    meta_losses = maml.meta_update(tasks)

    print("Meta-learning test:")
    print(f"  Meta loss: {meta_losses['meta_loss']:.4f}")
    print(f"  Avg task loss: {meta_losses['avg_task_loss']:.4f}")

    # Fast adaptation
    new_task_data = {
        'states': torch.randn(30, 1542),
        'actions': (torch.rand(30, 4) > 0.5).float(),
        'advantages': torch.randn(30),
        'returns': torch.randn(30)
    }
    adapted_policy = maml.fast_adapt(new_task_data, num_steps=5)
    print("\nFast adaptation completed!")

    print("\nMAML test passed!")
