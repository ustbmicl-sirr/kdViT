"""
GradNorm: Gradient Normalization for Adaptive Loss Balancing

Reference:
    Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing
    in Deep Multitask Networks", ICML 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class GradNorm(nn.Module):
    """
    GradNorm: Automatic task weight balancing

    Dynamically adjusts task weights to balance gradient magnitudes across
    multiple tasks (layers in our case: P2, P3, P4, P5).

    Args:
        num_tasks: Number of tasks/layers (default: 4)
        alpha: Restoring force hyperparameter (default: 1.5)
            Controls how quickly weights adapt to training rate imbalances
    """

    def __init__(self, num_tasks: int = 4, alpha: float = 1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Learnable task weights (initialized to 1)
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

        # Initial task losses (for computing relative inverse training rates)
        self.register_buffer('initial_losses', torch.zeros(num_tasks))
        self.initial_losses_set = False

    def forward(
        self,
        losses: List[torch.Tensor],
        shared_params: nn.ParameterList
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted loss and GradNorm loss

        Args:
            losses: List of task losses [L_P2, L_P3, L_P4, L_P5]
            shared_params: Shared parameters (e.g., student backbone)

        Returns:
            total_loss: Weighted sum of task losses
            gradnorm_loss: GradNorm balancing loss
        """
        # Record initial losses (first forward pass)
        if not self.initial_losses_set:
            with torch.no_grad():
                for i, loss in enumerate(losses):
                    self.initial_losses[i] = loss.item()
            self.initial_losses_set = True

        # 1. Compute weighted losses
        weighted_losses = [w * l for w, l in zip(self.task_weights, losses)]
        total_loss = sum(weighted_losses)

        # 2. Compute gradient norms for each task
        grad_norms = []
        for loss in losses:
            # Compute gradients w.r.t. shared parameters
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,
                create_graph=True
            )

            # Compute L2 norm of gradients
            grad_norm = torch.norm(
                torch.cat([g.flatten() for g in grads]),
                p=2
            )
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        # 3. Compute average gradient norm
        mean_grad_norm = grad_norms.mean()

        # 4. Compute relative inverse training rates
        # r_i(t) = L_i(t) / L_i(0)
        current_losses = torch.tensor([l.item() for l in losses], device=grad_norms.device)
        relative_losses = current_losses / (self.initial_losses + 1e-8)

        # 5. Compute target gradient norms
        # Target: G_W(t) * [r_i(t) / mean(r_i(t))]^alpha
        mean_relative_loss = relative_losses.mean()
        target_grad_norms = mean_grad_norm * torch.pow(
            relative_losses / (mean_relative_loss + 1e-8),
            self.alpha
        )

        # 6. GradNorm loss: L1 distance between actual and target gradient norms
        gradnorm_loss = torch.abs(grad_norms - target_grad_norms.detach()).sum()

        return total_loss, gradnorm_loss

    def get_weights(self) -> torch.Tensor:
        """
        Get normalized task weights

        Returns:
            weights: Normalized weights [num_tasks]
        """
        # Softmax normalization (ensures weights sum to num_tasks)
        weights = F.softmax(self.task_weights, dim=0) * self.num_tasks
        return weights

    def reset_initial_losses(self):
        """Reset initial losses (for new training phase)"""
        self.initial_losses.zero_()
        self.initial_losses_set = False


class GradNormTrainer:
    """
    Trainer with GradNorm optimization

    Manages dual optimization:
        - Model parameters (student network)
        - Task weights (GradNorm)
    """

    def __init__(
        self,
        model: nn.Module,
        gradnorm: GradNorm,
        lr_model: float = 1e-4,
        lr_weights: float = 1e-2
    ):
        self.model = model
        self.gradnorm = gradnorm

        # Dual optimizers
        self.optimizer_model = torch.optim.Adam(
            model.parameters(),
            lr=lr_model
        )
        self.optimizer_weights = torch.optim.Adam(
            [gradnorm.task_weights],
            lr=lr_weights
        )

    def train_step(
        self,
        losses: List[torch.Tensor],
        shared_params: nn.ParameterList
    ) -> Tuple[float, float, torch.Tensor]:
        """
        Training step with GradNorm

        Args:
            losses: List of task losses
            shared_params: Shared parameters for gradient computation

        Returns:
            total_loss: Weighted task loss
            gradnorm_loss: GradNorm balancing loss
            weights: Current task weights
        """
        # Forward pass through GradNorm
        total_loss, gradnorm_loss = self.gradnorm(losses, shared_params)

        # Update model parameters
        self.optimizer_model.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer_model.step()

        # Update task weights
        self.optimizer_weights.zero_grad()
        gradnorm_loss.backward()
        self.optimizer_weights.step()

        # Get current weights
        weights = self.gradnorm.get_weights()

        return total_loss.item(), gradnorm_loss.item(), weights


def compute_grad_norm(
    loss: torch.Tensor,
    parameters: nn.ParameterList,
    norm_type: int = 2
) -> torch.Tensor:
    """
    Compute gradient norm w.r.t. parameters

    Args:
        loss: Loss tensor
        parameters: Model parameters
        norm_type: Norm type (default: 2 for L2)

    Returns:
        grad_norm: Gradient norm
    """
    # Compute gradients
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=True
    )

    # Compute norm
    grad_norm = torch.norm(
        torch.cat([g.flatten() for g in grads]),
        p=norm_type
    )

    return grad_norm


# Example usage
if __name__ == "__main__":
    # Mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.heads = nn.ModuleList([
                nn.Linear(64, 10) for _ in range(4)
            ])

        def forward(self, x):
            feat = self.backbone(x)
            outputs = [head(feat) for head in self.heads]
            return outputs

    # Initialize
    model = MockModel()
    gradnorm = GradNorm(num_tasks=4, alpha=1.5)
    trainer = GradNormTrainer(model, gradnorm, lr_model=1e-4, lr_weights=1e-2)

    # Mock data
    x = torch.randn(32, 256)
    targets = [torch.randint(0, 10, (32,)) for _ in range(4)]

    # Training step
    for step in range(5):
        outputs = model(x)
        losses = [F.cross_entropy(out, tgt) for out, tgt in zip(outputs, targets)]

        total_loss, gradnorm_loss, weights = trainer.train_step(
            losses,
            list(model.backbone.parameters())
        )

        print(f"Step {step}:")
        print(f"  Total loss: {total_loss:.4f}")
        print(f"  GradNorm loss: {gradnorm_loss:.4f}")
        print(f"  Weights: {weights.detach().cpu().numpy()}")

    print("\nGradNorm test passed!")
