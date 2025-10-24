"""
Training script for RL-PyramidKD

Main training loop for RL-based dynamic layer selection.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.policy import PolicyNetwork
from rl.trainer import PPOTrainer, ReplayBuffer
from rl.environment import ParallelDistillationEnv
from utils.gradnorm import GradNorm, GradNormTrainer
from utils.logger import Logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train RL-PyramidKD")

    parser.add_argument(
        "--config",
        type=str,
        default="../configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (reduced dataset)"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_models(config: dict):
    """
    Build teacher and student models

    Returns:
        teacher: Teacher model with pyramid features
        student: Student model with pyramid features
    """
    # TODO: Implement model building
    # For now, return mock models
    print("Building teacher and student models...")

    class MockModel(nn.Module):
        def extract_pyramid(self, x):
            B = x.size(0)
            return {
                'P2': torch.randn(B, 256, 56, 56).cuda(),
                'P3': torch.randn(B, 256, 28, 28).cuda(),
                'P4': torch.randn(B, 256, 14, 14).cuda(),
                'P5': torch.randn(B, 256, 7, 7).cuda(),
                'global': torch.randn(B, 512).cuda()
            }

    teacher = MockModel()
    student = MockModel()

    return teacher, student


def build_dataloader(config: dict, split: str = "train"):
    """
    Build data loader

    Args:
        config: Configuration dictionary
        split: Data split ("train" or "val")

    Returns:
        dataloader: PyTorch DataLoader
    """
    # TODO: Implement real dataloader
    # For now, return mock dataloader
    print(f"Building {split} dataloader...")

    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1000 if split == "train" else 100

        def __getitem__(self, idx):
            return torch.randn(3, 224, 224), torch.randint(0, 80, (1,))

    dataset = MockDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=(split == "train"),
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )

    return dataloader


def train_phase1_pretraining(
    teacher, student, train_loader, config, logger
):
    """
    Phase 1: Pre-training with uniform weights

    Train student network with fixed uniform layer weights to collect
    baseline features and losses.
    """
    print("\n" + "="*60)
    print("Phase 1: Pre-training with Uniform Weights")
    print("="*60)

    num_epochs = config['training']['phase1_epochs']

    # Uniform weights
    uniform_weights = torch.ones(4).cuda() / 4.0

    # Simple optimizer
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=config['training']['lr_student']
    )

    for epoch in range(num_epochs):
        student.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.cuda()

            # Extract features
            teacher_feats = teacher.extract_pyramid(images)
            student_feats = student.extract_pyramid(images)

            # Compute distillation loss with uniform weights
            loss = 0
            for i, layer in enumerate(['P2', 'P3', 'P4', 'P5']):
                layer_loss = nn.functional.mse_loss(
                    student_feats[layer],
                    teacher_feats[layer].detach()
                )
                loss += uniform_weights[i] * layer_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        logger.log(f"Phase1 Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    print("Phase 1 completed!\n")


def train_phase2_rl_policy(
    teacher, student, train_loader, config, logger
):
    """
    Phase 2: RL Policy Learning

    Train RL policy with PPO while keeping student backbone fixed.
    """
    print("\n" + "="*60)
    print("Phase 2: RL Policy Learning")
    print("="*60)

    num_episodes = config['rl']['training']['num_episodes']
    num_parallel_envs = config['rl']['training']['num_parallel_envs']

    # Build policy and trainer
    policy = PolicyNetwork(
        state_dim=config['rl']['policy']['state_dim'],
        hidden_dim=config['rl']['policy']['hidden_dim']
    ).cuda()

    ppo_trainer = PPOTrainer(
        policy,
        lr=config['rl']['ppo']['lr'],
        clip_epsilon=config['rl']['ppo']['clip_epsilon']
    )

    # Build parallel environments
    env = ParallelDistillationEnv(
        teacher, student,
        num_envs=num_parallel_envs,
        lambda_tradeoff=config['rl']['environment']['lambda_tradeoff']
    )

    # GradNorm (optional)
    if config['gradnorm']['enabled']:
        gradnorm = GradNorm(
            num_tasks=config['gradnorm']['num_tasks'],
            alpha=config['gradnorm']['alpha']
        ).cuda()

        gradnorm_trainer = GradNormTrainer(
            student, gradnorm,
            lr_model=config['training']['lr_student'],
            lr_weights=config['gradnorm']['lr_weights']
        )

    # Training loop
    for episode in range(num_episodes):
        # Sample batch from dataloader
        images, _ = next(iter(train_loader))
        images = images[:num_parallel_envs].cuda()

        # Reset environments
        states = env.reset(images)

        # Collect rollout
        rollout_states = []
        rollout_actions = []
        rollout_rewards = []
        rollout_values = []
        rollout_log_probs = []
        rollout_dones = []

        for step in range(config['rl']['environment']['max_steps']):
            # Select actions
            actions, log_probs, values = policy.select_action(states)

            # Step environments
            next_states, rewards, dones, infos = env.step(actions)

            # Store rollout
            rollout_states.append(states)
            rollout_actions.append(actions)
            rollout_rewards.append(rewards)
            rollout_values.append(values)
            rollout_log_probs.append(log_probs)
            rollout_dones.append(dones)

            states = next_states

        # Prepare rollout data
        rollout_data = {
            'states': torch.stack(rollout_states).permute(1, 0, 2),  # [B, T, ...]
            'actions': torch.stack(rollout_actions).permute(1, 0, 2),
            'rewards': torch.stack(rollout_rewards).permute(1, 0),
            'values': torch.stack(rollout_values).permute(1, 0, 2).squeeze(-1),
            'log_probs': torch.stack(rollout_log_probs).permute(1, 0),
            'dones': torch.stack(rollout_dones).permute(1, 0)
        }

        # PPO update
        losses = ppo_trainer.train_step(rollout_data)

        # Logging
        if episode % config['logging']['log_interval'] == 0:
            logger.log(
                f"Episode {episode}/{num_episodes}: "
                f"Policy Loss = {losses['policy_loss']:.4f}, "
                f"Value Loss = {losses['value_loss']:.4f}, "
                f"Entropy = {losses['entropy']:.4f}"
            )

        # Save checkpoint
        if episode % config['rl']['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f"policy_episode_{episode}.pth"
            )
            torch.save({
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            logger.log(f"Saved checkpoint: {checkpoint_path}")

    print("Phase 2 completed!\n")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    torch.cuda.manual_seed(config['experiment']['seed'])

    # Create logger
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        experiment_name=config['experiment']['name']
    )
    logger.log(f"Starting experiment: {config['experiment']['name']}")
    logger.log(f"Configuration: {args.config}")

    # Build models
    teacher, student = build_models(config)
    teacher = teacher.cuda()
    student = student.cuda()
    teacher.eval()  # Teacher is frozen

    # Build dataloaders
    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="val")

    # Phase 1: Pre-training
    train_phase1_pretraining(teacher, student, train_loader, config, logger)

    # Phase 2: RL Policy Learning
    train_phase2_rl_policy(teacher, student, train_loader, config, logger)

    # Phase 3: Joint Fine-tuning (TODO)
    # Phase 4: Meta-learning (TODO)

    logger.log("Training completed!")


if __name__ == "__main__":
    main()
