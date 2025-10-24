"""
Logger for RL-PyramidKD experiments

Provides logging functionality for training progress, metrics, and visualization.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Experiment logger

    Logs training progress, metrics, and checkpoints to:
        - Console
        - Log file
        - TensorBoard (optional)
        - Weights & Biases (optional)
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        # TensorBoard writer
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir=str(self.log_dir / "tensorboard" / experiment_name)
            )
        else:
            self.tb_writer = None

        # Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="rl-pyramidkd",
                    name=experiment_name,
                    dir=str(self.log_dir)
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Disabling W&B logging.")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        # Step counter
        self.step = 0

        # Write header
        self.log("="*60)
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Start time: {timestamp}")
        self.log("="*60)

    def log(self, message: str):
        """
        Log message to console and file

        Args:
            message: Message to log
        """
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # Console
        print(log_message)

        # File
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics to TensorBoard and W&B

        Args:
            metrics: Dictionary of metric name -> value
            step: Global step (optional, uses internal counter if None)
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        if step is None:
            step = self.step
            self.step += 1

        # Console
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.log(f"Step {step}: {metric_str}")

        # TensorBoard
        if self.tb_writer is not None:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(
                    f"{prefix}{name}",
                    value,
                    step
                )

        # Weights & Biases
        if self.wandb is not None:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            self.wandb.log(wandb_metrics, step=step)

    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: Optional[int] = None
    ):
        """
        Log image to TensorBoard

        Args:
            tag: Image tag
            image: Image tensor [C, H, W] or [B, C, H, W]
            step: Global step
        """
        if self.tb_writer is None:
            return

        if step is None:
            step = self.step

        self.tb_writer.add_image(tag, image, step)

    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: Optional[int] = None
    ):
        """
        Log histogram to TensorBoard

        Args:
            tag: Histogram tag
            values: Values to histogram
            step: Global step
        """
        if self.tb_writer is None:
            return

        if step is None:
            step = self.step

        self.tb_writer.add_histogram(tag, values, step)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        filename: str
    ):
        """
        Save checkpoint

        Args:
            checkpoint: Checkpoint dictionary
            filename: Checkpoint filename
        """
        checkpoint_path = self.log_dir / "checkpoints" / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, checkpoint_path)
        self.log(f"Saved checkpoint: {checkpoint_path}")

    def close(self):
        """Close logger"""
        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.wandb is not None:
            self.wandb.finish()

        self.log("="*60)
        self.log("Experiment completed")
        self.log("="*60)


# Example usage
if __name__ == "__main__":
    logger = Logger(
        log_dir="experiments/logs",
        experiment_name="test_logger",
        use_tensorboard=True,
        use_wandb=False
    )

    # Log messages
    logger.log("Starting training...")

    # Log metrics
    for step in range(10):
        metrics = {
            'loss': 1.0 / (step + 1),
            'accuracy': step * 0.1,
            'reward': step * 0.05
        }
        logger.log_metrics(metrics, step=step, prefix="train/")

    # Close
    logger.close()

    print("\nLogger test passed!")
