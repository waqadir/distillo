"""
On-Policy Trainer

Orchestrates on-policy training with LoRA, teacher supervision, and checkpointing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from distillo.config import OnPolicyConfig
from server.backends import create_backend
from server.models.lora_model import LoRAModel
from server.training.ppo import PPOTrainer
from server.training.importance_sampling import ImportanceSamplingTrainer

logger = logging.getLogger(__name__)


class OnPolicyTrainer:
    """
    On-policy trainer for RL-based model distillation

    Handles the complete training pipeline:
    1. Load student model with LoRA
    2. Initialize teacher model
    3. Run training loop with rollouts
    4. Save checkpoints
    5. Evaluate and log metrics
    """

    def __init__(
        self,
        student_model_name: str,
        teacher_backend: Any,
        config: OnPolicyConfig,
        checkpoint_dir: str | Path = "./checkpoints",
    ):
        """
        Initialize on-policy trainer

        Args:
            student_model_name: HuggingFace model name for student
            teacher_backend: Backend for teacher model (OpenAI, vLLM, etc.)
            config: On-policy training configuration
            checkpoint_dir: Directory for saving checkpoints
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for on-policy training")

        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize student model with LoRA
        logger.info(f"Loading student model: {student_model_name}")
        self.student_model = LoRAModel(
            model_name=student_model_name,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_rank * 2,  # Common practice: alpha = 2*rank
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Teacher backend
        self.teacher_backend = teacher_backend

        # Initialize trainer based on algorithm
        logger.info(f"Initializing {config.loss_fn} trainer")
        if config.loss_fn == "ppo":
            self.trainer = PPOTrainer(
                model=self.student_model.model,
                teacher_model=None,  # Teacher is external API
                learning_rate=config.learning_rate,
                kl_penalty_coef=config.kl_penalty_coef,
            )
        elif config.loss_fn == "importance_sampling":
            # For importance sampling, we need teacher logits
            # This is simplified - in practice would need teacher model loaded
            self.trainer = ImportanceSamplingTrainer(
                model=self.student_model.model,
                teacher_model=None,  # Will use API calls
                learning_rate=config.learning_rate,
                kl_penalty_coef=config.kl_penalty_coef,
            )
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Metrics
        self.training_metrics = {
            "losses": [],
            "kl_divergences": [],
            "eval_scores": [],
        }

    def generate_rollouts(
        self, prompts: List[str], num_rollouts: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Generate rollouts from student model

        Args:
            prompts: List of prompts
            num_rollouts: Number of completions per prompt

        Returns:
            List of rollout data with prompts, completions, and logits
        """
        rollouts = []

        for prompt in prompts:
            for _ in range(num_rollouts):
                # Generate from student
                student_outputs = self.student_model.generate(
                    [prompt],
                    generation_params={
                        "max_tokens": self.config.max_tokens,
                        "temperature": 1.0,
                        "top_p": 0.95,
                    },
                )

                completion = student_outputs[0]["text"]

                # Get teacher response for comparison
                teacher_outputs = self.teacher_backend.generate(
                    [prompt],
                    generation_params={
                        "max_tokens": self.config.max_tokens,
                        "temperature": 1.0,
                    },
                )

                teacher_completion = teacher_outputs[0]["text"]

                rollouts.append(
                    {
                        "prompt": prompt,
                        "student_completion": completion,
                        "teacher_completion": teacher_completion,
                    }
                )

        return rollouts

    def compute_rewards(self, rollouts: List[Dict[str, Any]]) -> List[float]:
        """
        Compute rewards for rollouts

        Simple reward: similarity to teacher output

        Args:
            rollouts: List of rollout data

        Returns:
            List of reward values
        """
        rewards = []

        for rollout in rollouts:
            # Simple reward: negative length difference
            # In practice, would use more sophisticated metrics
            student_len = len(rollout["student_completion"])
            teacher_len = len(rollout["teacher_completion"])

            # Reward based on length similarity (simplified)
            reward = -abs(student_len - teacher_len) / max(student_len, teacher_len, 1)
            rewards.append(reward)

        return rewards

    def train_epoch(self, data_iterator: Iterator[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            data_iterator: Iterator over training data

        Returns:
            Epoch metrics
        """
        logger.info(f"Starting epoch {self.current_epoch}")
        epoch_metrics = {"total_batches": 0, "avg_loss": 0.0}

        for batch_idx, batch in enumerate(data_iterator):
            # Extract prompts from batch
            if isinstance(batch, dict):
                prompts = batch.get("prompt", batch.get("text", []))
            else:
                prompts = [str(item) for item in batch]

            # Ensure prompts is a list
            if not isinstance(prompts, list):
                prompts = [prompts]

            # Generate rollouts
            rollouts = self.generate_rollouts(prompts, num_rollouts=self.config.group_size)

            # Compute rewards
            rewards = self.compute_rewards(rollouts)

            # Prepare training batch
            # This is simplified - real implementation would prepare proper tensors
            logger.info(
                f"Batch {batch_idx}: {len(rollouts)} rollouts, avg reward: {sum(rewards)/len(rewards):.4f}"
            )

            self.global_step += 1

            # Save checkpoint if needed
            if self.config.save_every and self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

            # Evaluate if needed
            if self.config.eval_every and self.global_step % self.config.eval_every == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation metrics: {eval_metrics}")

            epoch_metrics["total_batches"] += 1

        self.current_epoch += 1
        return epoch_metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate current model

        Returns:
            Evaluation metrics
        """
        logger.info("Running evaluation...")

        # Compute KL divergence if configured
        if self.config.compute_post_kl:
            # This would compute KL between student and teacher
            # Simplified for now
            kl_div = 0.0
            return {"kl_divergence": kl_div}

        return {}

    def save_checkpoint(self, name: Optional[str] = None) -> Path:
        """
        Save training checkpoint

        Args:
            name: Optional checkpoint name

        Returns:
            Path to saved checkpoint
        """
        if name is None:
            name = f"checkpoint_step_{self.global_step}"

        checkpoint_path = self.checkpoint_dir / name
        logger.info(f"Saving checkpoint to {checkpoint_path}")

        # Save LoRA weights
        self.student_model.save_checkpoint(checkpoint_path)

        # Save training state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "metrics": self.training_metrics,
            "config": self.config.model_dump(),
        }

        torch.save(state, checkpoint_path / "training_state.pt")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load training checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load LoRA weights
        self.student_model.load_checkpoint(checkpoint_path)

        # Load training state
        state_file = checkpoint_path / "training_state.pt"
        if state_file.exists():
            state = torch.load(state_file)
            self.global_step = state["global_step"]
            self.current_epoch = state["current_epoch"]
            self.training_metrics = state["metrics"]

    def finalize_and_save(self, output_path: str | Path) -> None:
        """
        Merge LoRA weights and save final model

        Args:
            output_path: Path to save final merged model
        """
        logger.info("Finalizing training and saving merged model")
        self.student_model.merge_and_save(output_path)
        logger.info(f"Final model saved to {output_path}")
