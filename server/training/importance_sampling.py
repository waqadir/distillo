"""
Importance Sampling Trainer

Simpler alternative to PPO using importance sampling for on-policy training.
"""

import logging
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from server.training.kl_divergence import compute_kl_divergence

logger = logging.getLogger(__name__)


class ImportanceSamplingTrainer:
    """
    Importance Sampling trainer for language model distillation

    Uses importance weighting to correct for distribution mismatch
    between student and teacher models.

    Loss = E[w(s,a) * L(s,a)] + β * KL[π_teacher || π_student]

    where w(s,a) = π_teacher(a|s) / π_student(a|s)
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_model: nn.Module,
        learning_rate: float = 1e-5,
        kl_penalty_coef: float = 0.1,
        max_importance_weight: float = 10.0,
        max_grad_norm: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize importance sampling trainer

        Args:
            model: Student model to train
            teacher_model: Teacher model for supervision
            learning_rate: Learning rate
            kl_penalty_coef: KL divergence penalty coefficient
            max_importance_weight: Maximum importance weight (for stability)
            max_grad_norm: Maximum gradient norm for clipping
            temperature: Temperature for softmax distributions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for importance sampling training")

        self.model = model
        self.teacher_model = teacher_model
        self.kl_penalty_coef = kl_penalty_coef
        self.max_importance_weight = max_importance_weight
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

        # Metrics
        self.metrics = {
            "distillation_loss": [],
            "kl_divergence": [],
            "importance_weights": [],
            "total_loss": [],
        }

    def compute_importance_weights(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance weights

        w(s,a) = π_teacher(a|s) / π_student(a|s)

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            target_ids: Target token IDs

        Returns:
            Importance weights (clamped for stability)
        """
        # Compute probabilities
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Get probabilities for target tokens
        student_target_probs = student_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        teacher_target_probs = teacher_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # Compute importance weights
        importance_weights = teacher_target_probs / (student_target_probs + 1e-10)

        # Clamp for stability
        importance_weights = torch.clamp(importance_weights, 0.0, self.max_importance_weight)

        return importance_weights

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_ids: torch.Tensor,
        importance_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss

        L = -w(s,a) * log π_student(a|s)

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            target_ids: Target token IDs
            importance_weights: Importance weights

        Returns:
            Weighted distillation loss
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            target_ids.view(-1),
            reduction="none",
        )

        # Reshape and weight by importance
        ce_loss = ce_loss.view(target_ids.shape)
        weighted_loss = (importance_weights * ce_loss).mean()

        return weighted_loss

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one training step

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            target_ids: Target token IDs (if None, use input_ids shifted)

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.teacher_model.eval()

        # Use input_ids shifted as targets if not provided
        if target_ids is None:
            target_ids = input_ids.clone()

        # Forward pass - student
        student_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # Forward pass - teacher (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Compute importance weights
        importance_weights = self.compute_importance_weights(student_logits, teacher_logits, target_ids)

        # Distillation loss (weighted by importance)
        distill_loss = self.compute_distillation_loss(
            student_logits, teacher_logits, target_ids, importance_weights
        )

        # KL divergence penalty
        kl_loss = compute_kl_divergence(student_logits, teacher_logits, temperature=self.temperature)

        # Total loss
        total_loss = distill_loss + self.kl_penalty_coef * kl_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Track metrics
        avg_importance_weight = importance_weights.mean().item()
        self.metrics["distillation_loss"].append(distill_loss.item())
        self.metrics["kl_divergence"].append(kl_loss.item())
        self.metrics["importance_weights"].append(avg_importance_weight)
        self.metrics["total_loss"].append(total_loss.item())

        return {
            "distillation_loss": distill_loss.item(),
            "kl_divergence": kl_loss.item(),
            "avg_importance_weight": avg_importance_weight,
            "total_loss": total_loss.item(),
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get average training metrics"""
        return {
            key: sum(values[-10:]) / len(values[-10:]) if values else 0.0
            for key, values in self.metrics.items()
        }
