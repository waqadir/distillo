"""
Proximal Policy Optimization (PPO)

Implementation of PPO algorithm for on-policy RL training.
"""

import logging
from typing import Any, Dict, List, Optional

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


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for language model fine-tuning

    PPO Loss = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL[π_old || π_new]

    where:
    - r(θ) = π_new(a|s) / π_old(a|s) (probability ratio)
    - A = advantage function
    - ε = clipping parameter
    - β = KL penalty coefficient
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        kl_penalty_coef: float = 0.1,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        mini_batch_size: int = 4,
    ):
        """
        Initialize PPO trainer

        Args:
            model: Student model to train
            teacher_model: Optional teacher model for KL supervision
            learning_rate: Learning rate for optimizer
            clip_epsilon: PPO clipping parameter
            kl_penalty_coef: KL divergence penalty coefficient
            value_loss_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs per batch
            mini_batch_size: Mini-batch size for PPO updates
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for PPO training")

        self.model = model
        self.teacher_model = teacher_model
        self.clip_epsilon = clip_epsilon
        self.kl_penalty_coef = kl_penalty_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

        # Metrics
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "total_loss": [],
        }

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE)

        A_t = Σ (γλ)^l * δ_t+l
        where δ_t = r_t + γV(s_t+1) - V(s_t)

        Args:
            rewards: Reward signals (batch_size, seq_len)
            values: Value estimates (batch_size, seq_len)
            gamma: Discount factor
            lam: GAE lambda parameter

        Returns:
            Advantages (batch_size, seq_len)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        # Compute advantages backwards
        for t in reversed(range(seq_len - 1)):
            next_value = values[:, t + 1]
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * lam * last_gae

        return advantages

    def compute_policy_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute clipped PPO policy loss

        L^CLIP = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy
            advantages: Advantage estimates

        Returns:
            Policy loss
        """
        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # PPO loss
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss

    def compute_entropy_bonus(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy bonus to encourage exploration

        H(π) = -Σ π(a|s) * log π(a|s)

        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)

        Returns:
            Entropy value
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one PPO training step

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            old_log_probs: Log probabilities from old policy
            rewards: Reward signals
            values: Value estimates

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Compute advantages
        advantages = self.compute_advantages(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute log probabilities for generated tokens
            new_log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = new_log_probs.gather(
                dim=-1, index=input_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Policy loss
            policy_loss = self.compute_policy_loss(old_log_probs, new_log_probs, advantages)

            # Value loss (if we had a value head)
            value_loss = torch.tensor(0.0)

            # Entropy bonus
            entropy = self.compute_entropy_bonus(logits)

            # KL divergence penalty (if teacher model provided)
            kl_loss = torch.tensor(0.0)
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits

                kl_loss = compute_kl_divergence(logits, teacher_logits)

            # Total loss
            total_loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
                + self.kl_penalty_coef * kl_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Track metrics
            self.metrics["policy_loss"].append(policy_loss.item())
            self.metrics["value_loss"].append(value_loss.item())
            self.metrics["entropy"].append(entropy.item())
            self.metrics["kl_divergence"].append(kl_loss.item())
            self.metrics["total_loss"].append(total_loss.item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl_divergence": kl_loss.item(),
            "total_loss": total_loss.item(),
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get average training metrics"""
        return {
            key: sum(values[-10:]) / len(values[-10:]) if values else 0.0
            for key, values in self.metrics.items()
        }
