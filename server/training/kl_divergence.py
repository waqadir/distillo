"""
KL Divergence Calculation

Utilities for computing KL divergence between model distributions.
"""

import logging
from typing import Optional

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def compute_kl_divergence(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> "torch.Tensor":
    """
    Compute KL divergence between student and teacher distributions

    KL(teacher || student) = Σ teacher_probs * log(teacher_probs / student_probs)

    Args:
        student_logits: Logits from student model (batch_size, seq_len, vocab_size)
        teacher_logits: Logits from teacher model (batch_size, seq_len, vocab_size)
        temperature: Temperature for softmax (higher = softer distribution)
        reduction: Reduction method ('batchmean', 'sum', 'mean', 'none')

    Returns:
        KL divergence value
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for KL divergence computation")

    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Convert to log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Compute KL divergence
    # KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
    kl_div = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction=reduction,
        log_target=True,
    )

    return kl_div


def compute_forward_kl(
    student_logits: "torch.Tensor", teacher_logits: "torch.Tensor", temperature: float = 1.0
) -> "torch.Tensor":
    """
    Compute forward KL: KL(student || teacher)

    Used when we want to minimize student's entropy while matching teacher.

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Softmax temperature

    Returns:
        Forward KL divergence
    """
    return compute_kl_divergence(teacher_logits, student_logits, temperature)


def compute_reverse_kl(
    student_logits: "torch.Tensor", teacher_logits: "torch.Tensor", temperature: float = 1.0
) -> "torch.Tensor":
    """
    Compute reverse KL: KL(teacher || student)

    Used when we want to cover teacher's distribution with student.

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Softmax temperature

    Returns:
        Reverse KL divergence
    """
    return compute_kl_divergence(student_logits, teacher_logits, temperature)


def compute_js_divergence(
    student_logits: "torch.Tensor", teacher_logits: "torch.Tensor", temperature: float = 1.0
) -> "torch.Tensor":
    """
    Compute Jensen-Shannon divergence (symmetric KL)

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Softmax temperature

    Returns:
        JS divergence
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required")

    # Scale by temperature
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Compute probability distributions
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Compute mixture distribution
    mixture_probs = 0.5 * (student_probs + teacher_probs)
    mixture_log_probs = torch.log(mixture_probs + 1e-10)

    # Compute KL divergences to mixture
    kl_student = F.kl_div(mixture_log_probs, student_probs, reduction="batchmean")
    kl_teacher = F.kl_div(mixture_log_probs, teacher_probs, reduction="batchmean")

    # JS divergence is symmetric average
    js_div = 0.5 * (kl_student + kl_teacher)

    return js_div


def compute_per_token_kl(
    student_logits: "torch.Tensor", teacher_logits: "torch.Tensor", temperature: float = 1.0
) -> "torch.Tensor":
    """
    Compute per-token KL divergence (no reduction)

    Args:
        student_logits: Student logits (batch_size, seq_len, vocab_size)
        teacher_logits: Teacher logits (batch_size, seq_len, vocab_size)
        temperature: Softmax temperature

    Returns:
        Per-token KL values (batch_size, seq_len)
    """
    kl_per_token = compute_kl_divergence(
        student_logits, teacher_logits, temperature, reduction="none"
    )
    # Sum over vocabulary dimension
    return kl_per_token.sum(dim=-1)
