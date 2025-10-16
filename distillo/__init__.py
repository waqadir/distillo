"""
Distillo: Lightweight on/off-policy distillation framework

Distillo provides a simple client-server architecture for generating training data
through model distillation and on-policy learning.
"""

__version__ = "0.1.0"
__author__ = "Waqar Qadir"

from distillo.client import DistilloClient
from distillo.config import AppConfig, JobConfig, ServerConfig

__all__ = [
    "DistilloClient",
    "AppConfig",
    "JobConfig",
    "ServerConfig",
]
