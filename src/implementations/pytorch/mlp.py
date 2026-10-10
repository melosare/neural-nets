from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.base import FloatArray, IntArray
from implementations.base import NeuralNetwork, TrainingHistory
from models.mlp import MLPConfig

class PyTorchMLP(NeuralNetwork):
    """MLP implemented with PyTorch"""
    def __init__(self,
                 config: MLPConfig,
                 *,
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 50,
                 seed: int = 42,
                 device: str = "cpu",): -> None:
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if epochs <= 0:
            raise ValueError("Epochs must be positive.")


