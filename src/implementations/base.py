from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias

import numpy as np

from data.base import FloatArray, IntArray


TrainingHistory: TypeAlias = dict[str, list[float]]


class NeuralNetwork(ABC):
    """Common interface for all neural-network implementations.

    Backend implementations must use the same input conventions:
    - Features: float32 arrays of shape (n_samples, n_features).
    - Labels: int64 arrays of shape (n_samples,).
    - Predictions: integer class IDs of shape (n_samples,).
    - Probabilities: float32 arrays of shape (n_samples, n_classes).

    load(path) restores saved weights and state into an already
    initialized instance. Architecture configuration must match
    the saved model.
    """

    @abstractmethod
    def fit(
        self,
        X_train: FloatArray,
        y_train: IntArray,
        X_val: FloatArray,
        y_val: IntArray,
    ) -> TrainingHistory:
        """Train the model and return its training history.

        Suggested history keys include:
        - train_loss
        - val_loss
        - train_accuracy
        - val_accuracy

        All history values should be lists of floats, with one entry
        per epoch or recorded evaluation step.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: FloatArray) -> IntArray:
        """Return predicted class IDs with shape (n_samples,)."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return class probabilities with shape (n_samples, n_classes).

        Each row should contain finite probabilities in [0, 1]
        whose sum is approximately 1. Class columns must follow
        the same class ordering across all backends.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model weights and any necessary model state to path."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model weights and state from path into this instance."""
        raise NotImplementedError

    @abstractmethod
    def parameter_count(self) -> int:
        """Return the total number of trainable model parameters."""
        raise NotImplementedError

    def score(self, X: FloatArray, y: IntArray) -> float:
        """Return classification accuracy for the supplied examples."""
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature array.")

        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional label array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if X.shape[0] == 0:
            raise ValueError("Cannot calculate accuracy on an empty dataset.")

        predictions = self.predict(X)

        if predictions.shape != y.shape:
            raise ValueError(
                "predict() must return one class prediction per sample."
            )

        return float(np.mean(predictions == y))