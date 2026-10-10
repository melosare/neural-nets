from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class NeuralNetwork(ABC):
    """Common interface for every neural network backend."""

    @abstractmethod
    def fit(
        self,
        X_train: FloatArray,
        y_train: IntArray,
        X_val: FloatArray,
        y_val: IntArray,
    ) -> dict[str, list[float]]:
        """Train the model and return its training history."""


    @abstractmethod
    def predict(self, X: FloatArray) -> IntArray:
        """Return predicted class labels."""


    @abstractmethod
    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return class probabilities."""


    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model and its necessary state."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model state from disk."""


    @abstractmethod
    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""
        