from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class DatasetSplits:
    """Train, validation, and test data with consistent feature and label semantics."""

    X_train: FloatArray
    y_train: IntArray
    X_val: FloatArray
    y_val: IntArray
    X_test: FloatArray
    y_test: IntArray

    name: str
    num_classes: int
    input_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset name must not be empty.")

        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if not self.input_shape or any(
            dimension <= 0 for dimension in self.input_shape
        ):
            raise ValueError("input_shape must contain positive dimensions.")

        expected_features = int(np.prod(self.input_shape))

        splits = (
            ("train", self.X_train, self.y_train),
            ("validation", self.X_val, self.y_val),
            ("test", self.X_test, self.y_test),
        )

        for split_name, X, y in splits:
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise TypeError(f"{split_name} features and labels must be NumPy arrays.")

            if X.dtype != np.float32:
                raise TypeError(f"{split_name} features must have dtype float32.")

            if y.dtype != np.int64:
                raise TypeError(f"{split_name} labels must have dtype int64.")

            if X.ndim != 2:
                raise ValueError(
                    f"{split_name} features must have shape (n_samples, n_features)."
                )

            if y.ndim != 1:
                raise ValueError(f"{split_name} labels must be one-dimensional.")

            if X.shape[0] == 0:
                raise ValueError(f"{split_name} split must not be empty.")

            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{split_name} features and labels must have equal sample counts."
                )

            if X.shape[1] != expected_features:
                raise ValueError(
                    f"{split_name} has {X.shape[1]} features; "
                    f"expected {expected_features} from input_shape."
                )

            if not np.isfinite(X).all():
                raise ValueError(f"{split_name} features contain non-finite values.")

            if np.any(y < 0) or np.any(y >= self.num_classes):
                raise ValueError(
                    f"{split_name} labels must be in "
                    f"[0, {self.num_classes - 1}]."
                )

    @property
    def input_size(self) -> int:
        """Number of features presented to a model."""
        return int(np.prod(self.input_shape))

    @property
    def num_train_samples(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def num_val_samples(self) -> int:
        return int(self.X_val.shape[0])

    @property
    def num_test_samples(self) -> int:
        return int(self.X_test.shape[0])


class DatasetLoader(ABC):
    """Interface implemented by every dataset loader."""

    @abstractmethod
    def load(self, seed: int = 42) -> DatasetSplits:
        """Load, preprocess, and split a dataset reproducibly."""
        raise NotImplementedError