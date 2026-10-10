from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from data.base import DatasetSplits


class SklearnDigitsLoader:
    """Load and preprocess the sklearn handwritten-digits dataset."""

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.2,
    ) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")

        if not 0.0 < validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1.")

        if test_size + validation_size >= 1.0:
            raise ValueError(
                "test_size + validation_size must be less than 1."
            )

        self.test_size = test_size
        self.validation_size = validation_size

    def load(self, seed: int = 42) -> DatasetSplits:
        """Load digits and create deterministic, stratified data splits."""
        dataset = load_digits()

        X = np.asarray(dataset.data, dtype=np.float32)
        X /= np.float32(16.0)

        y = np.asarray(dataset.target, dtype=np.int64)

        # Reserve the test set before creating the training/validation split.
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=seed,
            stratify=y,
        )

        # validation_size is a fraction of the original complete dataset.
        validation_fraction = self.validation_size / (1.0 - self.test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_fraction,
            random_state=seed,
            stratify=y_train_val,
        )

        # Ensure consistent, contiguous memory layouts for all backends.
        arrays = [
            np.ascontiguousarray(X_train, dtype=np.float32),
            np.ascontiguousarray(y_train, dtype=np.int64),
            np.ascontiguousarray(X_val, dtype=np.float32),
            np.ascontiguousarray(y_val, dtype=np.int64),
            np.ascontiguousarray(X_test, dtype=np.float32),
            np.ascontiguousarray(y_test, dtype=np.int64),
        ]

        return DatasetSplits(
            X_train=arrays[0],
            y_train=arrays[1],
            X_val=arrays[2],
            y_val=arrays[3],
            X_test=arrays[4],
            y_test=arrays[5],
            name="sklearn_digits",
            num_classes=10,
            input_shape=(64,),
        )