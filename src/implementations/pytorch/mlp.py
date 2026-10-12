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

    def __init__(
        self,
        config: MLPConfig,
        *,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        self.config = config
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested, not available.")

        torch.manual_seed(seed)

        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def _build_model(self) -> nn.Sequential:
        """Construct the MLP from the framework-independent configuration."""
        activations: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        activation_type = activations[self.config.activation]
        layers: list[nn.Module] = []
        sizes = self.config.layer_sizes

        for index, (fan_in, fan_out) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            layers.append(nn.Linear(fan_in, fan_out))

            # The output layer returns logits for CrossEntropyLoss.
            if index < len(sizes) - 2:
                layers.append(activation_type())

        return nn.Sequential(*layers)

    def _validate_features(self, X: FloatArray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.dtype != np.float32:
            raise TypeError("X must have dtype float32.")
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if X.shape[1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} features, "
                f"received {X.shape[1]}."
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")

    def _validate_labels(
        self,
        y: IntArray,
        expected_samples: int,
        name: str,
    ) -> None:
        if not isinstance(y, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if y.dtype != np.int64:
            raise TypeError(f"{name} must have dtype int64.")
        if y.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if y.shape[0] != expected_samples:
            raise ValueError(f"{name} has an incorrect sample count.")
        if np.any(y < 0) or np.any(y >= self.config.num_classes):
            raise ValueError(
                f"{name} labels must be in "
                f"[0, {self.config.num_classes - 1}]."
            )

    def _to_tensor(self, X: FloatArray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(X)).to(self.device)

    def _evaluate(
        self,
        X: FloatArray,
        y: IntArray,
    ) -> tuple[float, float]:
        """Return mean cross-entropy loss and accuracy."""
        self.model.eval()
        features = self._to_tensor(X)
        labels = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)

        with torch.no_grad():
            logits = self.model(features)
            loss = self.criterion(logits, labels).item()
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

        return float(loss), float(accuracy)

    def fit(
        self,
        X_train: FloatArray,
        y_train: IntArray,
        X_val: FloatArray,
        y_val: IntArray,
    ) -> TrainingHistory:
        self._validate_features(X_train)
        self._validate_labels(y_train, X_train.shape[0], "y_train")
        self._validate_features(X_val)
        self._validate_labels(y_val, X_val.shape[0], "y_val")

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_dataset = TensorDataset(
            self._to_tensor(X_train),
            torch.from_numpy(np.ascontiguousarray(y_train)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        history: TrainingHistory = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        for _ in range(self.epochs):
            self.model.train()

            for features, labels in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            train_loss, train_accuracy = self._evaluate(X_train, y_train)
            val_loss, val_accuracy = self._evaluate(X_val, y_val)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_accuracy"].append(val_accuracy)

        return history

    def predict(self, X: FloatArray) -> IntArray:
        self._validate_features(X)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(self._to_tensor(X)).argmax(dim=1)

        return predictions.cpu().numpy().astype(np.int64, copy=False)

    def predict_proba(self, X: FloatArray) -> FloatArray:
        self._validate_features(X)
        self.model.eval()

        with torch.no_grad():
            probabilities = torch.softmax(
                self.model(self._to_tensor(X)),
                dim=1,
            )

        return probabilities.cpu().numpy().astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        """Save the architecture metadata and learned weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "input_size": self.config.input_size,
            "hidden_sizes": list(self.config.hidden_sizes),
            "num_classes": self.config.num_classes,
            "activation": self.config.activation,
            "state_dict": self.model.state_dict(),
        }

        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        """Restore weights into this initialized model."""
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=True,
        )

        expected_config = {
            "input_size": self.config.input_size,
            "hidden_sizes": list(self.config.hidden_sizes),
            "num_classes": self.config.num_classes,
            "activation": self.config.activation,
        }

        actual_config = {
            key: checkpoint[key]
            for key in expected_config
        }

        if actual_config != expected_config:
            raise ValueError(
                "Checkpoint architecture does not match this model's configuration."
            )

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)

    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )