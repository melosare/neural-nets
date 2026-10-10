from __future__ import annotations
from dataclasses import dataclass


SUPPORTED_ACTIVATIONS = ("relu", "tanh", "sigmoid")


@dataclass(frozen=True)
class MLPConfig:
    """Framework-independent configuration for a multilayer perceptron."""

    input_size: int
    hidden_sizes: tuple[int, ...]
    num_classes: int
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")

        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("Every hidden layer size must be positive.")

        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {self.activation!r}. "
                f"Choose from {SUPPORTED_ACTIVATIONS}."
            )

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Return the complete sequence of layer widths."""
        return (self.input_size, *self.hidden_sizes, self.num_classes)

    @property
    def num_hidden_layers(self) -> int:
        return len(self.hidden_sizes)

    @property
    def parameter_count(self) -> int:
        """Count weights and biases for a fully connected MLP."""
        return sum(
            fan_in * fan_out + fan_out
            for fan_in, fan_out in zip(
                self.layer_sizes[:-1],
                self.layer_sizes[1:],
            )
        )
