from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    experiment_id: str
    dataset: str
    architecture: str
    backend: str
    config: dict[str, Any]

    accuracy: float
    macro_f1: float
    test_loss: float

    training_time_seconds: float
    inference_latency_ms: float
    throughput_samples_per_second: float
    parameter_count: int
    model_size_bytes: int | None

    seed: int
    trial: int
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)