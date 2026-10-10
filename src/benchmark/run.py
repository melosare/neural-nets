from dataclasses import dataclass
from typing import Callable

from neural_nets.datasets.base import DatasetLoader
from neural_nets.implementations.base import NeuralNetwork


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset_name: str
    architecture: str
    backend: str
    model_config: dict


BackendFactory = Callable[[dict], NeuralNetwork]


def run_benchmark(
    dataset_loader: DatasetLoader,
    experiments: list[Experiment],
    backend_factories: dict[str, BackendFactory],
) -> list[dict]:
    splits = dataset_loader.load(seed=42)
    results = []

    for experiment in experiments:
        model = backend_factories[experiment.backend](
            experiment.model_config
        )

        # A separate function handles timing, fitting,
        # prediction, metrics, and error capture.
        result = evaluate_experiment(
            model=model,
            experiment=experiment,
            splits=splits,
        )
        results.append(result)

    return results