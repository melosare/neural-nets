# Neural Networks

The project is designed to support interchangeable implementations, reproducible experiments, and extensible dataset and model interfaces.

## Project Status

**Current stage:** Foundational interfaces and dataset loading.

The full benchmarking pipeline and all planned backends are not yet complete.

### Implemented or present

- [x] Git repository initialized.
- [x] Python source layout established under `src/`.
- [x] Shared dataset interface in `src/data/base.py`.
- [x] Scikit-learn digits loader in `src/data/sklearn_digits.py`.
- [x] Common neural-network interface in `src/implementations/base.py`.
- [x] Initial PyTorch MLP implementation file present.
- [x] Initial C++ implementation scaffolding present.
- [x] Dataset tests created and passing.
- [x] Pytest configured to discover modules from `src/`.

The presence of an implementation file does not necessarily mean its functionality is complete or fully tested.

## Project Goals

The long-term goal is to build a scalable Python framework that can train and compare neural networks implemented with different technologies.

Initial planned backends:

- **PyTorch** — neural networks implemented with PyTorch.
- **TensorFlow** — neural networks implemented with TensorFlow.
- **NumPy** — manual forward propagation and backpropagation.
- **C++** — native implementation exposed to Python through a binding.

The first dataset is `sklearn.datasets.load_digits()`. Additional datasets and architectures should be addable without rewriting the benchmark runner.

The benchmark system is intended to:

1. Load and preprocess datasets through a common interface.
2. Execute multiple architecture, backend, and configuration combinations.
3. Train models and record training history.
4. Evaluate classification quality.
5. Measure training time, inference latency, throughput, model size, and parameter count.
6. Support future memory-usage measurements.
7. Save model checkpoints and experiment results.
8. Rank eligible candidates according to configurable selection criteria.
9. Return the best eligible model/backend combination and a report of all experiments.

## Repository Structure

The project deliberately uses `src/` as the Python source root without an additional `src/neural_nets/` package directory.

```text
neural-nets/
├── README.md
├── .gitignore
├── pyproject.toml
├── tests/
│   ├── test_implementations.py
│   ├── test_benchmark.py
│   └── test_data.py
├── artifacts/
├── configs/
├── experiments/
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── sklearn_digits.py
│   ├── implementations/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pytorch/
│   │   │   ├── __init__.py
│   │   │   └── mlp.py
│   │   └── cpp/
│   │       ├── __init__.py
│   │       └── native/
│   │           ├── mlp.cpp
│   │           └── CMakeLists.txt
│   ├── models/
│   ├── benchmark/
│   └── results/
│       ├── __init__.py
│       └── schema.py
└── .idea/
```

Some directories and files are scaffolding for planned functionality and may be empty or incomplete. The tree above describes the intended/current project organization, not a guarantee that every listed component is fully implemented.

## Dataset Interface

Location: `src/data/base.py`

The dataset layer defines:

- `FloatArray` — NumPy arrays with `float32` features.
- `IntArray` — NumPy arrays with `int64` labels.
- `DatasetSplits` — a dataclass containing training, validation, and test data.
- `DatasetLoader` — an abstract interface for dataset loaders.

`DatasetSplits` validates basic properties such as array dtypes, dimensions, sample counts, feature counts, finite feature values, and label ranges.

Its metadata includes the dataset name, number of classes, and input shape. Convenience properties expose the flattened input size and sample counts.

All dataset loaders should implement:

```python
load(seed: int = 42) -> DatasetSplits
```

The dataset layer owns preprocessing and splitting. Neural-network implementations must not independently load or split datasets.

## Initial Dataset: Scikit-Learn Digits

Location: `src/data/sklearn_digits.py`

The `SklearnDigitsLoader` uses `sklearn.datasets.load_digits()`.

Current preprocessing and splitting behavior:

- Converts features to `float32`.
- Normalizes pixel intensities from `[0, 16]` to `[0, 1]`.
- Converts labels to `int64`.
- Uses flattened 64-feature inputs from 8×8 images.
- Creates stratified training, validation, and test splits.
- Uses a configurable random seed for reproducibility.
- Produces contiguous NumPy arrays.

Default split configuration:

| Split | Fraction of dataset | Observed sample count |
|---|---:|---:|
| Training | 60% | 1,077 |
| Validation | 20% | 360 |
| Test | 20% | 360 |
| Total | 100% | 1,797 |

Actual counts depend on scikit-learn's rounding behavior.

Dataset metadata:

- Name: `sklearn_digits`
- Input shape: `(64,)`
- Number of classes: `10`
- Feature dtype: `float32`
- Label dtype: `int64`

Validation data should be used for model selection. The test set should remain reserved for final evaluation.

## Neural-Network Interface

Location: `src/implementations/base.py`

The abstract `NeuralNetwork` interface defines the common API expected of backend implementations.

```python
fit(X_train, y_train, X_val, y_val) -> TrainingHistory

predict(X) -> IntArray

predict_proba(X) -> FloatArray

save(path: Path) -> None

load(path: Path) -> None

parameter_count() -> int

score(X, y) -> float
```

## Benchmark Architecture

The benchmark layer is intended to orchestrate experiments through a backend registry or factory rather than embedding framework-specific logic in the runner.

Planned modules:

- `src/benchmark/run.py` — experiment orchestration.
- `src/benchmark/timing.py` — training and inference measurements.
- `src/benchmark/metrics.py` — classification and training metrics.
- `src/benchmark/selection.py` — candidate eligibility and model selection.
- `src/results/schema.py` — common result schema.
- `src/results/exporters.py` — planned JSONL and CSV export.

Each experiment result should identify the dataset, architecture, backend, configuration, seed, trial, quality metrics, timing metrics, and environment metadata.

Fair comparisons should use identical data splits, matching architectures and hyperparameters, multiple recorded seeds, inference warm-up, and accelerator synchronization where appropriate. Initialization and compilation overhead should be separated from steady-state execution when possible.

For the C++ backend, Python-call end-to-end latency should be distinguished from native-only execution time where feasible.

## Environment Setup

Use a Python environment with the required dependencies installed.

Initial dependencies:

- Python 3.11 or another compatible Python version.
- NumPy.
- scikit-learn.
- pytest.

PyTorch, TensorFlow, and native build dependencies will be needed as their respective backends are completed.

If using Conda, activate your project environment before running commands.

## Python Import Configuration

The project uses `src/` as the source root.

The existing `pyproject.toml` contains:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

This configures pytest to discover application modules under `src/`. It does not, by itself, configure every standalone Python invocation.

In PyCharm, mark `src/` as **Sources Root**.

From the project root, standalone commands can temporarily use:

```bash
PYTHONPATH=src python -c "from data.sklearn_digits import SklearnDigitsLoader; print(SklearnDigitsLoader().load().name)"
```

Avoid mixing imports such as `data.base` and `src.data.base` in the same project.

## Testing

The dataset test suite is located at `tests/test_data.py`.

It covers dataset metadata, shapes, sample counts, dtypes, normalized feature values, class coverage, reproducibility, contiguous arrays, and invalid split configurations.

Run the dataset tests:

```bash
python -m pytest tests/test_data.py -v
```

Run the full test suite:

```bash
python -m pytest -v
```

**Current verification:** the dataset test suite has passed in the development environment. The full test suite and all backend implementations still require verification.

## Artifacts and Results

Generated checkpoints, reports, and other experiment outputs should normally be written under `artifacts/`.

Generated artifacts should generally be excluded from Git unless selected checkpoints or reports are intentionally versioned.

Experiment configurations belong under `configs/`, while experiment scripts or supporting materials can be organized under `experiments/`.

## License

TBD