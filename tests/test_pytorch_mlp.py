import numpy as np
import pytest
import torch

from models.mlp import MLPConfig
from implementations.pytorch.mlp import PyTorchMLP


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)

    X = rng.random((120, 64), dtype=np.float32)
    y = (X[:, :10].argmax(axis=1)).astype(np.int64)

    return X[:80], y[:80], X[80:100], y[80:100], X[100:], y[100:]


@pytest.fixture
def model():
    return PyTorchMLP(
        MLPConfig(
            input_size=64,
            hidden_sizes=(32,),
            num_classes=10,
        ),
        epochs=3,
        batch_size=16,
        seed=42,
    )


def test_parameter_count(model):
    assert model.parameter_count() == 64 * 32 + 32 + 32 * 10 + 10


def test_fit_returns_training_history(model, training_data):
    X_train, y_train, X_val, y_val, _, _ = training_data

    history = model.fit(X_train, y_train, X_val, y_val)

    assert set(history) == {
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    }
    assert all(len(values) == 3 for values in history.values())
    assert all(np.isfinite(values).all() for values in history.values())


def test_prediction_shapes_and_labels(model, training_data):
    X_train, y_train, X_val, y_val, X_test, y_test = training_data
    model.fit(X_train, y_train, X_val, y_val)

    predictions = model.predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert predictions.dtype == np.int64
    assert np.all((predictions >= 0) & (predictions < 10))


def test_probabilities_are_valid(model, training_data):
    X_train, y_train, X_val, y_val, X_test, y_test = training_data
    model.fit(X_train, y_train, X_val, y_val)

    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 10)
    assert probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X_test)),
        atol=1e-6,
    )


def test_score_returns_valid_accuracy(model, training_data):
    X_train, y_train, X_val, y_val, X_test, y_test = training_data
    model.fit(X_train, y_train, X_val, y_val)

    accuracy = model.score(X_test, y_test)

    assert 0.0 <= accuracy <= 1.0


def test_checkpoint_round_trip(model, training_data, tmp_path):
    X_train, y_train, X_val, y_val, X_test, y_test = training_data
    model.fit(X_train, y_train, X_val, y_val)

    predictions_before = model.predict(X_test)
    checkpoint = tmp_path / "mlp.pt"
    model.save(checkpoint)

    restored = PyTorchMLP(
        model.config,
        epochs=3,
        batch_size=16,
        seed=123,
    )
    restored.load(checkpoint)

    np.testing.assert_array_equal(
        restored.predict(X_test),
        predictions_before,
    )


def test_checkpoint_rejects_incompatible_architecture(
    model,
    training_data,
    tmp_path,
):
    X_train, y_train, X_val, y_val, _, _ = training_data
    model.fit(X_train, y_train, X_val, y_val)

    checkpoint = tmp_path / "mlp.pt"
    model.save(checkpoint)

    incompatible = PyTorchMLP(
        MLPConfig(
            input_size=64,
            hidden_sizes=(16,),
            num_classes=10,
        ),
        epochs=1,
    )

    with pytest.raises(ValueError, match="architecture"):
        incompatible.load(checkpoint)


def test_invalid_training_configuration():
    config = MLPConfig(
        input_size=64,
        hidden_sizes=(32,),
        num_classes=10,
    )

    with pytest.raises(ValueError):
        PyTorchMLP(config, epochs=0)

    with pytest.raises(ValueError):
        PyTorchMLP(config, batch_size=0)

    with pytest.raises(ValueError):
        PyTorchMLP(config, learning_rate=0.0)


def test_rejects_wrong_input_dtype(model):
    with pytest.raises(TypeError, match="float32"):
        model.predict(np.zeros((4, 64), dtype=np.float64))