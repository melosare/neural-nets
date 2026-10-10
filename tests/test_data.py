import numpy as np
import pytest

from data.sklearn_digits import SklearnDigitsLoader

@pytest.fixture
def dataset():
    return SklearnDigitsLoader().load(seed=42)


def test_dataset_metadata(dataset):
    assert dataset.name == "sklearn_digits"
    assert dataset.num_classes == 10
    assert dataset.input_shape == (64,)
    assert dataset.input_size == 64


def test_split_shapes_and_sample_counts(dataset):
    assert dataset.X_train.shape == (dataset.num_train_samples, 64)
    assert dataset.X_val.shape == (dataset.num_val_samples, 64)
    assert dataset.X_test.shape == (dataset.num_test_samples, 64)

    assert dataset.y_train.shape == (dataset.num_train_samples,)
    assert dataset.y_val.shape == (dataset.num_val_samples,)
    assert dataset.y_test.shape == (dataset.num_test_samples,)

    total = (
        dataset.num_train_samples
        + dataset.num_val_samples
        + dataset.num_test_samples
    )
    assert total == 1797


def test_array_dtypes(dataset):
    for X in (dataset.X_train, dataset.X_val, dataset.X_test):
        assert X.dtype == np.float32

    for y in (dataset.y_train, dataset.y_val, dataset.y_test):
        assert y.dtype == np.int64


def test_features_are_normalized_and_finite(dataset):
    for X in (dataset.X_train, dataset.X_val, dataset.X_test):
        assert np.isfinite(X).all()
        assert X.min() >= 0.0
        assert X.max() <= 1.0


def test_all_classes_appear_in_each_split(dataset):
    expected_classes = np.arange(10)

    for y in (dataset.y_train, dataset.y_val, dataset.y_test):
        np.testing.assert_array_equal(np.unique(y), expected_classes)


def test_splits_are_reproducible():
    loader = SklearnDigitsLoader()
    first = loader.load(seed=42)
    second = loader.load(seed=42)

    for name in (
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "y_test",
    ):
        np.testing.assert_array_equal(
            getattr(first, name),
            getattr(second, name),
        )


def test_different_seeds_change_the_split():
    loader = SklearnDigitsLoader()
    first = loader.load(seed=42)
    second = loader.load(seed=123)

    assert not np.array_equal(first.X_train, second.X_train)


def test_features_are_contiguous(dataset):
    for X in (dataset.X_train, dataset.X_val, dataset.X_test):
        assert X.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("test_size", "validation_size"),
    [
        (0.0, 0.2),
        (1.0, 0.2),
        (0.2, 0.0),
        (0.2, 1.0),
        (0.6, 0.4),
        (0.8, 0.3),
    ],
)
def test_invalid_split_sizes_raise(test_size, validation_size):
    with pytest.raises(ValueError):
        SklearnDigitsLoader(
            test_size=test_size,
            validation_size=validation_size,
        )