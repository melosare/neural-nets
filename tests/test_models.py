import pytest

from models.mlp import MLPConfig

def test_mlp_layer_sizes():
    config = MLPConfig(
        input_size=64,
        hidden_sizes=(128, 64),
        num_classes=10,
    )

    assert config.layer_sizes == (64, 128, 64, 10)
    assert config.num_hidden_layers == 2


def test_mlp_parameter_count():
    config = MLPConfig(
        input_size=64,
        hidden_sizes=(128, 64),
        num_classes=10,
    )

    assert config.parameter_count == 17226


def test_mlp_without_hidden_layers():
    config = MLPConfig(
        input_size=64,
        hidden_sizes=(),
        num_classes=10,
    )

    assert config.layer_sizes == (64, 10)
    assert config.num_hidden_layers == 0
    assert config.parameter_count == 650


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": 0, "hidden_sizes": (32,), "num_classes": 10},
        {"input_size": 64, "hidden_sizes": (0,), "num_classes": 10},
        {"input_size": 64, "hidden_sizes": (-1,), "num_classes": 10},
        {"input_size": 64, "hidden_sizes": (32,), "num_classes": 1},
        {
            "input_size": 64,
            "hidden_sizes": (32,),
            "num_classes": 10,
            "activation": "unknown",
        },
    ],
)
def test_invalid_mlp_configuration(kwargs):
    with pytest.raises(ValueError):
        MLPConfig(**kwargs)


def test_config_is_immutable():
    config = MLPConfig(
        input_size=64,
        hidden_sizes=(32,),
        num_classes=10,
    )

    with pytest.raises(AttributeError):
        config.input_size = 128