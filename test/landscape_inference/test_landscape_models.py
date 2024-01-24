"""Tests for landscape_inference/landscape_models.py."""
import torch

from src.landscape_inference import landscape_models

# pylint: disable=no-member, missing-function-docstring


def test_mse_loss():
    mse_loss = landscape_models.MSELoss()
    y = torch.Tensor([1, 2, 3, 4])
    y_hat = torch.Tensor([0, 5, 5, 6])
    variances = torch.Tensor([1, 2, 1, 1])
    loss = mse_loss(y, y_hat, variances)
    assert loss.numpy()[()] > 3.374 and loss.numpy()[()] < 3.376


def test_linear_regression_landscape_model():
    input_hparams = {"num_features": 8}
    lin_reg = landscape_models.LinearRegressionLandscapeModel(input_hparams)
    x = torch.Tensor([[[1, 0], [0, 1], [0, 1], [1, 0]], [[1, 0], [1, 0], [0, 1], [0, 1]]])
    assert list(lin_reg(x).shape) == [2]

    input_hparams = {"num_features": 8, "predict_variance": True}
    lin_reg = landscape_models.LinearRegressionLandscapeModel(input_hparams)
    fitnesses, var = lin_reg(x)
    assert list(fitnesses.shape) == [2]
    assert list(var.shape) == [2]


def test_cnn_landscape_model():
    input_hparams = {
        "num_features": 2,
        "num_channels": [3],
        "filter_size": [2],
        "intermediate_dim": 6,
    }
    cnn = landscape_models.CNNLandscapeModel(input_hparams)
    x = torch.Tensor(
        [[[1, 0], [0, 1], [0, 1], [1, 0]], [[1, 0], [1, 0], [0, 1], [0, 1]]]
    ).transpose(1, 2)
    assert list(cnn(x).shape) == [2]

    input_hparams = {
        "num_features": 2,
        "num_channels": [3],
        "filter_size": [2],
        "intermediate_dim": 6,
        "predict_variance": True,
    }
    cnn = landscape_models.CNNLandscapeModel(input_hparams)
    fitnesses, var = cnn(x)
    assert list(fitnesses.shape) == [2]
    assert list(var.shape) == [2]


def test_fnn_landscape_model():
    input_hparams = {"num_features": 8, "hidden_dim": [2]}
    fnn = landscape_models.FNNLandscapeModel(input_hparams)
    x = torch.Tensor([[[1, 0], [0, 1], [0, 1], [1, 0]], [[1, 0], [1, 0], [0, 1], [0, 1]]])
    assert list(fnn(x).shape) == [2]

    input_hparams = {"num_features": 8, "hidden_dim": [2], "predict_variance": True}
    fnn = landscape_models.FNNLandscapeModel(input_hparams)
    fitnesses, var = fnn(x)
    assert list(fitnesses.shape) == [2]
    assert list(var.shape) == [2]


def test_trivial_landscape_model():
    input_hparams = {"seq_length": 3, "num_amino_acids": 2}
    model = landscape_models.TrivialLandscapeModel(input_hparams)
    x = torch.Tensor([[[1, 0], [0, 1], [0, 1]], [[1, 0], [1, 0], [1, 0]]])
    fitnesses = model(x)
    assert list(fitnesses.shape) == [2]

    input_hparams = {"seq_length": 3, "predict_variance": True, "num_amino_acids": 2}
    model = landscape_models.TrivialLandscapeModel(input_hparams)
    fitnesses, var = model(x)
    assert list(fitnesses.shape) == [2]
    assert list(var.shape) == [2]
