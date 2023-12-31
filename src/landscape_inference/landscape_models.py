"""Model classes to train landscape models."""

# pylint: disable=no-name-in-module
import torch
from torch import nn

from src.common_utils import overwrite_hparams

# pylint: disable=no-member, too-many-ancestors, arguments-differ

ACTIVATION_DICT = {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}


class NegativeDataLoss(nn.Module):
    """A class to compute loss, incorpoating negative data information.

    Predictions incorporated as MSE loss. Negative data incorporated as a loss of 0
    if the upper bound is satisfied and sigmoid(x) - 0.5, with appropriate scaling,
    if the upper bound is not satisfied.

    Args:
        y: torch.Tensor, true y values, or upper bounds.
        y_hat: torch.Tensor, predicted y values
        var: torch.Tensor, variance on true y values (or 1 as a dummy)
        types: torch.Tensor, type of data (0 for fitness prediction and 1 for upper bound).
    """

    def forward(self, y, y_hat, var):
        """Computes loss with negative data."""
        loss = torch.sum(1 / var * (y_hat - y) ** 2)
        loss /= y.shape[0]
        return loss


class LinearRegressionLandscapeModel(nn.Module):
    """A simple linear regression model.

    This model uses the negative data loss function defined above to incorporate
    both positive and negative data into training a linear regression on
    the provided features.

    Attributes:
        linear: nn.Linear containing the linear layer.
        hparams: Dict containing hyperparameters for the model.
    """

    DEFAULT_HPARAMS = {
        "num_features": 20000,  # Number of features
        "predict_variance": False,  # Whether to predict a variance
    }

    def __init__(self, hparams=None):
        """Builds a model specified based on the given hyperparameters.

        Args:
            hparams: Dict containing hyperparameters for the model. Unspecified hyperparameters
                will be replaced as in DEFAULT_HPARAMS. See DEFAULT_HPARAMS for descriptions.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        output_dim = 2 if self.hparams["predict_variance"] else 1

        self.linear = nn.Linear(self.hparams["num_features"], output_dim)

    def forward(self, x):
        """Runs a single forward pass of the model through the linear layer.

        Args:
            x: torch.tensor with featurization of the genotype.
        """
        output = self.linear(x.reshape(x.shape[0], -1))
        if self.hparams["predict_variance"]:
            return output[:, 0], output[:, 1]
        return output.squeeze()


class CNNLandscapeModel(nn.Module):
    """A CNN model.

    This model uses the negative data loss function defined above to incorporate
    both positive and negative data into training a CNN on the provided features.

    Attributes:
        cnn: nn.Sequential module containing the CNN.
        output_nn: nn.Sequential module containing top-layer output.
        hparams: Dict containing hyperparameters for the model.
    """

    DEFAULT_HPARAMS = {
        "num_conv_layers": 1,  # Number of convolutional layers
        "num_features": 20,  # Number of features (amino acids for one-hot or per-amino-acid embedding dimension otherwise)
        "num_channels": [4],  # Number of channels in each convolutional layer
        "filter_size": [2],  # Filter size in each convolutional layer
        "activation": "elu",  # Activation after each convolutional layer, either elu, relu, or tanh
        "dropout": 0.2,  # Dropout rate for model
        "predict_variance": False,  # Whether to predict a variance
        "intermediate_dim": 2,  # Intermediate dim for final linear layers before output
    }

    def __init__(self, hparams=None):
        """Builds a model specified based on the given hyperparameters.

        Args:
            hparams: Dict containing hyperparameters for the model. Unspecified hyperparameters
                will be replaced as in DEFAULT_HPARAMS. See DEFAULT_HPARAMS for descriptions.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        output_dim = 2 if self.hparams["predict_variance"] else 1

        layers = []
        for i in range(self.hparams["num_conv_layers"]):
            if i == 0:
                layers += [
                    nn.Conv1d(
                        self.hparams["num_features"],
                        self.hparams["num_channels"][i],
                        self.hparams["filter_size"][i],
                        padding="same",
                    )
                ]
            else:
                layers += [
                    nn.Conv1d(
                        self.hparams["num_channels"][i - 1],
                        self.hparams["num_channels"][i],
                        self.hparams["filter_size"][i],
                        padding="same",
                    )
                ]
            layers += [nn.BatchNorm1d(self.hparams["num_channels"][i])]
            layers += [ACTIVATION_DICT[self.hparams["activation"]]]
        self.cnn = nn.Sequential(*layers)
        self.embedding_nn = nn.Sequential(
            nn.Linear(self.hparams["num_channels"][-1], self.hparams["intermediate_dim"]),
            ACTIVATION_DICT[self.hparams["activation"]],
        )
        self.output_nn = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.hparams["dropout"]),
            nn.Linear(self.hparams["intermediate_dim"], output_dim),
        )

    def forward(self, x):
        """Runs a single forward pass of the model.

        Args:
            x: torch.tensor with featurization of the genotype. Must be 3D (data in batch x features per amino acid x sequence length).
        """
        x = self.cnn(x).permute(0, 2, 1)
        x = self.embedding_nn(x)
        x = torch.permute(x, (0, 2, 1))
        x = nn.functional.max_pool1d(x, kernel_size=x.shape[2])
        x = self.output_nn(x)
        if self.hparams["predict_variance"]:
            return x[:, 0], x[:, 1]
        return x.squeeze()


class FNNLandscapeModel(nn.Module):
    """A feedforward network model.

    This model uses the negative data loss function defined above to incorporate
    both positive and negative data into training a FNN on the provided features.

    Attributes:
        model: nn.Sequential module containing the FNN.
        hparams: Dict containing hyperparameters for the model.
    """

    DEFAULT_HPARAMS = {
        "num_layers": 1,  # Number of layers (excluding last layer)
        "num_features": 20,  # Number of features (amino acids for one-hot or per-amino-acid embedding dimension otherwise)
        "hidden_dim": [4],  # Hidden dimension for each layer
        "activation": "elu",  # Activation after each layer, either elu, relu, or tanh
        "dropout": 0,  # Dropout rate for model
        "predict_variance": False,  # Whether to predict a variance
    }

    def __init__(self, hparams=None):
        """Builds a model specified based on the given hyperparameters.

        Args:
            hparams: Dict containing hyperparameters for the model. Unspecified hyperparameters
                will be replaced as in DEFAULT_HPARAMS. See DEFAULT_HPARAMS for descriptions.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        output_dim = 2 if self.hparams["predict_variance"] else 1

        layers = []
        for i in range(self.hparams["num_layers"]):
            if i == 0:
                layers += [nn.Linear(self.hparams["num_features"], self.hparams["hidden_dim"][i])]
            else:
                layers += [
                    nn.Linear(self.hparams["hidden_dim"][i - 1], self.hparams["hidden_dim"][i])
                ]
            layers += [ACTIVATION_DICT[self.hparams["activation"]]]
            layers += [nn.Dropout(self.hparams["dropout"])]
        layers += [nn.Linear(self.hparams["hidden_dim"][-1], output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Runs a single forward pass of the model.

        Args:
            x: torch.tensor with featurization of the genotype.
        """
        output = self.model(x.reshape(x.shape[0], -1))
        if self.hparams["predict_variance"]:
            return output[:, 0], output[:, 1]
        return output.squeeze()


class TrivialLandscapeModel(nn.Module):
    """A trivial landscape model that is simply a dictionary mapping sequence to fitness.

    This model has no ability to generalize since it can only directly map sequence
    to fitness.

    Do not use on sequences of larger than roughly 4 amino acids.

    Attributes:
        fitnesses: nn.Embedding containing fitnesses (and/or variance).
        mult_factor: torch.Tensor to convert from one-hot embedding to single integer.
    """

    DEFAULT_HPARAMS = {
        "seq_length": 3,  # Length of sequence
        "num_amino_acids": 20,  # Number of amino acids
        "predict_variance": False,  # Whether to predict a variance
    }

    def __init__(self, hparams=None):
        """Builds a model specified based on the given hyperparameters.

        Args:
            hparams: Dict containing hyperparameters for the model. Unspecified hyperparameters
                will be replaced as in DEFAULT_HPARAMS. See DEFAULT_HPARAMS for descriptions.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        output_dim = 2 if self.hparams["predict_variance"] else 1
        self.fitnesses = nn.Embedding(
            self.hparams["num_amino_acids"] ** self.hparams["seq_length"], output_dim
        )
        mult_factor = (
            torch.arange(self.hparams["num_amino_acids"])
            .unsqueeze(0)
            .repeat(self.hparams["seq_length"], 1)
        )
        powers = (
            torch.tensor(
                [self.hparams["num_amino_acids"] ** i for i in range(self.hparams["seq_length"])]
            )
            .unsqueeze(1)
            .repeat(1, self.hparams["num_amino_acids"])
        )
        mult_factor = mult_factor * powers
        self.register_buffer("mult_factor", mult_factor.to(torch.float32))

    def forward(self, x):
        """Runs a single forward pass of the model.

        Args:
            x: torch.Tensor with one-hot featurization of the sequence.
        """
        # convert to integer
        x = torch.einsum("ijk, jk -> i", x, self.mult_factor).long()
        output = self.fitnesses(x)
        if self.hparams["predict_variance"]:
            return output[:, 0], output[:, 1]
        return output.squeeze()
