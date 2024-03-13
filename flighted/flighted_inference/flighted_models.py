"""Models and layers for running FLIGHTED inference from DHARMA and selection data."""

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from torch import nn
from torch.distributions import transforms
from torch.nn import functional as F

from flighted.common_utils import overwrite_hparams
from flighted.landscape_inference import landscape_models

# pylint: disable=no-member, too-many-lines, invalid-name, not-callable


class IndependentNoiseModel(nn.Module):
    """Generates a sample DHARMA sequence given a fitness value.

    This model assumes independence at all positions in the canvas
    sequence. All rates are stored as logits, not as probabilities. Mutation probabilities
    are modeled as a one-hot distribution with the given logits.

    Attributes:
        hparams: Dict containing hparams.
        mutation_rates: pyro.param containing baseline mutation rates (not C to T) at every
            residue in the canvas sequence.
        baseline_edits: pyro.param containing baseline edit rates at 0 activity at every
            residue in the canvas sequence.
        slope_edits: pyro.param containing the slope of edit rate vs. activity at every
            residue in the canvas sequence.
    """

    DEFAULT_HPARAMS = {
        "num_residues": 190,  # total number of residues
        "cytosine_residues": [],  # which residues are cytosines
        "initial_mut_rate": -4,  # initial mutation rate (as logits)
        "baseline_unedited": False,  # whether to include a probability of no edits as a baseline
    }

    def __init__(self, hparams=None):
        """Builds a model based on the specified hyperparameters.

        Args:
            hparams: Dict containing defined hyperparameters. See DEFAULT_HPARAMS
                for details.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        self.mutation_rates_parameter = torch.nn.Parameter(
            self.hparams["initial_mut_rate"] * torch.ones((self.hparams["num_residues"],))
        )
        self.mutation_rates = pyro.param("mutation_rates", self.mutation_rates_parameter)
        self.baseline_edits_parameter = torch.nn.Parameter(
            self.hparams["initial_mut_rate"] * torch.ones((len(self.hparams["cytosine_residues"]),))
        )
        self.baseline_edits = pyro.param("baseline_edits", self.baseline_edits_parameter)
        self.slope_edits_parameter = torch.nn.Parameter(
            torch.ones((len(self.hparams["cytosine_residues"]),))
        )
        if self.hparams["baseline_unedited"]:
            self.baseline_unedited_parameter = torch.nn.Parameter(
                torch.ones(
                    1,
                )
            )
            self.baseline_unedited = pyro.param(
                "baseline_unedited", self.baseline_unedited_parameter
            )
        self.slope_edits = pyro.param("slope_edits", self.slope_edits_parameter)

    def forward(self, fitness, dharma_output=None):
        """Computes a predicted DHARMA sequence output from a given fitness value.

        Args:
            fitness: torch.Tensor with fitnesses, with shape batch_size.
            dharma_output: torch.Tensor with observed DHARMA output
                with shape batch_size x num_residues x 3 (given one-hot featurization).

        Returns:
            Sampled DHARMA output as a torch.Tensor with shape batch_size x num_residues x 3.
        """
        fitness_adj = self.baseline_edits[None, :] + self.slope_edits[None, :] * fitness[:, None]
        logits = torch.zeros(
            (fitness.shape[0], self.hparams["num_residues"], 3), device=fitness.device
        )
        logits[:, :, 2] = self.mutation_rates[None, :]
        logits[:, :, 1] = -np.inf
        if self.hparams["baseline_unedited"]:
            edited = pyro.sample(
                "edited",
                dist.Bernoulli(
                    logits=self.baseline_unedited
                    * torch.ones((fitness.shape[0]), device=fitness.device)
                ),
            )
            fitness_adj = torch.cat(
                (
                    -np.inf
                    * torch.ones(
                        (1, fitness.shape[0], len(self.hparams["cytosine_residues"])),
                        device=fitness.device,
                    ),
                    fitness_adj.unsqueeze(0),
                )
            )
            fitness_adj = fitness_adj[edited.long(), torch.arange(fitness.shape[0])]
        logits[:, self.hparams["cytosine_residues"], 1] = fitness_adj
        return pyro.sample(
            "dharma", dist.OneHotCategorical(logits=logits).to_event(2), obs=dharma_output
        )


class DHARMAToFitnessFNN(nn.Module):
    """Predicts fitness given a DHARMA readout.

    A FNN model with two outputs that predicts fitness given a DHARMA readout.
    Predicts a fitness mean and variance for every input, which is used for a normal
    distribution downstream.

    Attributes:
        hparams: Dict containing hyperparameters for this model.
        model: nn.Module containing the model as defined by provided hyperparameters.
    """

    DEFAULT_HPARAMS = {
        "num_residues": 190,  # total number of residues in canvas
        "cytosine_residues": [],  # which residues are cytosines
        "dharma_fitness_num_layers": 2,  # number of layers
        "dharma_fitness_hidden_dim": [10],  # hidden dimensions
    }

    def __init__(self, hparams=None):
        """Builds a model based on the specified hyperparameters.

        Args:
            hparams: Dict containing defined hyperparameters. See DEFAULT_HPARAMS
                for details.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        self.hparams["num_cytosine_residues"] = len(self.hparams["cytosine_residues"])

        assert (
            len(self.hparams["dharma_fitness_hidden_dim"])
            == self.hparams["dharma_fitness_num_layers"] - 1
        )

        layers = [nn.Flatten()]
        hidden_dims = self.hparams["dharma_fitness_hidden_dim"]
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                layers += [nn.Linear(self.hparams["num_cytosine_residues"], dim)]
            else:
                layers += [nn.Linear(hidden_dims[i - 1], dim)]
            layers += [nn.ReLU()]

        if self.hparams["dharma_fitness_num_layers"] > 1:
            layers += [nn.Linear(hidden_dims[-1], 2)]
        else:
            layers += [nn.Linear(self.hparams["num_cytosine_residues"], 2)]

        self.model = nn.Sequential(*layers)

    def forward(self, dharma_output):
        """Computes fitness mean and variance given the DHARMA readout.
        ll
        Args:
            dharma_output: torch.Tensor with observed DHARMA output with shape batch_size x
                num_residues x 3.

        Returns:
            Predicted mean and variance of fitness.
        """
        # Reprocess to eliminate irrelevant info
        dharma_processed_output = dharma_output[:, :, :2].clone().detach()
        dharma_processed_output[:, :, 0] += dharma_output[:, :, 2]
        dharma_processed_output = dharma_processed_output[
            :, self.hparams["cytosine_residues"], :
        ].to(torch.float32)
        dharma_processed_output = torch.argmax(dharma_processed_output, dim=2).to(torch.float32)

        fitness_pred = self.model(dharma_processed_output)
        return fitness_pred[:, 0], fitness_pred[:, 1]


class FLIGHTED_DHARMA(nn.Module):
    """The core FLIGHTED model for DHARMA.

    Attributes:
        landscape_model: nn.Module that predicts fitness mean and variance given protein
            sequence/featurization.
        dharma_model: nn.Module that predicts DHARMA sequence output from fitness.
        dharma_reverse_model: nn.Module that predicts fitness from DHARMA readout.
    """

    DEFAULT_HPARAMS = {
        "dharma_model": "independent",  # DHARMA generation model
        "dharma_reverse": "fnn",  # DHARMA reverse model
        "protein_landscape_model": "embedding",  # protein landscape model
        "aux_loss_multiplier": 0.1,  # multiplier for auxiliary classifier loss
        "fitness_cutoff": -2,  # fitness cutoff below which to set DHARMA edit prob as 0
        "supervised": False,  # whether to train with supervision or not
    }
    DHARMA_MODELS = {"independent": IndependentNoiseModel}
    DHARMA_REVERSE_MODELS = {"fnn": DHARMAToFitnessFNN}
    PROTEIN_LANDSCAPE_MODELS = {
        "linear": landscape_models.LinearRegressionLandscapeModel,
        "cnn": landscape_models.CNNLandscapeModel,
        "fnn": landscape_models.FNNLandscapeModel,
        "embedding": landscape_models.TrivialLandscapeModel,
    }

    def __init__(self, hparams=None):
        """Builds a model based on the specific hyperparameters.

        Args:
            hparams: Dict containing defined hyperparameters. See DEFAULT_HPARAMS
                for details.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        assert self.hparams["predict_variance"]

        self.landscape_model = self.PROTEIN_LANDSCAPE_MODELS[
            self.hparams["protein_landscape_model"]
        ](hparams=self.hparams)
        self.dharma_model = self.DHARMA_MODELS[self.hparams["dharma_model"]](hparams=self.hparams)
        self.dharma_reverse_model = self.DHARMA_REVERSE_MODELS[self.hparams["dharma_reverse"]](
            hparams=self.hparams
        )

    def model(self, seqs, dharmas):
        """Runs a model, generating DHARMA from the sequence.

        Args:
            seqs: torch.Tensor with featurization of the sequence.
            dharmas: torch.Tensor with observed DHARMA output with shape batch_size x
                num_residues x 3.

        Returns:
           Sampled DHARMA output, with same shapes as above.
        """
        pyro.module("landscape_model", self.landscape_model)
        pyro.module("dharma_model", self.dharma_model)
        with pyro.plate("data", seqs.shape[0]):
            fitnesses, variance = self.landscape_model(seqs)
            fitness_sampled = pyro.sample("fitness", dist.Normal(fitnesses, F.softplus(variance)))
            if len(fitness_sampled.shape) == 2:
                fitness_sampled = fitness_sampled[0]
            fitness_sampled_dharma = torch.clone(fitness_sampled)
            fitness_sampled_dharma[fitnesses <= self.hparams["fitness_cutoff"]] = self.hparams[
                "fitness_cutoff"
            ]
            dharma_sampled = self.dharma_model(fitness_sampled_dharma, dharma_output=dharmas)
            return dharma_sampled

    def guide(self, seqs, dharmas):
        """Guide (or variational distribution) used for SVI training.

        Args:
            seqs: torch.Tensor with featurization of the sequence.
            dharmas: torch.Tensor with observed DHARMA output with shape batch_size x
                num_residues x 3.
        """
        pyro.module("dharma_reverse_model", self.dharma_reverse_model)
        with pyro.plate("data", dharmas.shape[0]):
            fitness_mean, fitness_variance = self.dharma_reverse_model(dharmas)
            # This section requires that seqs be grouped with all datapoints for the same sequence
            # in order. Satisfied by the current implementation.
            if self.hparams["combine_reads"] and self.hparams["average_batch"]:
                indices_nonunique = torch.cumsum(
                    torch.bincount(torch.unique(seqs, dim=0, return_inverse=True)[1]), dim=0
                )
                fitness_mean_split = torch.tensor_split(
                    fitness_mean, indices_nonunique.to(torch.device("cpu"))
                )[:-1]
                fitness_variance_split = torch.tensor_split(
                    fitness_variance, indices_nonunique.to(torch.device("cpu"))
                )[:-1]
                fitness_variance_combined = torch.tensor(
                    list(1 / torch.sum(1 / F.softplus(var)) for var in fitness_variance_split),
                    device=fitness_variance.device,
                )
                fitness_mean_combined = torch.tensor(
                    list(
                        torch.sum(mean / F.softplus(var) * var_combined)
                        for (mean, var, var_combined) in zip(
                            fitness_mean_split, fitness_variance_split, fitness_variance_combined
                        )
                    ),
                    device=fitness_mean.device,
                )
                indices_unraveled = torch.unique(seqs, dim=0, return_inverse=True)[1]
                fitness_mean_combined = fitness_mean_combined[indices_unraveled]
                fitness_variance_combined = fitness_variance_combined[indices_unraveled]
            else:
                fitness_mean_combined = fitness_mean
                fitness_variance_combined = fitness_variance
            pyro.sample(
                "fitness", dist.Normal(fitness_mean_combined, F.softplus(fitness_variance_combined))
            )

    def forward(self, seqs):
        """Simply returns fitness mean and variance for given sequences.

        Args:
            seqs: torch.Tensor with featurization of the sequence.
        """
        return self.landscape_model(seqs)

    def infer_fitness_from_dharma(self, dharmas):
        """Infer fitness from multiple DHARMA readouts for a single sequence.

        Args:
            dharmas: torch.Tensor with observed DHARMA output with shape batch_size x
                num_residues x 3.

        Returns:
            Mean and variance of predicted fitness.
        """
        fitness_mean, fitness_variance = self.dharma_reverse_model(dharmas)
        # combine Gaussians
        total_fitness_variance = 1 / torch.sum(1 / F.softplus(fitness_variance))
        total_fitness_mean = (
            torch.sum(fitness_mean / F.softplus(fitness_variance)) * total_fitness_variance
        )
        return total_fitness_mean, total_fitness_variance


class SelectionNoiseModel(nn.Module):
    """Generates a selection result given fitness values for all variants in an experiment.

    This model uses the sampling-based selection model, where the samples are drawn
    before and after selection and fitness refers to probabilities drawn for each variant.

    Attributes:
        hparams: Dict containing hparams.
    """

    DEFAULT_HPARAMS = {
        "total_population": 1e11,  # total initial population
        "min_prob": 1e-3,  # minimum probability cutoff for fitness
        "max_prob": 0.999,  # maximum probability cutoff for fitness
    }

    def __init__(self, hparams=None):
        """Builds a model based on the specified hyperparameters.

        Args:
            hparams: Dict containing defined hyperparameters. See DEFAULT_HPARAMS
                for details.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)

    def forward(self, fitness, selection_output):
        """Computes a predicted selection output from a given fitness value.

        Args:
            fitness: torch.Tensor with fitnesses, with shape batch_size x num_variants.
            selection_output: torch.Tensor with observed output from selection experiments,
                with shape batch_size x num_variants x 5 (identity, pre-selection count,
                post-selection count, number of samples initially, number of samples drawn
                post-selection).
        """
        for i in pyro.plate("exps", len(selection_output)):
            pops_i = pyro.param(
                f"pops_{i}",
                lambda: torch.rand((selection_output.shape[1])).to(selection_output.device),
                constraint=constraints.simplex,
            )
            pyro.sample(
                f"initial_sample_{i}",
                dist.Multinomial(int(selection_output[i, 0, 3]), probs=pops_i.to(torch.float64)),
                obs=selection_output[i, :, 1].to(torch.float64),
            )
            fitnesses_sample = fitness[i, selection_output[i, :, 0].to(torch.long)]
            fitnesses_sample = torch.clamp(
                fitnesses_sample, min=self.hparams["min_prob"], max=self.hparams["max_prob"]
            )
            selected_all = pyro.sample(
                f"selected_all_{i}",
                dist.Binomial(
                    torch.floor(self.hparams["total_population"] * pops_i), probs=fitnesses_sample
                ).to_event(1),
            )
            pyro.sample(
                f"selected_sample_{i}",
                dist.Multinomial(
                    int(selection_output[i, 0, 4]), probs=selected_all.to(torch.float64)
                ),
                obs=selection_output[i, :, 2].to(torch.float64),
            )


class SelectionToFitnessFNN(nn.Module):
    """Predicts fitness given a selection result.

    A FNN model with two outputs per variant that predicts fitness given a selection result.
    Predicts a fitness mean and variance for every input, which is used for a normal
    distribution downstream.

    Attributes:
        hparams: Dict containing hyperparameters for this model.
        model: nn.Module containing the model as defined by provided hyperparameters.
    """

    DEFAULT_HPARAMS = {
        "selection_fitness_num_layers_variance": 1,  # number of layers (variance)
        "selection_fitness_hidden_dim_variance": [],  # hidden dimensions (variance)
        "include_N_sample": False,  # whether to include number of samples as a feature
        "max_cutoff": 20,  # max cutoff on enrichment ratio if initial sample is 0
        "enr_scale": 20,  # scale to apply to enrichment ratio
    }

    def __init__(self, hparams=None):
        """Builds a model based on the specified hyperparameters.

        Args:
            hparams: Dict containing defined hyperparameters. See DEFAULT_HPARAMS
                for details.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)

        assert (
            len(self.hparams["selection_fitness_hidden_dim_variance"])
            == self.hparams["selection_fitness_num_layers_variance"] - 1
        )

        hidden_dims = self.hparams["selection_fitness_hidden_dim_variance"]
        layers = []
        if self.hparams["include_N_sample"]:
            input_dim = 5
        else:
            input_dim = 3
        output_dim = 1
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                layers += [nn.Linear(input_dim, dim)]
            else:
                layers += [nn.Linear(hidden_dims[i - 1], dim)]
            layers += [nn.ReLU()]

        if self.hparams["selection_fitness_num_layers_variance"] > 1:
            layers += [nn.Linear(hidden_dims[-1], output_dim)]
        else:
            layers += [nn.Linear(input_dim, output_dim)]

        self.model_variance = nn.Sequential(*layers)

    def forward(self, selection_output):
        """Computes fitness mean and variance given the selection readout.

        Args:
            selection_output: torch.Tensor with observed selection output with shape batch_size x
                num_variants x 5.

        Returns:
            Predicted mean and variance of fitness, with shape batch_size x num_variants.
        """
        selection_output_processed = torch.zeros_like(selection_output).to(torch.float32)
        selection_output_processed[:, :, :2] = selection_output[:, :, 1:3].to(torch.float32)
        selection_output_processed[:, :, 0] /= selection_output[:, :, 3]
        selection_output_processed[:, :, 1] /= selection_output[:, :, 4]
        selection_output_processed[:, :, 3:] = torch.log(selection_output[:, :, 3:] / 1e8)
        enr_ratio = torch.where(
            selection_output_processed[:, :, 0] == 0,
            torch.where(
                selection_output_processed[:, :, 1] == 0,
                torch.ones_like(selection_output_processed[:, :, 1]).to(torch.float32),
                self.hparams["max_cutoff"]
                * torch.ones_like(selection_output_processed[:, :, 1]).to(torch.float32),
            ),
            selection_output_processed[:, :, 1] / selection_output_processed[:, :, 0],
        )
        selection_output_processed[:, :, 2] = enr_ratio

        fitness_pred_mean = torch.special.logit(enr_ratio / self.hparams["enr_scale"], eps=0.001)

        if self.hparams["include_N_sample"]:
            fitness_pred_variance = self.model_variance(selection_output_processed).squeeze(-1)
        else:
            fitness_pred_variance = self.model_variance(
                selection_output_processed[:, :, :3]
            ).squeeze(-1)
        return fitness_pred_mean, fitness_pred_variance


class FLIGHTED_Selection(nn.Module):
    """The core FLIGHTED model for single-step selection data.

    Attributes:
        landscape_model: nn.Module that predicts fitness mean and variance given protein
            sequence/featurization.
        selection_model: nn.Module that predicts selection output from fitness.
        selection_reverse_model: nn.Module that predicts fitness from selection output.
    """

    DEFAULT_HPARAMS = {
        "selection_model": "sampling",  # selection generation model
        "selection_reverse": "fnn",  # selection reverse model
        "protein_landscape_model": "embedding",  # protein landscape model
        "total_population": 1e11,  # total initial population
    }
    SELECTION_MODELS = {"sampling": SelectionNoiseModel}
    SELECTION_REVERSE_MODELS = {"fnn": SelectionToFitnessFNN}
    PROTEIN_LANDSCAPE_MODELS = {
        "linear": landscape_models.LinearRegressionLandscapeModel,
        "cnn": landscape_models.CNNLandscapeModel,
        "fnn": landscape_models.FNNLandscapeModel,
        "embedding": landscape_models.TrivialLandscapeModel,
    }

    def __init__(self, hparams=None):
        """Builds a model based on the specific hyperparameters.

        Args:
            hparams: Dict containing defined hyperparameters. See DEFAULT_HPARAMS
                for details.
        """
        super().__init__()
        self.hparams = overwrite_hparams(hparams, self.DEFAULT_HPARAMS)
        assert self.hparams["predict_variance"]

        self.landscape_model = self.PROTEIN_LANDSCAPE_MODELS[
            self.hparams["protein_landscape_model"]
        ](hparams=self.hparams)
        self.selection_model = self.SELECTION_MODELS[self.hparams["selection_model"]](
            hparams=self.hparams
        )
        self.selection_reverse_model = self.SELECTION_REVERSE_MODELS[
            self.hparams["selection_reverse"]
        ](hparams=self.hparams)

    def model(self, seqs, selection_output):
        """Runs a model, generating selection outputs from the sequence.

        Args:
            seqs: torch.Tensor with featurization of the sequence.
            selection_output: torch.Tensor with observed selection output with shape batch_size x
                num_variants x 5.
        """
        pyro.module("landscape_model", self.landscape_model)
        pyro.module("selection_model", self.selection_model)
        indices = (
            torch.unique(selection_output[:, :, 0]).to(torch.int64).to(selection_output.device)
        )
        fitnesses, variance = self.landscape_model(seqs)
        fitness_sampled_scattered = torch.zeros_like(fitnesses)
        fitnesses = fitnesses[indices]
        variance = variance[indices]
        fitness_sampled = pyro.sample(
            "fitness",
            dist.TransformedDistribution(
                dist.Normal(fitnesses, F.softplus(variance)), [transforms.SigmoidTransform()]
            ).to_event(1),
        )
        fitness_sampled = fitness_sampled_scattered.scatter(0, indices, fitness_sampled)
        fitness_sampled = fitness_sampled.unsqueeze(0).repeat(selection_output.shape[0], 1)
        self.selection_model(fitness_sampled, selection_output=selection_output)

    def guide(self, seqs, selection_output):
        """Guide (or variational distribution) used for SVI training.

        Args:
            seqs: torch.Tensor with featurization of the sequence.
            selection_output: torch.Tensor with observed selection output with shape batch_size x
                num_variants x 5.
        """
        num_variants = seqs.shape[0]
        pyro.module("selection_reverse_model", self.selection_reverse_model)
        fitness_mean, fitness_variance = self.selection_reverse_model(selection_output)
        fitness_mean_scattered = torch.zeros([selection_output.shape[0], num_variants]).to(
            selection_output.device
        )
        indices_nonunique = selection_output[:, :, 0].to(torch.int64).to(selection_output.device)
        fitness_mean_scattered = fitness_mean_scattered.scatter(1, indices_nonunique, fitness_mean)
        fitness_variance_scattered = torch.ones([selection_output.shape[0], num_variants]).to(
            selection_output.device
        ) * np.float64("nan")
        fitness_variance_scattered = fitness_variance_scattered.scatter(
            1, indices_nonunique, fitness_variance
        )
        fitness_variance_combined = 1 / torch.nansum(
            1 / F.softplus(fitness_variance_scattered), axis=0
        )
        fitness_mean_combined = (
            torch.nansum(fitness_mean_scattered / F.softplus(fitness_variance_scattered), axis=0)
            * fitness_variance_combined
        )
        indices = (
            torch.unique(selection_output[:, :, 0]).to(torch.int64).to(selection_output.device)
        )
        fitness_mean_combined = fitness_mean_combined[indices]
        fitness_variance_combined = fitness_variance_combined[indices]
        fitness = pyro.sample(
            "fitness",
            dist.TransformedDistribution(
                dist.Normal(fitness_mean_combined, fitness_variance_combined),
                [transforms.SigmoidTransform()],
            ).to_event(1),
        )
        fitness_sampled_scattered = torch.zeros(seqs.shape[0]).to(seqs.device)
        fitness_sampled = fitness_sampled_scattered.scatter(0, indices, fitness)
        fitness_sampled = fitness_sampled.unsqueeze(0).repeat(selection_output.shape[0], 1)
        for i in pyro.plate("exps", selection_output.shape[0]):
            pops_i = pyro.param(
                f"pops_{i}",
                lambda: torch.rand((selection_output.shape[1])).to(selection_output.device),
                constraint=constraints.simplex,
            )
            fitness_i = fitness_sampled[i, selection_output[i, :, 0].to(torch.long)]
            pyro.sample(
                f"selected_all_{i}",
                dist.Binomial(
                    torch.floor(self.hparams["total_population"] * pops_i), probs=fitness_i
                ).to_event(1),
            )

    def forward(self, seqs):
        """Simply returns fitness mean and variance for given sequences.

        Args:
            seqs: torch.Tensor with featurization of the sequence.
        """
        return self.landscape_model(seqs)
