"""Tests for flighted_inference/flighted_models.py."""
import pyro
import torch

from src.flighted_inference import flighted_models

# pylint: disable=missing-function-docstring, no-member


def test_independent_noise_model():
    pyro.clear_param_store()
    model = flighted_models.IndependentNoiseModel(
        {"num_residues": 10, "cytosine_residues": [0, 1, 2, 3, 4]}
    )
    fitness = torch.Tensor([1, 2, 3, 4])
    predicted_dharma = model(fitness)
    assert predicted_dharma.shape == (4, 10, 3)
    assert torch.all(torch.sum(predicted_dharma, axis=2) == 1)


def test_dharma_to_fitness_model():
    pyro.clear_param_store()
    model = flighted_models.DHARMAToFitnessFNN(
        {"num_residues": 6, "cytosine_residues": [0, 1, 2, 3, 5]}
    )
    dharma_output = torch.Tensor(
        [
            [[1, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]],
        ]
    ).transpose(1, 2)

    mean, variance = model(dharma_output)
    assert mean.shape == (2,)
    assert variance.shape == (2,)


def test_flighted_dharma():
    pyro.clear_param_store()
    hparams = {
        "num_residues": 6,
        "cytosine_residues": [0, 1, 2, 3, 5],
        "seq_length": 3,
        "num_amino_acids": 2,
        "predict_variance": True,
        "combine_reads": False,
        "supervised": True,
    }
    flighted_dharma = flighted_models.FLIGHTED_DHARMA(hparams)
    dharma_output = torch.Tensor(
        [
            [[1, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            [[1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]],
        ]
    ).transpose(1, 2)
    seqs = torch.Tensor([[[1, 0], [0, 1], [0, 1]], [[1, 0], [1, 0], [1, 0]]])
    dharma_sampled = flighted_dharma.model(seqs, dharma_output)
    assert torch.allclose(dharma_sampled, dharma_output)

    dharma_sampled = flighted_dharma.model(seqs, dharma_output)
    assert torch.allclose(dharma_sampled, dharma_output)

    flighted_dharma.guide(seqs, dharma_output)

    mean, var = flighted_dharma(seqs)
    assert mean.shape == (2,)
    assert var.shape == (2,)

    mean, var = flighted_dharma.infer_fitness_from_dharma(dharma_output)
    assert var.detach().numpy()[()] > 0


def test_selection_noise_model():
    pyro.clear_param_store()
    model = flighted_models.SelectionNoiseModel()
    fitness = torch.Tensor([[0.1, 0.2, 0.3, 0.4], [0.3, 0.1, 0.6, 0.2]])
    selection_output = torch.Tensor(
        [
            [[0, 5, 3, 16, 8], [1, 4, 2, 16, 8], [2, 3, 1, 16, 8], [3, 4, 2, 16, 8]],
            [[0, 6, 6, 17, 9], [1, 4, 1, 17, 9], [2, 3, 1, 17, 9], [3, 4, 1, 17, 9]],
        ]
    )
    model(fitness, selection_output)


def test_selection_to_fitness_model():
    pyro.clear_param_store()
    model = flighted_models.SelectionToFitnessFNN()
    selection_output = torch.Tensor(
        [
            [[0, 5, 3, 16, 8], [1, 4, 2, 16, 8], [2, 3, 1, 16, 8], [3, 4, 2, 16, 8]],
            [[0, 6, 6, 17, 9], [1, 4, 1, 17, 9], [2, 3, 1, 17, 9], [3, 4, 1, 17, 9]],
        ]
    )
    mean, variance = model(selection_output)
    assert mean.shape == (2, 4)
    assert variance.shape == (2, 4)


def test_flighted_selection():
    pyro.clear_param_store()
    hparams = {
        "seq_length": 3,
        "num_amino_acids": 2,
        "predict_variance": True,
    }
    flighted_selection = flighted_models.FLIGHTED_Selection(hparams)
    selection_output = torch.Tensor(
        [
            [[0, 5, 3, 16, 8], [1, 4, 2, 16, 8], [2, 3, 1, 16, 8], [3, 4, 2, 16, 8]],
            [[0, 6, 6, 17, 9], [1, 4, 1, 17, 9], [2, 3, 1, 17, 9], [3, 4, 1, 17, 9]],
        ]
    )
    seqs = torch.Tensor(
        [
            [[1, 0], [0, 1], [0, 1]],
            [[1, 0], [1, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 0]],
            [[0, 1], [1, 0], [1, 0]],
        ]
    )
    flighted_selection.model(seqs, selection_output)
    flighted_selection.guide(seqs, selection_output)
    mean, var = flighted_selection(seqs)
    assert mean.shape == (4,)
    assert var.shape == (4,)
