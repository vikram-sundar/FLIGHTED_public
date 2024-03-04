"""Tests for landscape_inference/landscape_datasets.py."""
from pathlib import Path

import torch

from src.landscape_inference import landscape_datasets

PATH = Path(__file__).parent

# pylint: disable=missing-function-docstring, no-member


def test_fitness_dataset():
    sequences = torch.Tensor([[0, 0], [1, 1]])
    fitnesses = torch.Tensor([0, 2])
    variances = torch.Tensor([1, 2])

    dataset = landscape_datasets.FitnessDataset(sequences, fitnesses, variances)
    assert len(dataset) == 2
    assert len(dataset[0]) == 3
    assert len(dataset[0][0]) == 2
