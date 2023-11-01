"""Tests for flighted_inference/flighted_datasets.py."""
from pathlib import Path

import pandas as pd
import torch

from src.flighted_inference import flighted_datasets

PATH = Path(__file__).parent

# pylint: disable=missing-function-docstring, no-member, unbalanced-tuple-unpacking


def test_dharma_dataset():
    dharma_dict = {
        "variant_seq": ["AA", "AA", "BB", "CC"],
        "featurized_seq_0": [0, 0, 1, 0],
        "featurized_seq_1": [0, 1, 0, 1],
    }
    dharma_df = pd.DataFrame(dharma_dict)
    hparams = {
        "variant_column": "variant_seq",
        "dharma_readout_prefix": "featurized_seq",
        "num_residues": 2,
        "oversampling": 2,
        "combine_reads": True,
    }

    def sequence_featurization(sequence):
        sequence_dict = {"AA": [0, 0], "BB": [1, 1], "CC": [1, 0]}
        return torch.tensor(sequence_dict[sequence], dtype=torch.float32)

    unsupervised_dataset = flighted_datasets.DHARMADataset(
        dharma_df, sequence_featurization, False, hparams=hparams
    )
    assert len(unsupervised_dataset) == 3
    assert len(unsupervised_dataset[0]) == 2
    assert unsupervised_dataset[0][0].shape == (2, 2)
    assert unsupervised_dataset[0][1].shape == (2, 2, 3)
    assert unsupervised_dataset[1][0].shape == (1, 2)
    assert unsupervised_dataset[1][1].shape == (1, 2, 3)


def test_collate_dharma():
    seq_batch = [torch.Tensor([[1, 0, 0], [0, 1, 0]]), torch.Tensor([[1, 0, 0]])]
    dharma_batch = [torch.Tensor([[1, 0], [0, 1]]), torch.Tensor([[1, 0]])]

    batch = list(zip(seq_batch, dharma_batch))
    seq_collated, dharma_collated = flighted_datasets.collate_dharma(batch)
    assert seq_collated.shape == (3, 3)
    assert dharma_collated.shape == (3, 2)


def test_selection_dataset():
    experiments = torch.Tensor(
        [[[1, 2, 0], [2, 1, 0], [3, 1, 1]], [[3, 2, 1], [4, 1, 5], [5, 1, 2]]]
    )
    selection_dataset = flighted_datasets.SelectionDataset(experiments)
    assert len(selection_dataset) == 2
    assert selection_dataset[0].shape == (3, 3)
