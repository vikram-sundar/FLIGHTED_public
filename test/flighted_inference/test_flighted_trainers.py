"""Tests for flighted_inference/flighted_trainers.py."""
from pathlib import Path

import pandas as pd
import torch

from flighted.common_utils import RNA_ALPHABET, to_one_hot
from flighted.flighted_inference import flighted_models, flighted_trainers

PATH = Path(__file__).parent

# pylint: disable=missing-function-docstring, no-member


def test_train_dharma():
    model_path = f"{str(PATH/'test_dharma_model/')}"

    dharma_dict = {
        "variant_seq": ["AA", "AC", "AG", "AU", "CA", "CC", "CG", "CU", "GA", "GC"],
        "featurized_seq_0": [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
        "featurized_seq_1": [1, 1, 1, 0, 0, 2, 2, 0, 0, 1],
        "featurized_seq_2": [2, 0, 0, 0, 0, 2, 2, 0, 0, 0],
    }
    dharma_df = pd.DataFrame(dharma_dict)

    def to_one_hot_sequence(sequence):
        return to_one_hot(sequence, alphabet=RNA_ALPHABET).transpose(0, 1)

    sequence_lambda = to_one_hot_sequence

    flighted_trainers.train_dharma_model(
        flighted_models.FLIGHTED_DHARMA,
        dharma_df,
        sequence_lambda,
        model_path,
        input_hparams={
            "num_residues": 3,
            "batch_size": 2,
            "test_batch_size": 1,
            "cytosine_residues": [0, 1],
            "dharma_fitness_num_layers": 1,
            "dharma_fitness_hidden_dim": [],
            "seq_length": 2,
            "num_amino_acids": 4,
            "predict_variance": True,
            "test_split": 0.2,
            "val_split": 0.2,
            "learning_rate": 1e-2,
            "aux_loss_multiplier": 0.1,
            "combine_reads": False,
        },
        cpu_only=True,
    )


def test_train_selection():
    model_path = f"{str(PATH/'test_selection_model/')}"

    sequences = ["AA", "AC", "AG", "AU", "CA", "CC", "CG", "CU", "GA", "GC"]
    selection_data = torch.Tensor(
        [
            [[0, 2, 1, 10, 10], [1, 2, 1, 10, 10], [2, 6, 8, 10, 10]],
            [[3, 4, 1, 10, 10], [4, 5, 2, 10, 10], [5, 1, 7, 10, 10]],
            [[6, 7, 2, 10, 10], [8, 2, 3, 10, 10], [5, 1, 5, 10, 10]],
            [[9, 2, 1, 10, 10], [0, 2, 1, 10, 10], [3, 6, 8, 10, 10]],
            [[0, 3, 2, 10, 10], [1, 3, 1, 10, 10], [2, 4, 7, 10, 10]],
        ]
    )

    def to_one_hot_sequence(sequence):
        return to_one_hot(sequence, alphabet=RNA_ALPHABET).transpose(0, 1)

    sequence_lambda = to_one_hot_sequence

    flighted_trainers.train_selection_model(
        flighted_models.FLIGHTED_Selection,
        sequences,
        selection_data,
        sequence_lambda,
        model_path,
        input_hparams={
            "num_residues": 3,
            "batch_size": 2,
            "test_batch_size": 1,
            "seq_length": 2,
            "num_amino_acids": 4,
            "predict_variance": True,
            "test_split": 0.2,
            "val_split": 0.2,
            "learning_rate": 1e-2,
            "total_population": 1000,
        },
        cpu_only=True,
    )
