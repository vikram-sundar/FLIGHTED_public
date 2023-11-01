"""Tests for landscape_inference/landscape_trainers.py."""
import functools
import os
import shutil
from pathlib import Path

import torch

from src.common_utils import RNA_ALPHABET, to_one_hot
from src.landscape_inference import landscape_models, landscape_trainers

PATH = Path(__file__).parent

# pylint: disable=missing-function-docstring, no-member


def test_train_landscape():
    model_path = f"{str(PATH/'test_landscape_model/')}"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    pred_fitness_genotypes = ["AA", "UU", "GA", "GC", "GU", "CG", "CU", "GG"]
    pred_fitnesses = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
    pred_variances = torch.Tensor([1, 2, 1, 2, 1, 2, 1, 2])
    sequence_lambda = functools.partial(to_one_hot, alphabet=RNA_ALPHABET)
    pred_sequences = torch.stack([sequence_lambda(sequence) for sequence in pred_fitness_genotypes])

    landscape_trainers.train_landscape_model(
        landscape_models.LinearRegressionLandscapeModel,
        pred_sequences[:4],
        pred_fitnesses[:4],
        pred_sequences[4:6],
        pred_fitnesses[4:6],
        pred_sequences[6:8],
        pred_fitnesses[6:8],
        model_path,
        train_variances=pred_variances[:4],
        val_variances=pred_variances[4:6],
        test_variances=pred_variances[6:8],
        input_hparams={
            "batch_size": 10,
            "test_batch_size": 2,
            "num_epochs": 10,
            "num_features": 8,
        },
        cpu_only=True,
    )

    landscape_trainers.train_landscape_model(
        landscape_models.CNNLandscapeModel,
        pred_sequences[:4],
        pred_fitnesses[:4],
        pred_sequences[4:6],
        pred_fitnesses[4:6],
        pred_sequences[6:8],
        pred_fitnesses[6:8],
        model_path,
        train_variances=pred_variances[:4],
        val_variances=pred_variances[4:6],
        test_variances=pred_variances[6:8],
        input_hparams={
            "batch_size": 10,
            "test_batch_size": 2,
            "num_epochs": 10,
            "num_features": 4,
        },
        cpu_only=True,
    )
