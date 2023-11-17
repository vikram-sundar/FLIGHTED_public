"""Script to train a FLIGHTED-selection model."""
import itertools
import logging
import os

import numpy as np
import pyro
import torch

from src.common_utils import PROTEIN_ALPHABET, to_one_hot
from src.flighted_inference import flighted_models, flighted_trainers

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format="%(message)s", level=logging.INFO)

output_path = "Data/FLIGHTED_Selection/"
data_path = "Data/Selection_Simulations/"
selection_data = torch.load(os.path.join(data_path, "selection_data.pt"))
p_sel = np.load(os.path.join(data_path, "p_sel.npy"))
sequences = ["".join(val) for val in itertools.product(PROTEIN_ALPHABET, repeat=4)]


def to_one_hot_sequence(sequence):
    return to_one_hot(sequence, alphabet=PROTEIN_ALPHABET).transpose(0, 1)


hparams = {
    "num_amino_acids": 20,
    "seq_length": 4,
    "predict_variance": True,
    "learning_rate": 1e-2,
    "landscape_model_learning_rate": 1e-1,
    "batch_size": 10,
    "num_epochs": 150,
    "test_batch_size": 1,
    "num_elbo_particles": 10,
    "include_N_sample": False,
    "total_population": 1e11,
    "enr_scale": 100,
    "lr_scheduler": True,
    "plateau_patience": 4,
}


flighted_trainers.train_selection_model(
    flighted_models.FLIGHTED_Selection,
    sequences,
    selection_data,
    to_one_hot_sequence,
    output_path,
    input_hparams=hparams,
)
np.save(os.path.join(output_path, "p_sel.npy"), p_sel)
np.save(os.path.join(output_path, "selection_data.npy"), selection_data)
