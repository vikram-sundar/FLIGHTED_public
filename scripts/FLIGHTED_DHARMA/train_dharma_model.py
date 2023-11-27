"""Script to train FLIGHTED_DHARMA model."""
import logging
import os

import pandas as pd
import pyro
from Bio import SeqIO

from src.common_utils import PROTEIN_ALPHABET_EXTENDED, to_one_hot
from src.flighted_inference import flighted_models, flighted_trainers

# pylint: disable=invalid-name

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format="%(message)s", level=logging.INFO)

data_folder = "Data/DHARMA_Input/"

with open(os.path.join(data_folder, "104363-canvas.fasta")) as f:
    wt_record = SeqIO.read(f, format="fasta")

cytosine_residues = []
for i, base in enumerate(wt_record.seq):
    if base == "C":
        cytosine_residues += [i]


def to_one_hot_sequence(sequence):
    """Converts sequence to one-hot representation."""
    return to_one_hot(sequence, alphabet=PROTEIN_ALPHABET_EXTENDED).transpose(0, 1)


fluorescence_df = pd.read_csv(os.path.join(data_folder, "220124gfpl-aa-rfu.csv"), sep=",")
fluorescence_df = fluorescence_df.rename({"AA_gfp": "variant_seq", "rfu": "fluorescence"}, axis=1)

dharma_df = pd.read_csv(os.path.join(data_folder, "822988_clean.csv"), sep="\t")

# Filter out sequences for which we have fluorescence data (to use for testing)
dharma_df = dharma_df[~dharma_df["variant_seq"].isin(fluorescence_df["variant_seq"])]

# Set optimal hparams
hparams = {
    "cytosine_residues": cytosine_residues,
    "num_residues": len(wt_record.seq),
    "supervised": False,
    "predict_variance": True,
    "test_split": 0.1,  # Must be > 0 and < 1
    "val_split": 0.1,  # Must be > 0 and < 1
    "learning_rate": 1e-4,
    "landscape_model_learning_rate": 1e-2,
    "batch_size": 2,
    "num_epochs": 25,
    "test_batch_size": 10,
    "num_elbo_particles": 1,
    "vectorize_particles": False,
    "dharma_fitness_num_layers": 2,
    "dharma_fitness_hidden_dim": [10],
    "lr_scheduler": True,
    "plateau_patience": 1,
    "scheduler_step": "train",
    "scheduler_change": "epoch",
    "combine_reads": False,
    "average_batch": False,
    "num_amino_acids": 21,
    "timepoint": "822988",
}

flighted_trainers.train_dharma_model(
    flighted_models.FLIGHTED_DHARMA,
    dharma_df,
    to_one_hot_sequence,
    "Data/DHARMA_Models",
    input_hparams=hparams,
)
