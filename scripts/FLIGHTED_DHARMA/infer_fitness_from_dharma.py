"""Script to infer fitnesses with a FLIGHTED_DHARMA model."""
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from flighted import common_utils, pretrained
from flighted.common_utils import PROTEIN_ALPHABET_EXTENDED
from flighted.flighted_inference import flighted_datasets, flighted_trainers

# pylint: disable=invalid-name, redefined-outer-name, no-member

data_folder = "Data/DHARMA_Input"
data_location = os.path.join(data_folder, "822988_clean.csv")
model_folder = "Data/DHARMA_Models/"


def to_one_hot_sequence(sequence):
    """Return one-hot representation of protein sequence."""
    return common_utils.to_one_hot(sequence, alphabet=PROTEIN_ALPHABET_EXTENDED).transpose(0, 1)


def load_ds():
    """Load DHARMA dataset and combine reads into one datapoint for inference."""
    with open(os.path.join(model_folder, "hparams.json"), "r") as f:
        hparams = json.load(f)
    dharma_df = pd.read_csv(data_location, sep="\t")

    hparams = common_utils.overwrite_hparams(hparams, flighted_trainers.DEFAULT_DHARMA_HPARAMS)

    hparams_ds = hparams.copy()
    hparams_ds["combine_reads"] = True
    dharma_ds = flighted_datasets.DHARMADataset(dharma_df, to_one_hot_sequence, False, hparams_ds)
    return dharma_ds


def load_model():
    """Load pretrained best model."""
    _, model = pretrained.load_trained_flighted_model("DHARMA", cpu_only=True)
    return model


def make_predictions(model, dharma_ds):
    """Make predictions using model."""
    pred_means = []
    pred_variances = []
    num_reads = []
    seqs = []

    for seq, data in tqdm(zip(dharma_ds.sequences, dharma_ds)):
        mean, variance = model.infer_fitness_from_dharma(data[1])
        pred_means += [mean.detach().numpy()[()]]
        pred_variances += [variance.detach().numpy()[()]]
        num_reads += [data[1].shape[0]]
        seqs += [seq]

    return pred_means, pred_variances, seqs, num_reads


dharma_ds = load_ds()
model = load_model()
pred_means, pred_variances, seqs, num_reads = make_predictions(model, dharma_ds)

np.save(os.path.join(data_folder, "pred_means.npy"), pred_means)
np.save(os.path.join(data_folder, "pred_variances.npy"), pred_variances)
np.save(os.path.join(data_folder, "seqs.npy"), seqs)
np.save(os.path.join(data_folder, "num_reads.npy"), num_reads)
