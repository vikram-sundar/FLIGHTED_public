"""Script to infer fitnesses with variance from the GB1 model."""
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.flighted_inference import flighted_models

# pylint: disable=no-member, invalid-name

# read in sequence and remove nans
GB1_landscape = pd.read_csv("Data/Fitness_Landscapes/GB1_landscape.csv")
GB1_landscape = GB1_landscape.dropna()
GB1_selection_data = GB1_landscape[["Count input", "Count selected"]].to_numpy()
totals = GB1_selection_data.sum(axis=0)
GB1_selection_data = np.hstack(
    [
        np.arange(GB1_selection_data.shape[0])[:, np.newaxis],
        GB1_selection_data,
        np.tile(totals, (GB1_selection_data.shape[0], 1)),
    ]
)
GB1_selection_data = torch.tensor(GB1_selection_data, dtype=torch.int32)
GB1_selection_data = GB1_selection_data.unsqueeze(0)

# load model
model_dir = "Data/FLIGHTED_Selection"
with open(os.path.join(model_dir, "hparams.json"), "r") as f:
    hparams = json.load(f)
model = flighted_models.FLIGHTED_Selection(hparams)
best_model_dict = torch.load(
    os.path.join(model_dir, "best_model.ckpt"), map_location=torch.device("cpu")
)
model.load_state_dict(best_model_dict["state_dict"])

# run inference using model (note that we predict variance, not std dev)
fitness_mean, fitness_var = model.selection_reverse_model(GB1_selection_data)

GB1_landscape["Updated Fitness"] = fitness_mean.detach().numpy()[0]
GB1_landscape["Fitness Variance"] = F.softplus(fitness_var).detach().numpy()[0]

# get the full sequence for use in downstream modeling
sites_mutated = [39, 40, 41, 54]
full_sequence = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQGPLDPNSMATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH"


def compute_complete_sequence(row):
    """Compute complete sequence from four-site abbreviation."""
    sequence = copy.copy(full_sequence)
    sequence = list(sequence)
    for i, site in enumerate(sites_mutated):
        sequence[site - 1] = row["Variants"][i]
    sequence = "".join(sequence)
    return sequence


GB1_landscape["Complete Sequence"] = GB1_landscape.apply(compute_complete_sequence, axis=1)
GB1_landscape.to_csv("Data/Fitness_Landscapes/GB1_landscape_with_var.csv")
