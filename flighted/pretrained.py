"""Import already-trained models for FLIGHTED."""
import json
import os
from urllib.request import urlopen, urlretrieve

import torch

from flighted.flighted_inference import flighted_models

DHARMA_HPARAMS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/DHARMA_Models/hparams.json"
DHARMA_WEIGHTS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/DHARMA_Models/best_model.ckpt"
SELECTION_HPARAMS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/FLIGHTED_Selection/hparams.json"
SELECTION_WEIGHTS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/FLIGHTED_Selection/best_model.ckpt"

# pylint: disable=consider-using-with


def load_trained_flighted_model(model_type, cpu_only=True):
    """Loads a trained FLIGHTED model for either selection or DHARMA.

    Args:
        model_type: string, whether to load selection or DHARMA.
        cpu_only: boolean, whether to load on CPU.

    Returns:
        hparams: Dict, hyperparameters for loaded model.
        model: nn.Module, loaded model.
    """
    if model_type == "Selection":
        hparams_url = SELECTION_HPARAMS_URL
        model_ckpt_url = SELECTION_WEIGHTS_URL
    elif model_type == "DHARMA":
        hparams_url = DHARMA_HPARAMS_URL
        model_ckpt_url = DHARMA_WEIGHTS_URL
    else:
        raise Exception(
            f"Model type {model_type} is not supported. Options are Selection or DHARMA."
        )
    response = urlopen(hparams_url)
    hparams = json.loads(response.read())

    if model_type == "Selection":
        model = flighted_models.FLIGHTED_Selection(hparams)
    else:
        model = flighted_models.FLIGHTED_DHARMA(hparams)

    path, _ = urlretrieve(model_ckpt_url)
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available() and not cpu_only
        else torch.device("cpu")
    )
    best_model_dict = torch.load(path, map_location=device)
    model = model.to(device)
    model.load_state_dict(best_model_dict["state_dict"])
    os.remove(path)
    return hparams, model
