"""Common utility methods for multiple classes."""
import copy
import glob
import os

import numpy as np
import pyro
import torch

# pylint: disable=no-member

RNA_ALPHABET = "ACGU"
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PROTEIN_ALPHABET_EXTENDED = "ACDEFGHIKLMNPQRSTVWXY"
PROTEIN_ALPHABET_DICT = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}
DHARMA_HPARAMS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/DHARMA_Models/hparams.json"
DHARMA_WEIGHTS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/DHARMA_Models/best_model.ckpt"
SELECTION_HPARAMS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/FLIGHTED_Selection/hparams.json"
SELECTION_WEIGHTS_URL = "https://raw.githubusercontent.com/vikram-sundar/FLIGHTED_public/main/Data/FLIGHTED_Selection/best_model.ckpt"


def to_one_hot(sequence, alphabet):
    """Converts a sequence to a one-hot representation using a given alphabet.

    Args:
        sequence: string representing the sequences.
        alphabet: string representing the given alphabet.

    Returns:
        torch.tensor with the given one-hot representation (alphabet length x sequence length).
    """
    indices = np.array([alphabet.index(char) for char in sequence.upper()])
    one_hot = np.eye(len(alphabet))[np.array(indices).reshape(-1)]
    return torch.tensor(
        one_hot.reshape(list(indices.shape) + [len(alphabet)]).T, dtype=torch.float32
    )


def load_common_optimizers(model, hparams):
    """Loads an Adam optimizer for the given model and hyperparameters."""
    if "weight_decay" not in hparams:
        hparams["weight_decay"] = 0
    return torch.optim.Adam(
        model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"]
    )


def load_lr_scheduler(optimizer, hparams):
    """Loads a learning rate scheduler with a plateau for given optimizer and hyperparameters."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=hparams["plateau_factor"],
        patience=hparams["plateau_patience"],
        threshold=hparams["plateau_threshold"],
    )


def store_model(output_dir, model, optimizer, hparams, epoch, scheduler=None, best=False):
    """Checkpoints and stores a model based on the given parameters.

    Args:
        output_dir: str with model output directory.
        model: nn.Module subclass whose parameters need to be stored.
        optimizer: torch or pyro optimizer whose parameters need to be stored.
        hparams: Dict containing hyperparameters for training.
        epoch: int, epoch number.
        scheduler: torch or pyro learning rate scheduler whose parameters need to be stored.
        best: boolean, whether this is the best model.
    """
    if best:
        output_file = os.path.join(output_dir, "best_model.ckpt")
    else:
        output_file = os.path.join(output_dir, f"checkpoint_epoch{epoch}.ckpt")
    if scheduler:
        if isinstance(scheduler, pyro.optim.lr_scheduler.PyroLRScheduler):
            scheduler_state_dict = scheduler.get_state()
            optimizer_state_dict = None
        else:
            scheduler_state_dict = scheduler.state_dict()
            optimizer_state_dict = optimizer.state_dict()
    else:
        if isinstance(optimizer, pyro.optim.optim.PyroOptim):
            optimizer_state_dict = optimizer.get_state()
        else:
            optimizer_state_dict = optimizer.state_dict()
    checkpoint_dir = {
        "state_dict": model.state_dict(),
        "optimizer_states": optimizer_state_dict,
        "epoch": epoch,
        "hyper_parameters": hparams,
    }
    if scheduler:
        checkpoint_dir["lr_schedulers"] = scheduler_state_dict
    torch.save(checkpoint_dir, output_file)


def load_model_from_epoch(output_dir, epoch, model, optimizer, scheduler=None):
    """Reloads a model from a given epoch (to save the best model).

    Args:
        output_dir: str with model output directory.
        model: nn.Module subclass whose parameters need to be restored.
        optimizer: torch or pyro optimizer whose parameters need to be restored.
        epoch: int, epoch number of the best epoch.
        scheduler: torch or pyro learning rate scheduler whose parameters need to be restored.

    Returns:
        model: nn.Module with restored parameters.
        optimizer: torch or pyro optimizer with restored parameters.
        scheduler: torch or pyro learning rate scheduler with restored parameters or None.
    """
    output_file = os.path.join(output_dir, f"checkpoint_epoch{epoch}.ckpt")
    checkpoint_dir = torch.load(output_file)
    model.load_state_dict(checkpoint_dir["state_dict"])
    if isinstance(optimizer, pyro.optim.optim.PyroOptim):
        optimizer.set_state(checkpoint_dir["optimizer_states"])
    else:
        if scheduler:
            if isinstance(scheduler, pyro.optim.lr_scheduler.PyroLRScheduler):
                scheduler.set_state(checkpoint_dir["lr_schedulers"])
            else:
                scheduler.load_state_dict(checkpoint_dir["lr_schedulers"])
                optimizer.load_state_dict(checkpoint_dir["optimizer_states"])
            return model, optimizer, scheduler
        optimizer.load_state_dict(checkpoint_dir["optimizer_states"])
    return model, optimizer, None


def clean_up_checkpoints(output_dir):
    """Remove unnecessary checkpoint files once model is done training.

    Args:
        output_dir: str with model output directory.
    """
    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint_epoch*.ckpt"))
    for filepath in checkpoint_files:
        os.remove(filepath)


def overwrite_hparams(hparams, default_hparams):
    """Overwrite the default hparams dict with any custom hparams.

    Args:
        hparams: Dict, custom hparams to overwrite.
        default_hparams: Dict, default hparams to be overwritten.

    Returns:
        Dict with replaced hparams.
    """
    real_hparams = copy.deepcopy(default_hparams)
    if hparams:
        for parameter, value in hparams.items():
            real_hparams[parameter] = value
    return real_hparams
