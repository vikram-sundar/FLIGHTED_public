"""Methods to train landscape models."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

from flighted import common_utils
from flighted.landscape_inference import landscape_datasets, landscape_models

DEFAULT_LANDSCAPE_HPARAMS = {
    "batch_size": 100,  # batch size for training
    "test_batch_size": 10,  # batch size for testing
    "num_epochs": 500,  # number of epochs to train for
    "learning_rate": 1e-2,  # learning rate
    "weight_decay": 0,  # weight decay for Adam (L2 regularization)
    "cnn_learning_rate": 1e-3,  # for CNN only, learning rate
    "cnn_weight_decay": 0,  # for CNN only, weight decay
    "embedding_learning_rate": 5e-5,  # for CNN only, embedding learning rate
    "embedding_weight_decay": 0.05,  # for CNN only, embedding weight decay
    "linear_learning_rate": 5e-6,  # for CNN only, output learning rate
    "linear_weight_decay": 0.05,  # for CNN only, output weight decay
    "lr_scheduler": False,  # whether to use a plateau scheduler
    "plateau_patience": 10,  # number of epochs to wait for plateau scheduler
    "plateau_factor": 0.1,  # factor to multiply learning rate by on plateau
    "plateau_threshold": 1e-4,  # threshold for minimum on plateau
}

# pylint: disable=no-member, too-many-statements


def train_landscape_model(
    model_class,
    train_sequences,
    train_fitnesses,
    val_sequences,
    val_fitnesses,
    test_sequences,
    test_fitnesses,
    output_dir,
    train_variances=None,
    val_variances=None,
    test_variances=None,
    input_hparams=None,
    cpu_only=False,
):
    """Trains a landscape model with the given fitnesses and negative data information.

    Args:
        model_class: nn.Module subclass to initialize with the given hparams and train.
        train_sequences: torch.Tensor with featurization of sequences in the training set.
        train_fitnesses: torch.Tensor with fitnesses in the training set.
        val_sequences: torch.Tensor with featurization of sequences in the validation set.
        val_fitnesses: torch.Tensor with fitnesses in the validation set.
        test_sequences: torch.Tensor with featurization of sequences in the test set.
        test_fitnesses: torch.Tensor with fitnesses in the test set.
        output_dir: String, directory to output model.
        train_variances: torch.Tensor with variances of the training set.
        val_variances: torch.Tensor with variances of the validation set.
        test_variances: torch.Tensor with variances of the test set.
        input_hparams: Dict with hyperparameters for model.
        cpu_only: boolean, whether to use cpu only for this model
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "hparams.json"), "w") as f:
        json.dump(input_hparams, f)

    hparams = common_utils.overwrite_hparams(input_hparams, DEFAULT_LANDSCAPE_HPARAMS)

    model = model_class(hparams)
    device = (
        torch.device("cuda:0")
        if not cpu_only and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    loss_fn = landscape_models.MSELoss()
    loss_fn.to(device)

    # dummy variance
    if train_variances is None:
        train_variances = torch.Tensor([1 for _ in train_fitnesses])
    if val_variances is None:
        val_variances = torch.Tensor([1 for _ in val_fitnesses])
    if test_variances is None:
        test_variances = torch.Tensor([1 for _ in test_fitnesses])

    train_dataset = landscape_datasets.FitnessDataset(
        train_sequences, train_fitnesses, train_variances
    )
    val_dataset = landscape_datasets.FitnessDataset(val_sequences, val_fitnesses, val_variances)
    test_dataset = landscape_datasets.FitnessDataset(test_sequences, test_fitnesses, test_variances)

    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams["test_batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams["test_batch_size"], shuffle=False)

    if isinstance(model, landscape_models.CNNLandscapeModel):
        optimizer = optim.Adam(
            [
                {
                    "params": model.cnn.parameters(),
                    "lr": hparams["cnn_learning_rate"],
                    "weight_decay": hparams["cnn_weight_decay"],
                },
                {
                    "params": model.embedding_nn.parameters(),
                    "lr": hparams["embedding_learning_rate"],
                    "weight_decay": hparams["embedding_weight_decay"],
                },
                {
                    "params": model.output_nn.parameters(),
                    "lr": hparams["linear_learning_rate"],
                    "weight_decay": hparams["linear_weight_decay"],
                },
            ]
        )
    else:
        optimizer = common_utils.load_common_optimizers(model, hparams)
    if hparams["lr_scheduler"]:
        scheduler = common_utils.load_lr_scheduler(optimizer, hparams)
    else:
        scheduler = None

    min_val_loss = np.inf
    best_epoch = -1

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    for epoch in trange(hparams["num_epochs"]):
        history["epoch"] += [epoch]

        train_loss = 0

        for x, y, var in train_dataloader:
            optimizer.zero_grad()
            x, y, var = x.to(device), y.to(device), var.to(device)
            y_hat = model(x)
            loss = loss_fn(y, y_hat, var)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()[()]
        train_loss /= len(train_dataloader)
        history["train_loss"] += [train_loss]

        with torch.no_grad():
            val_loss = 0
            for x, y, var in val_dataloader:
                optimizer.zero_grad()
                x, y, var = x.to(device), y.to(device), var.to(device)
                y_hat = model(x)
                val_loss += loss_fn(y, y_hat, var)
            val_loss /= len(val_dataloader)
            val_loss = val_loss.detach().cpu().numpy()[()]
            history["val_loss"] += [val_loss]

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
        if scheduler:
            scheduler.step(val_loss)

        common_utils.store_model(output_dir, model, optimizer, hparams, epoch, scheduler=scheduler)

    # load and store best model
    if hparams["lr_scheduler"]:
        model, optimizer, scheduler = common_utils.load_model_from_epoch(
            output_dir, best_epoch, model, optimizer, scheduler=scheduler
        )
    else:
        model, optimizer, _ = common_utils.load_model_from_epoch(
            output_dir, best_epoch, model, optimizer
        )
    model.eval()

    test_preds = []
    test_trues = []
    test_vars = []
    with torch.no_grad():
        test_loss = 0
        for x, y, var in test_dataloader:
            x, y, var = x.to(device), y.to(device), var.to(device)
            y_hat = model(x)
            test_preds += [y_hat]
            test_trues += [y]
            test_vars += [var]
            test_loss += loss_fn(y, y_hat, var)
        test_loss /= len(test_dataloader)
        test_loss = test_loss.detach().cpu().numpy()[()]

    np.save(os.path.join(output_dir, "test_loss.npy"), np.array([test_loss]))

    common_utils.store_model(
        output_dir, model, optimizer, hparams, best_epoch, scheduler=scheduler, best=True
    )
    common_utils.clean_up_checkpoints(output_dir)
    history = pd.DataFrame(history)
    history.to_pickle(os.path.join(output_dir, "history.pkl"))

    test_preds = torch.cat(test_preds)
    np.save(os.path.join(output_dir, "test_preds.npy"), test_preds.detach().cpu().numpy())
    test_trues = torch.cat(test_trues)
    np.save(os.path.join(output_dir, "test_trues.npy"), test_trues.detach().cpu().numpy())
    test_vars = torch.cat(test_vars)
    np.save(os.path.join(output_dir, "test_vars.npy"), test_vars.detach().cpu().numpy())
