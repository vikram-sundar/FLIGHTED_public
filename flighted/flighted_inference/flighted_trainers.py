"""Methods to train the FLIGHTED models."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pyro
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange

from flighted import common_utils
from flighted.flighted_inference import flighted_datasets

DEFAULT_DHARMA_HPARAMS = {
    "variant_column": "variant_seq",  # variant sequence column name
    "dharma_readout_prefix": "featurized_seq",  # DHARMA readout prefix column
    "num_residues": 190,  # number of residues in canvas
    "batch_size": 10,  # batch size for training
    "test_batch_size": 10,  # batch size for testing
    "combine_reads": False,  # whether to combine multiple DHARMA reads for the same sequence
    "average_batch": False,  # whether to average DHARMA reads for the same sequence in a given batch
    "num_epochs": 10,  # number of epochs to train for
    "test_split": 0.1,  # proportion of data to put into test set
    "val_split": 0.1,  # proportion of data to put into validation set
    "oversampling": 1,  # how much to oversample supervised datapoints
    "learning_rate": 1e-2,  # learning rate
    "landscape_model_learning_rate": 1e-2,  # learning rate for landscape model
    "num_elbo_particles": 1,  # number of particles to use in ELBO
    "vectorize_particles": False,  # whether to vectorize the ELBO computation over `num_particles`
    "grad_clip_norm": 10,  # gradient clipping
    "lr_scheduler": False,  # whether to use a plateau scheduler
    "plateau_patience": 10,  # number of epochs to wait for plateau scheduler
    "plateau_factor": 0.1,  # factor to multiply learning rate by on plateau
    "plateau_threshold": 1e-4,  # threshold for minimum on plateau
    "scheduler_step": "train",  # whether to use train or val loss for scheduler step
    "scheduler_change": "epoch",  # whether to step the scheduler every epoch or batch
}

DEFAULT_SELECTION_HPARAMS = {
    "batch_size": 10,  # batch size for training
    "test_batch_size": 10,  # batch size for testing
    "num_epochs": 10,  # number of epochs to train for
    "test_split": 0.1,  # proportion of data to put into test set
    "val_split": 0.1,  # proportion of data to put into validation set
    "learning_rate": 1e-2,  # learning rate
    "landscape_model_learning_rate": 1e-2,  # learning rate for landscape model
    "num_elbo_particles": 1,  # number of particles to use in ELBO
    "grad_clip_norm": 10,  # gradient clipping
    "lr_scheduler": False,  # whether to use a plateau scheduler
    "plateau_patience": 10,  # number of epochs to wait for plateau scheduler
    "plateau_factor": 0.1,  # factor to multiply learning rate by on plateau
    "plateau_threshold": 1e-4,  # threshold for minimum on plateau
}

# pylint: disable=no-member, too-many-statements


def train_dharma_model(
    model_class,
    dharma_df,
    sequence_lambda,
    output_dir,
    input_hparams=None,
    cpu_only=False,
):
    """Trains a DHARMA model with the given fluorescences and DHARMA readouts.
    Args:
        model_class: nn.Module subclass to initialize with the given hparams and train.
        dharma_df: pd.DataFrame holding DHARMA readout information
        sequence_lambda: function to apply to sequences to featurize them.
        output_dir: String, directory to output model.
        input_hparams: Dict with hyperparameters for model.
        cpu_only: boolean, whether to use cpu only for this model
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "hparams.json"), "w") as f:
        json.dump(input_hparams, f)

    pyro.clear_param_store()
    hparams = common_utils.overwrite_hparams(input_hparams, DEFAULT_DHARMA_HPARAMS)

    device = (
        torch.device("cuda:0")
        if not cpu_only and torch.cuda.is_available()
        else torch.device("cpu")
    )
    vae = model_class(hparams)
    vae.to(device)

    sequences_unlabelled = list(dharma_df[hparams["variant_column"]].values)
    trainval_sequences_unlabelled, test_sequences_unlabelled = train_test_split(
        sequences_unlabelled, test_size=hparams["test_split"]
    )
    train_sequences_unlabelled, val_sequences_unlabelled = train_test_split(
        trainval_sequences_unlabelled, test_size=hparams["val_split"] / (1 - hparams["test_split"])
    )
    train_sequences = list(train_sequences_unlabelled)
    val_sequences = list(val_sequences_unlabelled)
    test_sequences = list(test_sequences_unlabelled)

    dharma_df_train = dharma_df[
        [val in train_sequences for val in dharma_df[hparams["variant_column"]].values]
    ]
    dharma_df_val = dharma_df[
        [val in val_sequences for val in dharma_df[hparams["variant_column"]].values]
    ]
    dharma_df_test = dharma_df[
        [val in test_sequences for val in dharma_df[hparams["variant_column"]].values]
    ]

    dharma_df_train.to_csv(os.path.join(output_dir, "train_set.csv.gz"), index=None)
    dharma_df_val.to_csv(os.path.join(output_dir, "val_set.csv.gz"), index=None)
    dharma_df_test.to_csv(os.path.join(output_dir, "test_set.csv.gz"), index=None)

    if hparams["combine_reads"]:
        collate_fn = flighted_datasets.collate_dharma
    else:
        collate_fn = None

    train_ds_unsup = flighted_datasets.DHARMADataset(
        dharma_df_train, sequence_lambda, False, hparams
    )
    val_ds_unsup = flighted_datasets.DHARMADataset(dharma_df_val, sequence_lambda, False, hparams)
    test_ds_unsup = flighted_datasets.DHARMADataset(dharma_df_test, sequence_lambda, False, hparams)
    train_dl_unsup = DataLoader(
        train_ds_unsup, batch_size=hparams["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_dl_unsup = DataLoader(
        val_ds_unsup, batch_size=hparams["test_batch_size"], shuffle=False, collate_fn=collate_fn
    )
    test_dl_unsup = DataLoader(
        test_ds_unsup, batch_size=hparams["test_batch_size"], shuffle=False, collate_fn=collate_fn
    )

    def param_learning_rate(param_name):
        if param_name.startswith("landscape_model."):
            return {"lr": hparams["landscape_model_learning_rate"]}
        return {"lr": hparams["learning_rate"]}

    elbo_main = pyro.infer.Trace_ELBO(num_particles=hparams["num_elbo_particles"])
    if hparams["lr_scheduler"]:
        optimizer = torch.optim.Adam
        scheduler = pyro.optim.ReduceLROnPlateau(
            {
                "optimizer": optimizer,
                "optim_args": param_learning_rate,
                "factor": hparams["plateau_factor"],
                "patience": hparams["plateau_patience"],
                "threshold": hparams["plateau_threshold"],
            }
        )
        svi_main = pyro.infer.SVI(vae.model, vae.guide, scheduler, elbo_main)
    else:
        optimizer = pyro.optim.Adam(param_learning_rate, {"clip_norm": hparams["grad_clip_norm"]})
        scheduler = None
        svi_main = pyro.infer.SVI(vae.model, vae.guide, optimizer, elbo_main)

    min_val_loss = np.inf
    best_epoch = -1
    history = {
        "epoch": [],
        "train_loss_unsup": [],
        "train_loss": [],
        "val_loss_unsup": [],
        "val_loss": [],
    }
    train_loss_by_batch = []
    batches_per_epoch = len(train_dl_unsup)
    val_batches_per_epoch = len(val_dl_unsup)
    test_batches_per_epoch = len(test_dl_unsup)
    for epoch in trange(hparams["num_epochs"]):
        history["epoch"] += [epoch]

        unsup_iter = iter(train_dl_unsup)
        loss_unsup = 0

        for _ in trange(batches_per_epoch):
            loss = 0

            try:
                sequences, dharmas = next(unsup_iter)
                sequences, dharmas = sequences.to(device), dharmas.to(device)

                loss = svi_main.step(sequences, dharmas)
                loss_unsup += loss
                del sequences, dharmas
            except StopIteration:
                pass

            if (
                scheduler is not None
                and hparams["scheduler_change"] == "batch"
                and hparams["scheduler_step"] == "train"
            ):
                scheduler.step(loss)
            train_loss_by_batch += [loss]

        loss_unsup /= len(train_ds_unsup)
        history["train_loss_unsup"] += [loss_unsup]
        history["train_loss"] += [loss_unsup]

        with torch.no_grad():
            unsup_iter = iter(val_dl_unsup)
            loss_unsup = 0

            for _ in range(val_batches_per_epoch):
                try:
                    sequences, dharmas = next(unsup_iter)
                    sequences, dharmas = sequences.to(device), dharmas.to(device)
                    loss = svi_main.evaluate_loss(sequences, dharmas)
                    loss_unsup += loss
                    del sequences, dharmas
                except StopIteration:
                    pass

            loss_unsup /= len(val_ds_unsup)
            history["val_loss_unsup"] += [loss_unsup]
            val_loss = loss_unsup
            history["val_loss"] += [val_loss]

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch

        if scheduler is not None and hparams["scheduler_change"] == "epoch":
            if hparams["scheduler_step"] == "train":
                scheduler.step(history["train_loss"][-1])
            elif hparams["scheduler_step"] == "val":
                scheduler.step(history["val_loss"][-1])
            else:
                raise Exception(
                    f"Scheduler step parameter {hparams['scheduler_step']} not supported."
                )
        print(f"Losses: {history['train_loss'][-1]}, {history['val_loss'][-1]}")

        common_utils.store_model(output_dir, vae, optimizer, hparams, epoch, scheduler=scheduler)

    if hparams["lr_scheduler"]:
        vae, optimizer, scheduler = common_utils.load_model_from_epoch(
            output_dir, best_epoch, vae, optimizer, scheduler=scheduler
        )
    else:
        vae, optimizer, _ = common_utils.load_model_from_epoch(
            output_dir, best_epoch, vae, optimizer
        )

    with torch.no_grad():
        unsup_iter = iter(test_dl_unsup)
        loss_unsup = 0

        for _ in range(test_batches_per_epoch):
            try:
                sequences, dharmas = next(unsup_iter)
                sequences, dharmas = sequences.to(device), dharmas.to(device)

                loss = svi_main.evaluate_loss(sequences, dharmas)
                loss_unsup += loss
                del sequences, dharmas
            except StopIteration:
                pass
        loss_unsup /= len(test_ds_unsup)
        test_loss = loss_unsup
        print(f"Test loss: {test_loss}")

    np.save(os.path.join(output_dir, "test_loss.npy"), np.array([test_loss]))

    common_utils.store_model(
        output_dir, vae, optimizer, hparams, best_epoch, best=True, scheduler=scheduler
    )
    common_utils.clean_up_checkpoints(output_dir)
    history = pd.DataFrame(history)
    history.to_pickle(os.path.join(output_dir, "history.pkl"))
    np.save(os.path.join(output_dir, "train_loss_by_batch.npy"), train_loss_by_batch)


def train_selection_model(
    model_class,
    sequences,
    selection_data,
    sequence_lambda,
    output_dir,
    input_hparams=None,
    cpu_only=False,
):
    """Trains a selection model with the given selection data.

    Args:
        model_class: nn.Module subclass to initialize with the given hparams and train.
        sequences: List of sequences for selection experiment.
        selection_data: torch.Tensor holding selection information for all experiments.
        sequence_lambda: function to apply to sequences to featurize them.
        output_dir: String, directory to output model.
        input_hparams: Dict with hyperparameters for model.
        cpu_only: boolean, whether to use cpu only for this model
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "hparams.json"), "w") as f:
        json.dump(input_hparams, f)

    pyro.clear_param_store()
    hparams = common_utils.overwrite_hparams(input_hparams, DEFAULT_SELECTION_HPARAMS)

    device = (
        torch.device("cuda:0")
        if not cpu_only and torch.cuda.is_available()
        else torch.device("cpu")
    )
    vae = model_class(hparams)
    vae.to(device)

    sequences_processed = torch.stack([sequence_lambda(sequence) for sequence in sequences]).to(
        device
    )

    trainval_experiments, test_experiments = train_test_split(
        selection_data, test_size=hparams["test_split"]
    )
    train_experiments, val_experiments = train_test_split(
        trainval_experiments, test_size=hparams["val_split"] / (1 - hparams["test_split"])
    )
    np.save(os.path.join(output_dir, "train_experiments.npy"), train_experiments)
    np.save(os.path.join(output_dir, "val_experiments.npy"), val_experiments)
    np.save(os.path.join(output_dir, "test_experiments.npy"), test_experiments)

    train_ds = flighted_datasets.SelectionDataset(train_experiments)
    val_ds = flighted_datasets.SelectionDataset(val_experiments)
    test_ds = flighted_datasets.SelectionDataset(test_experiments)
    train_dl = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=hparams["test_batch_size"], shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=hparams["test_batch_size"], shuffle=False)

    def param_learning_rate(param_name):
        if param_name.startswith("landscape_model."):
            return {"lr": hparams["landscape_model_learning_rate"]}
        return {"lr": hparams["learning_rate"]}

    elbo = pyro.infer.Trace_ELBO(num_particles=hparams["num_elbo_particles"])
    if hparams["lr_scheduler"]:
        optimizer = torch.optim.Adam
        scheduler = pyro.optim.ReduceLROnPlateau(
            {
                "optimizer": optimizer,
                "optim_args": param_learning_rate,
                "factor": hparams["plateau_factor"],
                "patience": hparams["plateau_patience"],
                "threshold": hparams["plateau_threshold"],
            }
        )
        svi = pyro.infer.SVI(vae.model, vae.guide, scheduler, elbo)
    else:
        optimizer = pyro.optim.Adam(param_learning_rate, {"clip_norm": hparams["grad_clip_norm"]})
        scheduler = None
        svi = pyro.infer.SVI(vae.model, vae.guide, optimizer, elbo)

    min_val_loss = np.inf
    best_epoch = -1
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }
    batches_per_epoch = len(train_dl)
    val_batches_per_epoch = len(val_dl)
    test_batches_per_epoch = len(test_dl)
    for epoch in trange(hparams["num_epochs"]):
        history["epoch"] += [epoch]

        train_iter = iter(train_dl)
        loss_all = 0

        for _ in trange(batches_per_epoch):
            try:
                experiments = next(train_iter)
                experiments = experiments.to(device)

                loss = svi.step(sequences_processed, experiments)
                loss_all += loss

            except StopIteration:
                pass

        loss_all /= len(train_ds)
        history["train_loss"] += [loss_all]

        with torch.no_grad():
            val_iter = iter(val_dl)
            loss_all = 0

            for _ in trange(val_batches_per_epoch):
                try:
                    experiments = next(val_iter)
                    experiments = experiments.to(device)

                    loss = svi.step(sequences_processed, experiments)
                    loss_all += loss
                except StopIteration:
                    pass

            loss_all /= len(val_ds)
            history["val_loss"] += [loss_all]

            if loss_all < min_val_loss:
                min_val_loss = loss_all
                best_epoch = epoch

        if scheduler is not None:
            scheduler.step(history["train_loss"][-1])
        print(f"Losses: {history['train_loss'][-1]}, {history['val_loss'][-1]}")

        common_utils.store_model(output_dir, vae, optimizer, hparams, epoch, scheduler=scheduler)

    if hparams["lr_scheduler"]:
        vae, optimizer, scheduler = common_utils.load_model_from_epoch(
            output_dir, best_epoch, vae, optimizer, scheduler=scheduler
        )
    else:
        vae, optimizer, _ = common_utils.load_model_from_epoch(
            output_dir, best_epoch, vae, optimizer
        )

    with torch.no_grad():
        test_iter = iter(test_dl)
        test_loss = 0

        for _ in trange(test_batches_per_epoch):
            try:
                experiments = next(test_iter)
                experiments = experiments.to(device)

                loss = svi.step(sequences_processed, experiments)
                test_loss += loss
            except StopIteration:
                pass

            test_loss /= len(test_ds)

    np.save(os.path.join(output_dir, "test_loss.npy"), np.array([test_loss]))

    common_utils.store_model(
        output_dir, vae, optimizer, hparams, best_epoch, best=True, scheduler=scheduler
    )
    common_utils.clean_up_checkpoints(output_dir)
    history = pd.DataFrame(history)
    history.to_pickle(os.path.join(output_dir, "history.pkl"))
