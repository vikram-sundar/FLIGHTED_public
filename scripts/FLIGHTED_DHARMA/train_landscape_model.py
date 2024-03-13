"""Script to model the TEV landscape with FLIGHTED."""
import argparse
import functools
import json
import os

import numpy as np
import pandas as pd

from src.featurization import embeddings
from src.landscape_inference import landscape_models, landscape_trainers

# pylint: disable=invalid-name

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    choices=["one_vs_rest", "two_vs_rest", "three_vs_rest"],
    help="Split to use",
    required=True,
)
parser.add_argument("--output", type=str, help="Output directory", required=True)
parser.add_argument("--hparams", type=str, help="File with hparams for this run", required=True)
parser.add_argument(
    "--model_type",
    type=str,
    choices=["linear", "cnn", "fnn", "esmcnn", "esmfnn"],
    help="Type of model to use",
    required=True,
)
parser.add_argument(
    "--embedding",
    type=str,
    choices=[
        "one_hot",
        "georgiev",
        "tape",
        "esm1b",
        "esm1v",
        "esm2_8m",
        "esm2_35m",
        "esm2_150m",
        "esm2_650m",
        "esm2",
        "prottrans",
        "carp_600k",
        "carp_38M",
        "carp_76M",
        "carp",
        "tape_all",
        "esm1b_all",
        "esm1v_all",
        "esm2_8m_all",
        "esm2_35m_all",
        "esm2_150m_all",
        "esm2_650m_all",
        "esm2_all",
        "prottrans_all",
        "carp_600k_all",
        "carp_38M_all",
        "carp_76M_all",
        "carp_all",
        "augmented_esm1v",
        "augmented_esm2_8m",
        "augmented_esm2_35m",
        "augmented_esm2_150m",
        "augmented_esm2_650m",
        "augmented_esm2",
        "augmented_carp_600k",
        "augmented_carp_38M",
        "augmented_carp_76M",
        "augmented_carp",
        "augmented_evcoupling",
        "esm2_8m_finetune",
        "esm2_35m_finetune",
        "esm2_150m_finetune",
        "esm2_650m_finetune",
        "esm2_finetune",
    ],
    help="Type of embedding to use",
    required=True,
)
args = parser.parse_args()

MODEL_CLASSES = {
    "linear": landscape_models.LinearRegressionLandscapeModel,
    "cnn": landscape_models.CNNLandscapeModel,
    "fnn": landscape_models.FNNLandscapeModel,
    "esmfnn": landscape_models.ESMFinetuneLandscapeModel,
    "esmcnn": landscape_models.ESMCNNLandscapeModel,
}
WT_SEQUENCE = "MFKGPRDYNPISSTICHLTNESDGHTTSLYGIGFGPFIITNKHLFRRNNGTLLVQSLHGVFKVKNTTTLQQHLIDGRDMIIIRMPKDFPPFPQKLKFREPQREERICLVTTNFQTKSMSSMVSDTSCTFPSSDGIFWKHWIQTKDGQCGSPLVSTRDGFIVGIHSASNFTNTNNYFTSVPKNFMELLTNQEAQQWVSGWRLNADSVLWGGHKVFMVKPEEPFQPVKEATQLMN"
EMBEDDINGS = {
    "one_hot": embeddings.one_hot_embedding,
    "tape": functools.partial(
        embeddings.tape_embedding, model="transformer", work_dir=args.output, aggregation="mean"
    ),
    "esm1b": functools.partial(
        embeddings.esm_embedding, model="esm-1b", work_dir=args.output, aggregation="mean"
    ),
    "esm1v": functools.partial(
        embeddings.esm_embedding, model="esm-1v-1", work_dir=args.output, aggregation="mean"
    ),
    "esm2_8m": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-8m",
        work_dir=args.output,
        aggregation="mean",
        batch_size=512,
    ),
    "esm2_35m": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-35m",
        work_dir=args.output,
        aggregation="mean",
        batch_size=512,
    ),
    "esm2_150m": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-150m",
        work_dir=args.output,
        aggregation="mean",
        batch_size=512,
    ),
    "esm2_650m": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-650m",
        work_dir=args.output,
        aggregation="mean",
        batch_size=512,
    ),
    "esm2": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-3b",
        work_dir=args.output,
        aggregation="mean",
        batch_size=16,
    ),
    "prottrans": functools.partial(
        embeddings.prottrans_embedding, model="prott5-xl-u50", aggregation="mean", batch_size=16
    ),
    "carp_600k": functools.partial(
        embeddings.carp_embedding,
        model="600k",
        aggregation="mean",
        work_dir=args.output,
        batch_size=2,
    ),
    "carp_38M": functools.partial(
        embeddings.carp_embedding,
        model="38M",
        aggregation="mean",
        work_dir=args.output,
        batch_size=2,
    ),
    "carp_76M": functools.partial(
        embeddings.carp_embedding,
        model="76M",
        aggregation="mean",
        work_dir=args.output,
        batch_size=2,
    ),
    "carp": functools.partial(
        embeddings.carp_embedding,
        model="640M",
        aggregation="mean",
        work_dir=args.output,
        batch_size=2,
    ),
    "tape_all": functools.partial(
        embeddings.tape_embedding, model="transformer", work_dir=args.output
    ),
    "esm1b_all": functools.partial(embeddings.esm_embedding, model="esm-1b", work_dir=args.output),
    "esm1v_all": functools.partial(
        embeddings.esm_embedding, model="esm-1v-1", work_dir=args.output
    ),
    "esm2_8m_all": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-8m",
        work_dir=args.output,
        batch_size=512,
    ),
    "esm2_35m_all": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-35m",
        work_dir=args.output,
        batch_size=512,
    ),
    "esm2_150m_all": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-150m",
        work_dir=args.output,
        batch_size=512,
    ),
    "esm2_650m_all": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-650m",
        work_dir=args.output,
        batch_size=16,
    ),
    "esm2_all": functools.partial(
        embeddings.esm_embedding,
        model="esm-2-3b",
        work_dir=args.output,
        batch_size=16,
    ),
    "prottrans_all": functools.partial(
        embeddings.prottrans_embedding, model="prott5-xl-u50", batch_size=16
    ),
    "carp_600k_all": functools.partial(
        embeddings.carp_embedding, model="600k", work_dir=args.output, batch_size=2
    ),
    "carp_38M_all": functools.partial(
        embeddings.carp_embedding, model="38M", work_dir=args.output, batch_size=2
    ),
    "carp_76M_all": functools.partial(
        embeddings.carp_embedding, model="76M", work_dir=args.output, batch_size=2
    ),
    "carp_all": functools.partial(
        embeddings.carp_embedding, model="640M", work_dir=args.output, batch_size=2
    ),
    "augmented_esm1v": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.esm_variant_embedding, wt_sequence=WT_SEQUENCE, model="esm-1v-1"
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_esm2_8m": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.esm_variant_embedding, wt_sequence=WT_SEQUENCE, model="esm-2-8m"
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_esm2_35m": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.esm_variant_embedding, wt_sequence=WT_SEQUENCE, model="esm-2-35m"
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_esm2_150m": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.esm_variant_embedding, wt_sequence=WT_SEQUENCE, model="esm-2-150m"
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_esm2_650m": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.esm_variant_embedding, wt_sequence=WT_SEQUENCE, model="esm-2-650m"
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_esm2": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.esm_variant_embedding, wt_sequence=WT_SEQUENCE, model="esm-2-3b"
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_carp_600k": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.carp_variant_embedding,
                model="600k",
                work_dir=args.output,
                batch_size=8,
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_carp_38M": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.carp_variant_embedding,
                model="38M",
                work_dir=args.output,
                batch_size=8,
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_carp_76M": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.carp_variant_embedding,
                model="76M",
                work_dir=args.output,
                batch_size=8,
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "augmented_carp": functools.partial(
        embeddings.concat_embeddings,
        embedding_funcs=[
            functools.partial(
                embeddings.carp_variant_embedding,
                model="640M",
                work_dir=args.output,
                batch_size=8,
            ),
            functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        ],
    ),
    "esm2_8m_finetune": functools.partial(embeddings.esm_finetune_embedding, model="esm-2-8m"),
    "esm2_35m_finetune": functools.partial(embeddings.esm_finetune_embedding, model="esm-2-35m"),
    "esm2_150m_finetune": functools.partial(embeddings.esm_finetune_embedding, model="esm-2-150m"),
    "esm2_650m_finetune": functools.partial(embeddings.esm_finetune_embedding, model="esm-2-650m"),
    "esm2_finetune": functools.partial(embeddings.esm_finetune_embedding, model="esm-2-3b"),
}

model_class = MODEL_CLASSES[args.model_type]
embedding_func = EMBEDDINGS[args.embedding]


with open(os.path.join("hparam_files", args.hparams), "r") as f:
    hparams = json.load(f)

input_dir = "Data/TEV_Landscape/Splits/"
output_dir = "/Data/TEV_Models/"
train_data = pd.read_csv(os.path.join(input_dir, args.split + "_train.csv"))
val_data = pd.read_csv(os.path.join(input_dir, args.split + "_val.csv"))
test_data = pd.read_csv(os.path.join(input_dir, args.split + "_test.csv"))
train_sequences = embedding_func(list(train_data["Complete Sequence"].values))
val_sequences = embedding_func(list(val_data["Complete Sequence"].values))
test_sequences = list(test_data["Complete Sequence"].values)

if args.model_type == "linear" and args.embedding == "one_hot":
    hparams["num_features"] = train_sequences.shape[1] * train_sequences.shape[2]
if args.model_type == "linear" and args.embedding.startswith("augmented"):
    hparams["num_features"] = train_sequences.shape[1]

hparams["test_batch_size"] = 1000

train_fitnesses = train_data["Fitness Mean"].values
val_fitnesses = val_data["Fitness Mean"].values
test_fitnesses = test_data["Fitness Mean"].values
train_variances = train_data["Fitness Variance"].values
val_variances = val_data["Fitness Variance"].values
test_variances = test_data["Fitness Variance"].values

landscape_trainers.train_landscape_model(
    model_class,
    train_sequences,
    train_fitnesses,
    val_sequences,
    val_fitnesses,
    test_sequences,
    test_fitnesses,
    embedding_func,
    os.path.join(output_dir, args.output),
    train_variances,
    val_variances,
    test_variances,
    hparams,
)

np.save(
    os.path.join(output_dir, args.output, "test_sequences.npy"),
    np.array(list(test_data["Complete Sequence"].values)),
)
