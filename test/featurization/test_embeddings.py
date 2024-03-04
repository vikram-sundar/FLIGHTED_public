"""Tests for featurization/embeddings.py."""
import functools
from pathlib import Path

import pytest

from src.common_utils import PROTEIN_ALPHABET
from src.featurization import embeddings

PATH = Path(__file__).parent

# pylint: disable=no-member, missing-function-docstring


def test_one_hot_embedding():
    sequences = ["AA", "AC", "AG", "AU"]
    embedding = embeddings.one_hot_embedding(sequences, alphabet="rna")
    assert list(embedding.shape) == [4, 4, 2]
    assert embedding[0].numpy().tolist() == [[1, 1], [0, 0], [0, 0], [0, 0]]
    assert embedding[2].numpy().tolist() == [[1, 0], [0, 0], [0, 1], [0, 0]]

    embedding = embeddings.one_hot_embedding(sequences, alphabet="rna", flatten=True)
    assert list(embedding.shape) == [4, 8]


def test_tape_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding = embeddings.tape_embedding(
        sequences, "transformer", work_dir="temp_tape", cpu_only=True
    )
    assert list(embedding.shape) == [3, 768, 2]

    embedding = embeddings.tape_embedding(
        sequences, "transformer", work_dir="temp_tape_mean", aggregation="mean", cpu_only=True
    )
    assert list(embedding.shape) == [3, 768]

    embedding = embeddings.tape_embedding(
        sequences, "unirep", work_dir="temp_unirep", cpu_only=True
    )
    assert list(embedding.shape) == [3, 1900, 2]


def test_esm_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding = embeddings.esm_embedding(sequences, "esm-small", work_dir="temp_esm", cpu_only=True)
    assert list(embedding.shape) == [3, 768, 2]

    embedding = embeddings.esm_embedding(
        sequences, "esm-small", work_dir="temp_esm", aggregation="mean", cpu_only=True
    )
    assert list(embedding.shape) == [3, 768]


@pytest.mark.slow
def test_prottrans_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding = embeddings.prottrans_embedding(
        sequences, "prott5-xl-u50", batch_size=2, cpu_only=True
    )
    assert list(embedding.shape) == [3, 1024, 2]

    embedding = embeddings.prottrans_embedding(
        sequences, "prott5-xl-u50", aggregation="mean", batch_size=2, cpu_only=True
    )
    assert list(embedding.shape) == [3, 1024]


def test_carp_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding = embeddings.carp_embedding(sequences, "600k", work_dir="temp_carp", cpu_only=True)
    assert list(embedding.shape) == [3, 128, 2]

    embedding = embeddings.carp_embedding(
        sequences, "600k", work_dir="temp_carp", aggregation="mean", cpu_only=True
    )
    assert list(embedding.shape) == [3, 128]


def test_esm_variant_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding = embeddings.esm_variant_embedding(sequences, "AA", "esm-small", cpu_only=True)
    assert list(embedding.shape) == [3, 1]
    assert embedding[0, 0] == 0


def test_carp_variant_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding = embeddings.carp_variant_embedding(
        sequences, "600k", work_dir="temp_carp", cpu_only=True
    )
    assert list(embedding.shape) == [3, 1]


@pytest.mark.slow
def test_evcoupling_variant_embedding():
    wt_sequence = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQGPLDPNSMATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH"
    sequences = []
    for i in range(3):
        sequences += [PROTEIN_ALPHABET[i] + wt_sequence[1:]]
    embedding = embeddings.evcoupling_variant_embedding(
        sequences,
        wt_sequence,
        work_dir="temp_evcoupling",
        uniprot_location=PATH,
        plmc_location="../plmc-master/bin",
        hmmer_location="../bin/",
        test_only=True,
    )
    assert list(embedding.shape) == [3, 1]


def test_concat_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding_funcs = [
        functools.partial(embeddings.one_hot_embedding, alphabet="protein", flatten=True),
        functools.partial(
            embeddings.tape_embedding,
            model="transformer",
            work_dir="temp_tape_mean",
            aggregation="mean",
            cpu_only=True,
        ),
    ]
    embedding = embeddings.concat_embeddings(sequences, embedding_funcs)
    assert list(embedding.shape) == [3, 808]


def test_esm_fineutne_embedding():
    sequences = ["AA", "AC", "AD"]
    embedding = embeddings.esm_finetune_embedding(sequences, "esm-small")
    assert list(embedding.shape) == [3, 3]
