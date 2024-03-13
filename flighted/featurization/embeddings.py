"""This class generates embeddings from a list of sequences."""
import os
import re
import subprocess
import sys

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

try:
    from esm import FastaBatchedDataset, pretrained
except ModuleNotFoundError:
    pass

try:
    from evcouplings.couplings import CouplingsModel
    from evcouplings.utils import read_config_file, write_config_file
    from evcouplings.utils.pipeline import execute
except ModuleNotFoundError:
    pass
try:
    from sequence_models.pretrained import (
        load_model_and_alphabet as carp_load_model_and_alphabet,
    )
    from sequence_models.utils import parse_fasta
except ModuleNotFoundError:
    pass
from torch.utils.data import DataLoader

try:
    from transformers import T5EncoderModel, T5Tokenizer
except ModuleNotFoundError:
    pass

from flighted.common_utils import PROTEIN_ALPHABET, RNA_ALPHABET, to_one_hot

TAPE_PRETRAINED_DICT = {"transformer": "bert-base", "unirep": "babbler-1900"}
TAPE_TOKENIZER = {"transformer": "iupac", "unirep": "unirep"}
ESM_MODEL = {
    "esm-2-15b": "esm2_t48_15B_UR50D",
    "esm-2-3b": "esm2_t36_3B_UR50D",
    "esm-2-650m": "esm2_t33_650M_UR50D",
    "esm-2-150m": "esm2_t30_150M_UR50D",
    "esm-2-35m": "esm2_t12_35M_UR50D",
    "esm-2-8m": "esm2_t6_8M_UR50D",
    "esm-1b": "esm1b_t33_650M_UR50S",
    "esm-small": "esm1_t6_43M_UR50S",
    "esm-1v-1": "esm1v_t33_650M_UR90S_1",
    "esm-1v-2": "esm1v_t33_650M_UR90S_2",
    "esm-1v-3": "esm1v_t33_650M_UR90S_3",
    "esm-1v-4": "esm1v_t33_650M_UR90S_4",
    "esm-1v-5": "esm1v_t33_650M_UR90S_5",
}
ESM_MAX_LENGTH = 1022  # maximum length of sequences for ESM model
PROTTRANS_MODEL = {"prott5-xl-u50": "Rostlab/prot_t5_xl_half_uniref50-enc"}
CARP_MODEL = {"600k": "carp_600k", "38M": "carp_38M", "76M": "carp_76M", "640M": "carp_640M"}

# pylint: disable=no-member, invalid-name


def one_hot_embedding(sequences, alphabet="protein", flatten=False):
    """Converts a list of sequences to a one-hot representation.

    Args:
        sequences: list of strings representing the given sequences.
        alphabet: string representing the given alphabet (either rna or protein).
        flatten: boolean, whether to flatten the per-sequence embedding.

    Returns:
        torch.tensor with the given one-hot representation (number of sequences x alphabet length x sequence length or number of sequences x latent dimension if flatten is True).
    """
    if alphabet == "protein":
        alphabet_string = PROTEIN_ALPHABET
    elif alphabet == "rna":
        alphabet_string = RNA_ALPHABET
    else:
        raise Exception(f"Unexpected alphabet {alphabet} found.")

    if flatten:
        embeddings = torch.stack([to_one_hot(sequence, alphabet_string) for sequence in sequences])
        return torch.flatten(embeddings, start_dim=1)
    return torch.stack([to_one_hot(sequence, alphabet_string) for sequence in sequences])


def write_fasta(sequences, fasta_filename, n_batches=1):
    """Writes out a list of sequences to a group of FASTA files.

    Args:
        sequences: list of strings representing the given sequences.
        fasta_filename: string, filename of FASTA file for writing out (the batch number and .fasta
            will be appended).
        n_batches: int, number of batches to write out.

    Returns:
        An array containing filenames of FASTA files used for storing embeddings.
    """
    fasta_filenames = [f"{fasta_filename}_batch{i}.fasta" for i in range(n_batches)]
    for batch, fasta_filename_batch in zip(np.array_split(sequences, n_batches), fasta_filenames):
        temp_seqs = [
            SeqRecord(Seq(sequence), id=sequence, description="")
            for j, sequence in enumerate(batch)
        ]
        with open(fasta_filename_batch, "w") as f:
            SeqIO.write(temp_seqs, f, "fasta")
    return fasta_filenames


def tape_embedding(sequences, model, aggregation="none", work_dir="", n_batches=1, cpu_only=False):
    """Converts a list of protein sequences to an embedding from the TAPE repository.

    Args:
        sequences: list of strings representing the given sequences.
        model: string, which model to use (either 'transformer' or 'unirep').
        aggregation: string, which aggregation method to use ('none' or 'mean')
        work_dir: string, working directory to use (if any)
        n_batches: int, number of batches
        cpu_only: boolean, whether to only use CPU

    Returns:
        torch.tensor with the given representation (number of sequences x latent dimension length x sequence length or number of sequences x latent dimension length if mean is used).
    """
    if work_dir != "" and not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fasta_filenames = write_fasta(
        sequences, os.path.join(work_dir, "tape_embedding_temp"), n_batches=n_batches
    )
    all_embeddings = [[] for _ in sequences]
    for i, fasta_filename in enumerate(fasta_filenames):
        temp_filename = os.path.join(work_dir, f"tape_output_temp_batch{i}.npz")
        command = [
            "tape-embed",
            model,
            fasta_filename,
            temp_filename,
            TAPE_PRETRAINED_DICT[model],
            "--tokenizer",
            TAPE_TOKENIZER[model],
            "--full_sequence_embed",
        ]
        if cpu_only:
            command += ["--no_cuda"]
        try:
            _ = subprocess.run(command, check=True)
        except Exception as exc:
            raise Exception("TAPE package must be installed to use TAPE embeddings.") from exc
        with open(temp_filename, "rb") as f:
            embeddings = np.load(f, allow_pickle=True)
            for sequence, embedding in embeddings.items():
                all_embeddings[sequences.index(sequence)] = torch.Tensor(
                    np.transpose(embedding[()]["seq"])[:, 1:-1]
                )
        subprocess.run(["rm", temp_filename], check=True)
        subprocess.run(["rm", fasta_filename], check=True)
        if work_dir != "":
            subprocess.run(["rm", "-rf", work_dir], check=True)
    if aggregation == "none":
        return torch.stack(all_embeddings)
    return torch.mean(torch.stack(all_embeddings), axis=2)


def esm_embedding(
    sequences, model, aggregation="none", work_dir="", batch_size=4096, cpu_only=False
):
    """Converts a list of protein sequences to an embedding from the Facebook ESM repository.

    Only supports models that take sequences (not MSA transformer, not ESM-1F).

    Args:
        sequences: list of strings representing the given sequences.
        model: string, which model to use (see ESM_MODEL)
        aggregation: string, which aggregation method to use ('none' or 'mean')
        work_dir: string, working directory to use (if any)
        batch_size: int, number of data points per batch
        cpu_only: boolean, whether to only use CPU

    Returns:
        torch.tensor with the given representation (number of sequences x latent dimension length x sequence length or number of sequences x latent dimension length if mean is used).
    """
    if "esm" not in sys.modules:
        raise Exception("ESM package must be installed to use ESM embeddings.")

    if work_dir != "" and not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fasta_filename = write_fasta(
        sequences, os.path.join(work_dir, "esm_embedding_temp"), n_batches=1
    )[0]
    model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL[model])
    if torch.cuda.is_available() and not cpu_only:
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_filename)
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    data_loader = DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )

    all_embeddings = [[] for _ in sequences]
    with torch.no_grad():
        for _, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available() and not cpu_only:
                toks = toks.to(device="cuda", non_blocking=True)

            # Truncation as specified by ESM model
            toks = toks[:, :ESM_MAX_LENGTH]

            out = model(toks, repr_layers=[model.num_layers])
            representations = out["representations"][model.num_layers].to(device="cpu")

            for i, label in enumerate(labels):
                all_embeddings[sequences.index(label)] = (
                    representations[i, 1 : len(strs[i]) + 1].clone().transpose(0, 1)
                )

    subprocess.run(["rm", fasta_filename], check=True)
    if work_dir != "":
        subprocess.run(["rm", "-rf", work_dir], check=True)
    if aggregation == "none":
        return torch.stack(all_embeddings)
    return torch.mean(torch.stack(all_embeddings), axis=2)


def prottrans_embedding(sequences, model, aggregation="none", batch_size=4096, cpu_only=False):
    """Converts a list of protein sequences to an embedding from the ProtTrans repository.

    Args:
        sequences: list of strings representing the given sequences.
        model: string, which model to use (see PROTTRANS_MODEL)
        aggregation: string, which aggregation method to use ('none' or 'mean')
        batch_size: int, number of data points per batch
        cpu_only: boolean, whether to only use CPU

    Returns:
        torch.tensor with the given representation (number of sequences x latent dimension length x sequence length or number of sequences x latent dimension length if mean is used).
    """
    if "transformers" not in sys.modules:
        raise Exception("ProtTrans must be installed to use ProtTrans embeddings.")

    if torch.cuda.is_available() and not cpu_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained(PROTTRANS_MODEL[model], do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTTRANS_MODEL[model]).to(device)

    # replace rare/ambiguous amino acids and introduce whitespace
    sequences_processed = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences
    ]
    all_embeddings = []
    for i in range(0, len(sequences_processed), batch_size):
        start = i
        end = min(start + batch_size, len(sequences_processed))
        sequences_batch = sequences_processed[start:end]
        ids = tokenizer(sequences_batch, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        all_embeddings += [embedding_repr.last_hidden_state[:, : len(sequences[0])].transpose(1, 2)]

    embedding = torch.cat(all_embeddings, axis=0)
    if aggregation == "mean":
        return torch.mean(embedding, axis=2)
    return embedding


def carp_embedding(
    sequences, model, aggregation="none", work_dir="", batch_size=4096, cpu_only=False
):
    """Converts a list of protein sequences to an embedding from the CARP repository.

    Args:
        sequences: list of strings representing the given sequences.
        model: string, which model to use (see CARP_MODEL)
        aggregation: string, which aggregation method to use ('none' or 'mean')
        work_dir: string, working directory to use (if any)
        batch_size: int, number of data points per batch
        cpu_only: boolean, whether to only use CPU

    Returns:
        torch.tensor with the given representation (number of sequences x latent dimension length x sequence length or number of sequences x latent dimension length if mean is used).
    """
    if "sequence_models" not in sys.modules:
        raise Exception("CARP must be installed to use CARP embeddings.")

    if work_dir != "" and not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fasta_filename = write_fasta(
        sequences, os.path.join(work_dir, "carp_embedding_temp"), n_batches=1
    )[0]

    model, collater = carp_load_model_and_alphabet(CARP_MODEL[model])
    if torch.cuda.is_available() and not cpu_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    seqs = parse_fasta(fasta_filename)
    lengths = [len(s) for s in seqs]
    seqs = [[s] for s in seqs]
    all_embeddings = []
    for i in range(0, len(seqs), batch_size):
        start = i
        end = min(start + batch_size, len(seqs))
        seq_batch = seqs[start:end]
        lengths_batch = lengths[start:end]
        x = collater(seq_batch)[0].to(device)
        results = model(x)
        for _, rep in results["representations"].items():
            for r, ell in zip(rep, lengths_batch):
                all_embeddings += [r[:ell].detach()]
        del results
        del x
        torch.cuda.empty_cache()

    subprocess.run(["rm", fasta_filename], check=True)
    if work_dir != "":
        subprocess.run(["rm", "-rf", work_dir], check=True)
    all_embeddings = torch.stack(all_embeddings).transpose(1, 2)
    if aggregation == "mean":
        return torch.mean(all_embeddings, axis=2)
    return all_embeddings


def esm_variant_embedding(sequences, wt_sequence, model, cpu_only=False):
    """Computes zero-shot variant probabilities from the ESM model for each sequence.

    Uses the masked marginal approach proposed in the ESM paper. Should work on any ESM-1v
    model and ESM-2.

    Args:
        sequences: list of strings representing the given sequences.
        wt_sequence: string with wild-type sequence.
        model: string, which model to use (see ESM_MODEL)
        cpu_only: boolean, whether to only use CPU

    Returns:
        torch.tensor with the given representation (number of sequences x 1).
    """
    if "esm" not in sys.modules:
        raise Exception("ESM package must be installed to use ESM embeddings.")

    model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL[model])
    if torch.cuda.is_available() and not cpu_only:
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    data = [("wild-type", wt_sequence)]
    _, _, batch_tokens = batch_converter(data)
    all_token_probs = []
    for i in range(batch_tokens.size(1)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        if torch.cuda.is_available() and not cpu_only:
            batch_tokens_masked = batch_tokens_masked.cuda()
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
        all_token_probs.append(token_probs[:, i])
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

    final_probs = []
    for sequence in sequences:
        score = 0
        for i, (mut, wt) in enumerate(zip(sequence, wt_sequence)):
            if mut != wt:
                score += (
                    token_probs[0, i + 1, alphabet.get_idx(mut)]
                    - token_probs[0, i + 1, alphabet.get_idx(wt)]
                )
        final_probs += [score]

    return torch.tensor(final_probs).unsqueeze(1)


def carp_variant_embedding(sequences, model, work_dir="", batch_size=4096, cpu_only=False):
    """Converts zero-shot variant probabilities from the CARP model for each sequence.

    Uses default logits produced from the CARP model.

    Args:
        sequences: list of strings representing the given sequences.
        model: string, which model to use (see CARP_MODEL)
        work_dir: string, working directory to use (if any)
        batch_size: int, number of data points per batch
        cpu_only: boolean, whether to only use CPU

    Returns:
        torch.tensor with the given representation (number of sequences x 1).
    """
    if "sequence_models" not in sys.modules:
        raise Exception("CARP must be installed to use CARP embeddings.")

    if work_dir != "" and not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fasta_filename = write_fasta(
        sequences, os.path.join(work_dir, "carp_embedding_temp"), n_batches=1
    )[0]

    model, collater = carp_load_model_and_alphabet(CARP_MODEL[model])
    if torch.cuda.is_available() and not cpu_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    seqs = parse_fasta(fasta_filename)
    lengths = [len(s) for s in seqs]
    seqs = [[s] for s in seqs]
    all_embeddings = []
    for i in range(0, len(seqs), batch_size):
        start = i
        end = start + batch_size
        seq_batch = seqs[start:end]
        lengths_batch = lengths[start:end]
        x = collater(seq_batch)[0].to(device)
        results = model(x, logits=True)
        rep = results["logits"]
        for r, ell, src in zip(rep, lengths_batch, x):
            all_embeddings += [
                r.log_softmax(dim=-1)[:ell][torch.arange(len(src)), src]
                .mean()
                .detach()
                .cpu()
                .numpy()[()]
            ]

    subprocess.run(["rm", fasta_filename], check=True)
    if work_dir != "":
        subprocess.run(["rm", "-rf", work_dir], check=True)
    print(all_embeddings)
    return torch.tensor(all_embeddings).unsqueeze(1)


def evcoupling_variant_embedding(
    sequences,
    wt_sequence,
    work_dir,
    uniprot_location="",
    plmc_location="",
    hmmer_location="",
    test_only=False,
):
    """
    Args:
        sequences: list of strings representing the given sequences.
        wt_sequence: string with wild-type sequence.
        work_dir: string, working directory to use (if any).
        uniprot_location: string, location of Uniprot databases.
        plmc_location: string, location of plmc source file.
        hmmer_location: string, location of HMMER source files.
        test_only: boolean, whether this is being run as a test only.

    Returns:
        torch.tensor with the given representation (number of sequences x 1).
    """
    if "evcouplings" not in sys.modules:
        raise Exception("EVCouplings must be installed to use EVCouplings embeddings.")

    if os.path.exists(work_dir):
        subprocess.run(["rm", "-rf", work_dir], check=True)
    os.makedirs(work_dir)
    fasta_filename = write_fasta(
        [wt_sequence], os.path.join(work_dir, "evcoupling_embedding_temp"), n_batches=1
    )[0]

    config = read_config_file("src/featurization/EVcoupling_config.txt", preserve_order=True)
    config["global"]["prefix"] = work_dir
    config["global"]["sequence_id"] = "wt_protein"
    config["global"]["sequence_file"] = fasta_filename

    config["databases"]["uniref100"] = os.path.join(uniprot_location, "uniref100_2023_10.fasta")

    if test_only:
        config["databases"]["uniref100"] = os.path.join(uniprot_location, "test_db.fasta")

    config["tools"]["jackhmmer"] = os.path.join(hmmer_location, "jackhmmer")
    config["tools"]["hmmbuild"] = os.path.join(hmmer_location, "hmmbuild")
    config["tools"]["hmmsearch"] = os.path.join(hmmer_location, "hmmsearch")
    config["tools"]["hhfilter"] = os.path.join(hmmer_location, "hhfilter")
    config["tools"]["plmc"] = os.path.join(plmc_location, "plmc")

    write_config_file(os.path.join(work_dir, "temp_config.txt"), config)
    _ = execute(**config)

    couplings_model = CouplingsModel(os.path.join(work_dir, "couplings", work_dir + ".model"))
    wt_energy = couplings_model.hamiltonians([couplings_model.seq()])[0][0]
    energies = []
    for sequence in sequences:
        energies += [couplings_model.hamiltonians([list(sequence)])[0][0] - wt_energy]

    subprocess.run(["rm", fasta_filename], check=True)
    subprocess.run(["rm", "-rf", work_dir], check=True)
    subprocess.run(["rm", work_dir + "_final.outcfg"], check=True)

    return torch.tensor(energies, dtype=torch.float32).unsqueeze(1)


def concat_embeddings(sequences, embedding_funcs):
    """Concatenate a number of embeddings into a single combined embedding.

    Each embedding should have dimensions sequence length x latent dimension.

    Args:
        sequences: list of strings representing the given sequences.
        embedding_funcs: list of embeddings to concatenate.

    Returns:
        torch.tensor with the given representation (number of sequences x latent dimension length).
    """
    embeddings = [embedding_func(sequences) for embedding_func in embedding_funcs]
    return torch.cat(embeddings, axis=1).to(torch.float32)


def esm_finetune_embedding(sequences, model):
    """Converts a list of protein sequences to an ESM tokenized version.

    Only for training a fine-tuned ESM model (not a normal ESM model). Only supports
    models that take sequences (not MSA transformer, not ESM-1F).

    Args:
        sequences: list of strings representing the given sequences.
        model: string, which model to use (see ESM_MODEL)

    Returns:
        torch.tensor with the given representation (number of sequences x sequence length + 1).
    """
    if "esm" not in sys.modules:
        raise Exception("ESM package must be installed to use ESM embeddings.")

    _, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL[model])
    _, _, toks = alphabet.get_batch_converter()([(str(i), seq) for i, seq in enumerate(sequences)])
    return toks
