"""Dataset objects for FLIGHTED inference."""
import numpy as np
import torch
from torch.utils.data import Dataset

# pylint: disable=no-member, not-callable


class DHARMADataset(Dataset):
    """A dataset class to summarize all DHARMA readout data.

    This dataset stores sequences of variants and their corresponding DHARMA readouts.

    Attributes:
        supervised: boolean, whether to run in supervised mode
        combine_reads: boolean, whether to combine DHARMA reads for the same sequence
        sequences: List of strings with variant sequences
        sequences_processed: torch.Tensor with sequences featurized as specified
        dharma_readouts: torch.Tensor with DHARMA readouts
    """

    def __init__(self, dharma_df, sequence_lambda, supervised, hparams):
        """Reads in DHARMA and fluorescence data from pandas dataframes.

        Args:
            dharma_df: pd.DataFrame holding DHARMA readout information
            sequence_lambda: function to apply to sequences to featurize them
            supervised: boolean, whether to run in supervised mode
            hparams: Dict containing variant_column, dharma_readout_prefix, num_residues,
                and combine_reads
        """
        super().__init__()
        self.supervised = supervised
        self.combine_reads = hparams["combine_reads"]
        variant_column = hparams["variant_column"]
        dharma_readout_prefix = hparams["dharma_readout_prefix"]
        num_residues = hparams["num_residues"]

        if self.combine_reads:
            self.sequences = dharma_df[variant_column].unique()
            self.sequences_index = dharma_df[variant_column].values
        else:
            self.sequences = dharma_df[variant_column].values
        self.sequences_processed = torch.stack(
            [sequence_lambda(sequence) for sequence in dharma_df[variant_column].values]
        )
        cols = []
        for i in range(num_residues):
            cols += [dharma_df[f"{dharma_readout_prefix}_{i}"].values]
        self.dharma_readouts = torch.tensor(np.array(cols), dtype=torch.float32).transpose(0, 1)
        self.dharma_readouts = torch.nn.functional.one_hot(
            self.dharma_readouts.long(), num_classes=3
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.combine_reads:
            filter_vals = self.sequences_index == self.sequences[idx]
        else:
            filter_vals = idx

        return self.sequences_processed[filter_vals], self.dharma_readouts[filter_vals]


def collate_dharma(batch):
    """A collate function to collate a batch of DHARMA dataset values.
    Args:
        batch: A list of tuples of DHARMA dataset values,
            including sequences and DHARMA readouts
    Returns:
        Collated sequences and DHARMA readouts.
    """
    seq_batch = []
    dharma_batch = []
    for datapoint in batch:
        seq_batch += [datapoint[0]]
        dharma_batch += [datapoint[1]]
    seq_batch = torch.cat(seq_batch, dim=0)
    dharma_batch = torch.cat(dharma_batch, dim=0)
    return seq_batch, dharma_batch


class SelectionDataset(Dataset):
    """A dataset class to summarize all selection data."""

    def __init__(self, experiments):
        self.experiments = experiments

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        return self.experiments[idx]
