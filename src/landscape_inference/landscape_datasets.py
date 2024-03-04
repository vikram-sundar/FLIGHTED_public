"""Dataset objects for fitness landscape learning."""

import torch
from torch.utils.data import Dataset

# pylint: disable=no-member


class FitnessDataset(Dataset):
    """A dataset class to summarize all fitness data from fitness models.

    This dataset has x-coordinates corresponding to sequences and y-coordinates
    corresponding to fitnesses. It also has a type attribute telling what type
    of data is stored at each point (currently supporting predictions and
    upper bounds).

    Attributes:
        sequences: torch.Tensor with sequences, featurized as specified
        fitnesses: torch.Tensor with fitnesses
        variances: torch.Tensor with variances
    """

    def __init__(self, sequences, fitnesses, variances):
        """Reads in fitness data and stores it for training.

        Args:
            sequences: torch.Tensor with featurized sequences.
            fitnesses: torch.Tensor of fitnesses for each sequence, as floats.
            variances: torch.Tensor of variances (or 1 for all as a dummy), as floats.
        """
        super().__init__()
        self.sequences = sequences
        self.fitnesses = torch.tensor(fitnesses, dtype=torch.float32)
        self.variances = torch.tensor(variances, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.fitnesses[idx], self.variances[idx]
