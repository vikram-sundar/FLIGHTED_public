"""Tests for common_utils.py."""
import numpy as np

from flighted import common_utils

# pylint: disable=missing-function-docstring


def test_one_hot():
    tensor = common_utils.to_one_hot("BCDA", "ABCDE")
    assert np.allclose(
        np.array(tensor.numpy()),
        np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]).T,
    )


# TODO: Add more test coverage here
