"""Tests for pretrained.py."""

from src import pretrained

# pylint: disable=missing-function-docstring


def test_load_trained_flighted_model():
    pretrained.load_trained_flighted_model("Selection")

    pretrained.load_trained_flighted_model("DHARMA")
