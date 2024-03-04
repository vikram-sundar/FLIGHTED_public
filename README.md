# FLIGHTED

FLIGHTED, a machine learning method for inference of fitness landscapes from high-throughput experimental data.

Author: Vikram Sundar.

## Installation

The following packages are required for use and development of FLIGHTED:

- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- seaborn
- biopython
- pytest
- pytest-pylint
- pytorch
- black
- isort
- pre-commit
- pyro-ppl
- fair-esm
- tape_proteins
- evcouplings
- transformers
- sentencepiece
- sequence-models

An env.yml file is provided for your convenience, especially when installing the tricky dependencies around `evcoupling`, though it has a few packages that are not necessary for FLIGHTED. If you do not want to run the full landscape modeling and just want to use FLIGHTED, you do not need to install anything below `pyro-ppl` on the list above, which can make your environment considerably simpler.

Once these packages are installed, add the FLIGHTED directory to your Python path (either as an environment variable or within an individual script) to run the model.

## Tests

Before using the package, you should test the code after installation to ensure that all the requirements are installed correctly and the package is working. To test the code, run `make test-all` in the main directory. This will automatically run a linter and all tests.

While developing code, you can skip slow tests by running `make test` instead.

## Data

Data, including the trained model weights for FLIGHTED-Selection and FLIGHTED-DHARMA, has been deposited at Zenodo [here](https://zenodo.org/records/10777739) and [here](https://zenodo.org/records/10779337). Data should be downloaded and installed in a `Data/` folder within this directory so the filepaths match in scripts and notebooks.

## Scripts

A number of scripts are provided for the most important functionality of FLIGHTED. Specifically, within `scripts/FLIGHTED_selection/`, we have:

1. `selection_simulations.py`: generates simulated single-step selection experiments for use as training data for FLIGHTED-selection.
2. `train_selection_model.py`: trains FLIGHTED-selection (should be run on a GPU).
3. `infer_fitness_from_selection.py`: infers fitnesses with variance from FLIGHTED-selection on a provided landscape. The script is written to use the GB1 landscape, but can be easily adapted.
4. `train_landscape_model.py`: trains a landscape model (with the given options) on the GB1 landscape with and without FLIGHTED-selection. Can be extended to train models on other landscapes.

Within `scripts/FLIGHTED_DHARMA`, we have:

1. `train_dharma_model.py`: trains FLIGHTED-DHARMA on the provided data (should be run on a GPU).
2. `infer_fitness_from_dharma.py`: infers fitnesses from DHARMA data.
3. `train_landscape_model.py`: trains a landscape model (with the given options) on the TEV landscape. Can be extended to train models on other landscapes.

These scripts should allow you to regenerate all data provided in Zenodo, aside from the DHARMA input data that was obtained experimentally.

## Notebooks

The provided notebooks reproduce all figures used in the main text and supplement, when provided with the Data deposited in Zenodo. They also provide examples of how to evaluate the performance of FLIGHTED-selection and FLIGHTED-DHARMA.

## Citing FLIGHTED

For now, please cite the MLSB paper.
