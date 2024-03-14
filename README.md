# FLIGHTED

FLIGHTED, a machine learning method for inference of fitness landscapes from high-throughput experimental data.

Author: Vikram Sundar.

## Installation

For convenience, you can install FLIGHTED on pip using `pip install flighted` in a clean virtual environment. This installs FLIGHTED, but not any other packages you may want for landscape modeling.

You can also install directly from the source code here on Github. The following packages are required for use and development of FLIGHTED:

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

The following packages are not required for use of FLIGHTED, but needed for landscape modeling:

- fair-esm
- tape_proteins
- evcouplings
- transformers
- sentencepiece
- sequence-models

An env.yml file is provided for your convenience, especially when installing the tricky dependencies around `evcoupling`, though it has a few packages that are not necessary for FLIGHTED. 

Once these packages are installed, add the FLIGHTED directory to your Python path (either as an environment variable or within an individual script) to run the model.

## Basic Use

### FLIGHTED-Selection

To run FLIGHTED-Selection, first prepare a tensor with the following five columns:

1. Variant number (0 - number of variants)
2. Number of samples of this variant observed pre-selection
3. Number of samples of this variant observed post-selection
4. Total samples taken pre-selection
5. Total samples taken post-selection

This should be a tensor with shape (number of variants, 5). Then run:

```
from flighted import pretrained

hparams, model = pretrained.load_trained_flighted_model("Selection", cpu_only=True)
fitness_mean, fitness_var = model.selection_reverse_model(selection_data)
```

The result is the fitness mean and variance (not standard deviation).

### FLIGHTED-DHARMA

To run FLIGHTED-DHARMA, collect all DHARMA reads for a given variant into a single tensor with shape (number of reads, canvas length, 3). The last column is a one-hot encoding for 3 categories: wild-type, C to T Mutation, other mutation. Then run:

```
from flighted import pretrained

hparams, model = pretrained.load_trained_flighted_model("DHARMA", cpu_only=True)
mean, variance = model.infer_fitness_from_dharma(dharma_data)
```

The result is the fitness mean and variance (not standard deviation).

## Tests

If you install from Github, you should test the code after installation to ensure that all the requirements are installed correctly and the package is working. To test the code, run `make test-all` in the main directory. This will automatically run a linter and all tests.

While developing code, you can skip slow tests by running `make test` instead.

## Data

Data has been deposited at Zenodo [here](https://zenodo.org/records/10777739) and [here](https://zenodo.org/records/10779337). Data should be downloaded and installed in a `Data/` folder within this directory so the filepaths match in scripts and notebooks. The trained model weights and hyperparameters are on Github and may be automatically downloaded as described above.

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
