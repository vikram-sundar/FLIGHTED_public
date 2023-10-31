# FLIGHTED

FLIGHTED, a machine learning method for inference of fitness landscapes from high-throughput experimental data.

Author: Vikram Sundar.

## Requirements

    - numpy
    - scipy
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn
    - biopython
    - pytest
    - pytest-pylint
    - python-levenshtein
    - pytorch
    - black
    - isort
    - pre-commit
    - pyro
    - fair-esm
    - tape_proteins
    - evcouplings
    - transformers
    - sentencepiece
    - sequence-models

An env.yml file is provided, but it might not necessarily be complete.

TODO: add some sort of requirements file.

## Tests

Before using the package, you should test the code after installation to ensure that all the requirements are installed correctly and the package is working. To test the code, run `make test-all` in the main directory. This will automatically run a linter and all relevant tests.

While developing code, you can skip slow tests by running `make test` instead.

