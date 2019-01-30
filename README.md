[![Build Status](https://travis-ci.org/ammaraskar/black-holes-black-boxes.svg?branch=master)](https://travis-ci.org/ammaraskar/black-holes-black-boxes)
[![Code Coverage](https://codecov.io/gh/ammaraskar/black-holes-black-boxes/branch/master/graph/badge.svg)](https://codecov.io/gh/ammaraskar/black-holes-black-boxes)

# Finding Black Holes with Black Boxes

This repository contains the supporting code and machine learning classifier
used in the "Finding Black Holes with Black Boxes" paper. There is also an
interactible Jupyter notebook which contains an interface that end users may
provide with observational values to get predictions based on MOCCA simulations.

## Use Interactively

To provide observational features and use the code interactively, click here:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ammaraskar/black-holes-black-boxes/master?filepath=notebook.ipynb)

Once launched just click "Run" in the top and scroll down to the form.

## Directory Structure

- `tests/` - Folder containing some basic tests.
- `simulations.csv` - CSV file containing data form MOCCA simulations.
- `classifier.py` - Python code to generate and train the classifier.
- `notebook.ipnyb` - Interactible Jupyter notebook.

## Running Tests

You may run the unit tests with:

`python -m pytest`