[![Build Status](https://travis-ci.org/ammaraskar/black-holes-black-boxes.svg?branch=master)](https://travis-ci.org/ammaraskar/black-holes-black-boxes)
[![Code Coverage](https://codecov.io/gh/ammaraskar/black-holes-black-boxes/branch/master/graph/badge.svg)](https://codecov.io/gh/ammaraskar/black-holes-black-boxes)

# Finding Black Holes with Black Boxes

This repository contains the supporting code and machine learning classifier
used in the "Finding Black Holes with Black Boxes" paper. There is also an
interactible Jupyter notebook which contains an interface that end users may
provide with observational values to get predictions based on MOCCA simulations.

## Directory Structure

- `tests/` - Folder containing some basic tests.
- `simulations.csv` - CSV file containing data form MOCCA simulations.
- `classifier.py` - Python code to generate and train the classifier.
- `notebook.ipyb` - Interactible Jupyter notebook.