# Variational Physics-Informed Neural Networks for Non-Linear Heat Transfer

This repository contains the code and data to reproduce the numerical experiments from the paper. 
We solve the 1D non-linear heat equation with temperature-dependent thermophysical parameters and data-driven boundary conditions.

## Structure
* `src/`: Model implementations (RVPINN) and notebooks to train models and reproduce the results.
* `data/`: Experimental temperature data and material parameters.
* 'figs/': The figures in the paper.

## Requirements
* Python 3.10+
* TensorFlow >= 2.10
* TensorFlow Probability
* NumPy, SciPy, Matplotlib

## Usage
To run the RVPINN model and generate the results figures:
1. Install dependencies.
2. Execute the `notebooks/parabolicRVPINNs_Paper.ipynb` notebook.
