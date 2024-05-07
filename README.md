# NMR Spectral Analysis Project

This repository contains the code and data for the analysis of Nuclear Magnetic Resonance (NMR) spectral data using an Attention-Based Bidirectional LSTM Autoencoder (ABBL-SB) and various clustering and similarity measures.

## Table of Contents

1. [Overview](#overview)
2. [Data Sets](#data-sets)
3. [Code Structure](#code-structure)
4. [Dependencies](#dependencies)
5. [Usage](#usage)
6. [License](#license)

## Overview

The project aims to analyze NMR spectral data to uncover intrinsic patterns and relationships that may correlate with chemical composition, sample origin, or other experimental factors. The analysis pipeline includes data generation, preprocessing, feature extraction, and clustering.

## Data Sets

The `dataset` directory contains synthetic NMR spectral data sets used in this project. Each data set is stored in a separate CSV file. A `README.md` file within the `dataset` directory provides detailed information about the data sets.

## Code Structure

The project code is organized into the following scripts:

- `Attention-Based BiLSTM Autoencoder.py`: Implements the ABBL-SB for sequence reconstruction and feature representation.
- `Cluster.py`: Contains the K-means clustering algorithm for grouping the feature representations.
- `compare.py`: Implements various sequence similarity measures for comparing NMR spectra.
- `Data_generation.py`: Generates synthetic NMR spectral sequences for analysis.
- `Feature Extraction.py`: Extracts features from preprocessed NMR spectral data.
- `Spectral Preprocessing.py`: Preprocesses raw NMR spectral data for noise reduction, baseline correction, and normalization.

## Dependencies

The project requires the following Python packages:

- `numpy`
- `scipy`
- `sklearn`
- `matplotlib`
- `dtw` (for Dynamic Time Warping)

You can install these dependencies using `pip`:

```bash
pip install numpy scipy scikit-learn matplotlib dtw

## Usage
To use the project, follow these steps:

Place the synthetic data sets in the dataset directory.
Run Data_generation.py to generate additional required data.
Preprocess the data using Spectral Preprocessing.py.
Extract features from the preprocessed data with Feature Extraction.py.
Train the Attention-Based BiLSTM Autoencoder using Attention-Based BiLSTM Autoencoder.py.
Perform clustering on the embedded representations with Cluster.py.
Compare different sequence similarity methods by running compare.py.

## License
This project is licensed under the MIT License.
