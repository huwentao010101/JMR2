# NMR Spectral Analysis Project

This repository hosts the solution for analyzing Nuclear Magnetic Resonance (NMR) spectral data through an innovative integration of an Attention-Based Bidirectional LSTM Autoencoder (ABBL-SB) alongside various clustering and similarity measures.

## Table of Contents

- [Overview](#overview)
- [Data Sets](#data-sets)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Overview

Our aim with this project is to meticulously analyze NMR spectral data to unravel the hidden patterns and relationships potentially associated with chemical compositions, sample origins, or other pertinent experimental factors. The analysis pipeline comprises stages such as data generation, preprocessing, feature extraction, and clustering to ensure a comprehensive examination.

## Data Sets

Within the `dataset` directory, you'll find synthetic NMR spectral data sets meticulously crafted for this project. Each data set resides in an individual CSV file. For a detailed exposition of these data sets, please refer to the `README.md` file located in the `dataset` directory.

## Code Structure

The core components of this project are divided across the following scripts for better manageability and understanding:

- `Attention-Based BiLSTM Autoencoder.py`: Facilitates sequence reconstruction and detailed feature representation using the ABBL-SB model.
- `Cluster.py`: Hosts the K-means clustering algorithm for efficiently grouping the extracted feature representations.
- `compare.py`: Dedicated to implementing various sequence similarity measures crucial for comparing NMR spectra.
- `Data_generation.py`: Capable of generating synthetic NMR spectral sequences tailored for analysis.
- `Feature Extraction.py`: Employs intelligent algorithms to extract meaningful features from preprocessed NMR spectral data.
- `Spectral Preprocessing.py`: Takes charge of preprocessing the raw NMR spectral data focusing on noise reduction, baseline correction, and normalization for accurate analysis.

## Dependencies

To ensure the smooth execution of the project, it relies on the following Python packages:

- `numpy`
- `scipy`
- `sklearn`
- `matplotlib`
- `dtw` (Dynamic Time Warping)

To install these dependencies, run the following command:

```bash
pip install numpy scipy scikit-learn matplotlib dtw
```
## Usage
- `Harness the potential of this project by following these instructions:
- `Deposit the synthetic data sets in the dataset directory.
- `Invoke Data_generation.py to generate the additional needed data.
- `Apply preprocessing to the data by executing Spectral Preprocessing.py.
- `Leverage Feature Extraction.py to derive features from the preprocessed data.
- `Train the Attention-Based BiLSTM Autoencoder by running Attention-Based BiLSTM Autoencoder.py.
- `Conduct clustering on the extracted features utilizing Cluster.py.
- `Explore different sequence similarity measures by executing compare.py.
## License
This project is generously made available under the MIT License, promoting free software initiatives.
