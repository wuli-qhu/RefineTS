# RefineTS

This repository provides the official PyTorch implementation of **RefineTS**, a multivariate time series forecasting framework designed to refine cross-channel interactions and suppress interference in temporal and frequency representations.

## Overview

RefineTS aims to improve multivariate time series forecasting through reliable cross-channel interaction refinement. The framework consists of two main components:

- **Channel Relation-Aware Learner**: Exploits inter-channel relationships to guide knowledge sharing among channel-specific adapters.
- **Interference-Suppressed Temporal Learner**: Combines global temporal representation guidance with input-adaptive frequency filtering to preserve informative patterns and mitigate unreliable cross-channel interactions.

## Code Availability

The main implementation of RefineTS is available in this repository. We are continuing to organize and document the codebase, including training configurations and experimental logs, to facilitate reproducibility.

Additional documentation and updates will be provided as the repository is further refined.

## Requirements

RefineTS is implemented in Python using PyTorch.

## Datasets

Our experiments cover several widely used multivariate time series forecasting benchmarks:

- **ETT**: ETTh1, ETTh2, ETTm1, and ETTm2
- **Electricity**: ECL
- **Traffic**
- **Weather**
- **Solar**
- **PEMS**: PEMS03, PEMS04, PEMS07, and PEMS08

The input sequence length is set to 96. Prediction horizons are {96, 192, 336, 720} for conventional long-term forecasting datasets and {12, 24, 48, 96} for PEMS datasets.

## Training and Evaluation

The repository contains the implementation used for training and evaluating RefineTS.

Model configurations are selected based on validation MSE, with the test set reserved for final evaluation.

The DTW-based channel relation prior is computed offline using the training data and cached before model training.

**Training and Evaluation commands**

[nohup bash scripts/PEMS04/RefineTS_04.sh  > logs/pems04_12.log &]

## Updates

We will continue maintaining the repository and improving the documentation as additional materials are organized.
