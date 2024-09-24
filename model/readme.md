# Anderson Impurity Model Self-Energy Predictors

This repository contains four trained models for predicting the self-energy of a single-impurity Anderson model using machine learning techniques.

## Configuration

- Random seed: 42
- Dataset: `data/data_50k.csv`
- Batch size: 32
- Validation set size: 10%
- Test set size: 10%
- Input scaling is applied

## Model Architecture

The model uses a custom encoder-decoder transformer with the following key parameters:

- Embedding dimension: 256
- Encoder:
  - Max sequence length: Based on input features
  - Input dimension: 1
  - Feedforward dimension: 1024
  - Number of attention heads: 4
  - Number of layers: 4
- Decoder:
  - Max sequence length: Based on output labels
  - Input dimension: 1
  - Feedforward dimension: 1024
  - Number of attention heads: 4
  - Number of layers: 4
- Dropout: 0.1
- Activation function: GELU
- Bias: Enabled

## Models

1. `model-2024-09-24-hybridization-approximations.pth`
   - Features: beta, u, first-order approximation, occupation, hybridization function, and second-order approximation
   - Performance: 5.3% loss with autoregressive sampling

2. `model-2024-09-19-approximations.pth`
   - Features: beta, u, e_1 to e_5, v_1 to v_5, first-order approximation, occupation, and second-order approximation
   - Performance: 3.4% loss with autoregressive sampling (best performing)

3. `model-2024-09-23-symmetry.pth`
   - Features: beta, u, e_1 to e_5, v_1 to v_5, first-order approximation, occupation, and second-order approximation
   - Note: Labels are paired up
   - Performance: 16.2% loss with autoregressive sampling

4. `model-2024-09-23-hybridization.pth`
   - Features: beta, u, occupation, and hybridization function
   - Performance: 3.5% loss with autoregressive sampling

For more details on the models, their architecture, and performance, please refer to the accompanying thesis.