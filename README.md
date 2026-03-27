# Positional Encoding: How Transformers Understand Order Without Recurrence

This repository contains the code and tutorial material for a machine learning tutorial on positional encoding in transformers. The project explains why self-attention does not inherently represent sequence order, how positional encoding addresses this limitation, and how different positional encoding choices affect learning on an order-sensitive task.

## Overview

Transformers process all tokens in parallel through self-attention, which makes them powerful but also creates a challenge: token order is not built into the architecture in the same way as in recurrent models. This project demonstrates how positional encoding makes order available to the model by combining token embeddings with positional information before attention is applied.

The tutorial focuses on:
- why sequence order matters
- why self-attention alone is order-agnostic
- how sinusoidal positional encoding works
- how learned and fixed positional encodings compare on a toy classification task

## How to Run

Set up a Python environment and install the required dependencies before running the project. After the environment is ready, run the notebook or script from start to finish so that the dataset generation, model training, and visualisations are executed in the correct order. Running the workflow in sequence will reproduce the figures used in the tutorial, including the sinusoidal encoding heatmap, the performance comparison across positional encoding settings, and the validation accuracy curves.

The code is designed to run on a standard CPU and does not require a GPU. Random seeds are fixed to improve reproducibility.

## Dependencies

The project uses common Python libraries for deep learning, numerical computation, and visualisation. In particular, it requires:
- PyTorch
- NumPy
- Matplotlib

These dependencies are sufficient to reproduce the experiment and figures in the tutorial.

## Output

Running the project reproduces the main teaching results of the tutorial:
- a visualisation of sinusoidal positional encodings
- a comparison of performance with no positional encoding, sinusoidal positional encoding, and learned positional encoding
- validation accuracy curves showing how models learn under each positional setting

These outputs support the central conclusion that transformers require explicit positional information to solve order-sensitive tasks.

## Licence

This project is released under the MIT License. This allows the code to be reused, modified, and distributed, provided the original licence terms and attribution are retained.
