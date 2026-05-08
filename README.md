# Conditional Date Generative Models

This repository contains the implementation of four distinct generative AI models designed to solve a structured sequence generation task: generating valid dates (`dd-mm-yyyy`) conditioned on four discrete categorical inputs (Day of Week, Month, Leap Year status, and Decade).

Because multiple valid dates can exist for a single set of conditions, standard cross-entropy accuracy is an invalid evaluation metric. This project utilizes a custom Condition Satisfaction Rate powered by Python's native `calendar` module to validate outputs.

## Repository Structure

```text
repo/
├── data/
│   ├── data.txt                      # Full training dataset
│   └── example_input.txt             # Sample input for testing predict.py
├── model/
│   ├── predict.py                    # REQUIRED EXACT PATH: Inference script
│   ├── train_all.py                  # Master training loop for all 4 models
│   ├── evaluate.py                   # Computes condition satisfaction metrics
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cgan.py                   # Model 1: Conditional WGAN-GP
│   │   ├── seq2seq_transformer.py    # Model 2: Encoder-Decoder Transformer
│   │   ├── autoregressive.py         # Model 3: Decoder-Only Transformer
│   │   └── cvae.py                   # Model 4: Conditional VAE
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tokenizer.py              # Vocabulary and encoding/decoding
│   │   ├── dataset.py                # Dataset loader and sampler
│   │   ├── metrics.py                # Condition satisfaction evaluator
│   │   └── date_validator.py         # Calendar-based validation
│   └── weights/                      # Saved model weights
│       ├── cgan_gen.pt
│       ├── seq2seq.pt
│       ├── autoregressive.pt
│       └── cvae.pt
├── environment.yml                   # Conda environment configuration
├── README.md                         # Project documentation
└── Assignment_1_YourName_YourID.pdf  # Final report
```

## Model Architectures

### 1. Conditional WGAN-GP (`cgan.py`)
A Wasserstein GAN with Gradient Penalty for stable training and reduced mode collapse. The generator uses Gumbel-Softmax to enable gradient flow through discrete token generation.

### 2. Seq2Seq Transformer (`seq2seq_transformer.py`)
An encoder-decoder Transformer where the condition tokens are encoded and the target date is generated autoregressively.

### 3. Autoregressive Transformer (`autoregressive.py`)
A decoder-only Transformer with causal masking. The input conditions act as a prompt for generating the date token-by-token.

### 4. Conditional VAE (`cvae.py`)
A Variational Autoencoder that learns a latent distribution over valid dates while conditioning on the input constraints.

## Setup and Installation

This project uses Conda for dependency and environment management.

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate date_gen
```

## Usage Guide

### Training

Train all four models sequentially using:

```bash
python model/train_all.py
```

The training pipeline:
- Splits the dataset into training, validation, and test sets
- Applies weighted sampling to address class imbalance
- Saves the best model weights in `model/weights/`

### Evaluation

Evaluate trained models using:

```bash
python model/evaluate.py
```

The evaluation script computes:
- Day condition accuracy
- Month condition accuracy
- Leap year accuracy
- Decade accuracy
- Valid date rate
- Full pass rate

### Inference

Run inference using:

```bash
python model/predict.py -i <input_file> -o <output_file>
```

Example:

```bash
python model/predict.py -i data/example_input.txt -o predictions.txt
```

### Input Format

```text
[WED] [JAN] [False] [180]
```

### Output Format

```text
[WED] [JAN] [False] [180] 1-1-1800
```

## Evaluation Metric

The primary evaluation metric is `full_pass_rate`.

A generated output passes evaluation only if:
- It forms a valid Gregorian calendar date
- The year is within the range `1800-2200`
- The generated date satisfies all four requested conditions

Validation is performed using Python's built-in `calendar` and `datetime` utilities.

## Dependencies

Main libraries used in this project:
- Python 3.10+
- PyTorch
- NumPy
- Pandas
- tqdm

All required dependencies are listed in `environment.yml`.

## Notes

- All models are implemented in PyTorch.
- Model checkpoints are stored in `model/weights/`.
- The repository follows a modular structure for easier experimentation and extension.