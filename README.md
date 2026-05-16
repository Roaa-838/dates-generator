# Conditional Date Generation — Assignment 1

## Quick-Start: Exact Steps to Run Everything

Follow these steps **in order**. Each step is a single command.

---

### Step 0 — Clone & place data

```bash
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Place the provided files:
cp /path/to/data.txt          data/data.txt
cp /path/to/example_input.txt data/example_input.txt
```

---

### Step 1 — Create conda environment

```bash
conda env create -f environment.yml
conda activate date_gen
```

---

### Step 2 — Install MLflow (if not pulled by conda)

```bash
pip install mlflow>=2.9.0
```

---

### Step 3 — Train all models (with auto-resume)

```bash
cd model
python train_all.py --data_path ../data/data.txt
```

**If training is interrupted**, simply re-run the same command.  
The `--resume` flag is on by default. Training picks up from the last  
completed epoch for each model via `weights/<model>_checkpoint.pt`.

Train a single model only:
```bash
python train_all.py --data_path ../data/data.txt --models autoregressive
```

Override epoch counts:
```bash
python train_all.py --data_path ../data/data.txt \
    --epochs_gan 100 --epochs_seq 50 --epochs_ar 50 --epochs_cvae 50
```

Force restart from epoch 1 (ignore checkpoints):
```bash
python train_all.py --data_path ../data/data.txt --no_resume
```

---

### Step 4 — Monitor training with MLflow

In a **separate terminal**:
```bash
conda activate date_gen
mlflow ui --port 5000
```
Open **http://localhost:5000** in your browser.  
Every epoch's metrics are logged there in real time.

---

### Step 5 — Run inference (required assignment interface)

```bash
python model/predict.py \
    -i data/example_input.txt \
    -o data/predictions.txt
```

Output format (matches `data.txt` exactly):
```
[WED] [JAN] [False] [180] 1-1-1800
[MON] [JAN] [False] [190] 5-1-1901
...
```

---

### Step 6 — Evaluate all models on the test set

```bash
python model/evaluate.py --data_path data/data.txt --output_dir eval_results
```

Produces:
- `eval_results/evaluation_report.csv` — per-model condition satisfaction rates
- `eval_results/example_outputs.txt`   — sample predictions for the report

---

### Step 7 — Run compliance checker

```bash
python scripts/check_compliance.py
```

This verifies every assignment rule:
- Repo structure ✓
- predict.py CLI interface ✓
- Output format ✓
- Date range [1800–2200] ✓
- All 4 conditions satisfied ✓
- GAN implementation ✓
- Bonus best practices ✓

---

### Step 8 — Push to GitHub

**First time:**
```bash
chmod +x scripts/github_setup.sh
./scripts/github_setup.sh YOUR_GITHUB_USERNAME YOUR_REPO_NAME
```

**Subsequent pushes:**
```bash
./scripts/push.sh "your commit message"
```

---

## File Structure

```
repo/
├── data/
│   ├── data.txt                    ← full dataset (place here)
│   └── example_input.txt           ← assignment-provided input
├── model/
│   ├── predict.py                  ← REQUIRED: python predict.py -i ... -o ...
│   ├── train_all.py                ← trains all 4 models (with MLflow + resume)
│   ├── evaluate.py                 ← test-set evaluation
│   ├── models/
│   │   ├── cgan.py                 ← Model 1: Conditional WGAN-GP
│   │   ├── seq2seq_transformer.py  ← Model 2: Enc-Dec Transformer
│   │   ├── autoregressive.py       ← Model 3: Decoder-Only Transformer
│   │   └── cvae.py                 ← Model 4: Conditional VAE
│   ├── utils/
│   │   ├── tokenizer.py            ← custom year-first tokeniser
│   │   ├── date_validator.py       ← calendar.isleap() based validator + fallback
│   │   ├── dataset.py              ← Dataset, WeightedRandomSampler, DataLoaders
│   │   └── metrics.py              ← condition_satisfaction_rate (primary metric)
│   └── weights/                    ← checkpoints & best-model weights saved here
├── mlruns/                         ← MLflow tracking data (auto-created)
├── scripts/
│   ├── github_setup.sh             ← one-shot git init + push to GitHub
│   ├── push.sh                     ← quick commit-and-push
│   └── check_compliance.py         ← verifies all assignment rules
├── environment.yml
└── Assignment_1_YourName_YourID.md ← report
```

## Resume Logic

| File | Purpose |
|---|---|
| `weights/<model>_checkpoint.pt` | Full checkpoint saved **every epoch** (model + optimizer + scheduler + epoch + best_score) |
| `weights/<model>_run_id.txt` | MLflow run_id — lets resumed training append to the **same** MLflow run |
| `weights/<model>.pt` | Best model weights only (saved when `full_pass_rate` improves) |
| `weights/training_log.csv` | CSV copy of all epoch metrics (backup if MLflow unavailable) |

If training crashes at epoch 37 out of 50, re-running `train_all.py` will:
1. Load `weights/seq2seq_checkpoint.pt` → restore model, optimizer, scheduler
2. Read `weights/seq2seq_run_id.txt` → resume the same MLflow run
3. Continue from epoch 38, not epoch 1

## Primary Evaluation Metric

`full_pass_rate` = fraction of generated dates satisfying **all four** conditions simultaneously.

Accuracy is NOT used — multiple valid outputs exist per input.
```
