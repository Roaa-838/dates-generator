"""
evaluate.py — Evaluate all trained models on the test split.

Usage:
    python evaluate.py --data_path ../data/data.txt [--output_dir ./eval_results]

Applies the same day-of-week correction layer used in predict.py so that
evaluation scores reflect real inference performance (not raw model output).

Produces:
  - Console table: per-model condition satisfaction rates (raw + corrected)
  - eval_results/evaluation_report.csv
  - eval_results/example_outputs.txt
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.tokenizer import DateTokenizer
from utils.dataset import get_dataloaders
from utils.metrics import condition_satisfaction_rate
from utils.date_validator import parse_date_string
from predict import apply_day_correction   # reuse the correction layer

from models.cgan import ConditionalWGANGP
from models.seq2seq_transformer import Seq2SeqDateTransformer
from models.autoregressive import AutoregressiveDateTransformer
from models.cvae import ConditionalVAE

SEED: int = 42
WEIGHTS_DIR = Path(__file__).parent / "weights"


def set_seeds(seed: int = SEED) -> None:
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(name: str, device: torch.device) -> torch.nn.Module | None:
    """Load a trained model from checkpoint. Returns None if not found."""
    ckpt_map = {
        "cgan":           WEIGHTS_DIR / "cgan_gen.pt",
        "seq2seq":        WEIGHTS_DIR / "seq2seq.pt",
        "autoregressive": WEIGHTS_DIR / "autoregressive.pt",
        "cvae":           WEIGHTS_DIR / "cvae.pt",
    }
    model_map = {
        "cgan":           ConditionalWGANGP,
        "seq2seq":        Seq2SeqDateTransformer,
        "autoregressive": AutoregressiveDateTransformer,
        "cvae":           ConditionalVAE,
    }
    ckpt = ckpt_map.get(name)
    if ckpt is None or not ckpt.exists() or ckpt.stat().st_size == 0:
        print(f"  [skip] {name}: checkpoint not found or empty")
        return None
    try:
        model = model_map[name]()
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"  [skip] {name}: load error — {e}")
        return None


@torch.no_grad()
def run_model_eval(
    model       : torch.nn.Module,
    model_name  : str,
    test_loader : torch.utils.data.DataLoader,
    tokenizer   : DateTokenizer,
    device      : torch.device,
    n_examples  : int = 10,
) -> tuple[dict[str, float], dict[str, float], list[dict]]:
    """
    Evaluate a model on the test set.

    Returns both RAW scores (model output only) and CORRECTED scores
    (after the day-of-week correction layer) so both can be reported.

    Parameters
    ----------
    model, model_name, test_loader, tokenizer, device : standard
    n_examples : int   Number of example outputs to collect for the report.

    Returns
    -------
    raw_metrics       : dict[str, float]  before day correction
    corrected_metrics : dict[str, float]  after day correction
    examples          : list[dict]
    """
    raw_preds  : list[str]                       = []
    corr_preds : list[str]                       = []
    conditions : list[tuple[str, str, str, str]] = []
    examples   : list[dict]                      = []

    for batch in tqdm(test_loader, desc=f"Evaluating {model_name}", leave=False):
        day    = batch["day_id"].to(device)
        month  = batch["month_id"].to(device)
        leap   = batch["leap_id"].to(device)
        decade = batch["decade_id"].to(device)

        if model_name == "cgan":
            token_ids = model.generate(day, month, leap, decade, device=device)
        elif model_name in ("seq2seq", "autoregressive"):
            token_ids = model.generate(day, month, leap, decade)
        else:
            token_ids = model.generate(day, month, leap, decade, sample=True)

        for i, ids in enumerate(token_ids.cpu().tolist()):
            raw_date = tokenizer.decode_output(ids) or ""
            dc  = batch["day_cond"][i]
            mc  = batch["month_cond"][i]
            lc  = batch["leap_cond"][i]
            dec = batch["decade_cond"][i]

            # Apply day-of-week correction (same as predict.py)
            if raw_date:
                corrected = apply_day_correction(raw_date, dc, mc, lc, dec)
            else:
                corrected = ""

            raw_preds.append(raw_date)
            corr_preds.append(corrected)
            conditions.append((dc, mc, lc, dec))

            if len(examples) < n_examples:
                examples.append({
                    "model"       : model_name,
                    "condition"   : tokenizer.decode_input(
                        batch["day_id"][i].item(), batch["month_id"][i].item(),
                        batch["leap_id"][i].item(), batch["decade_id"][i].item(),
                    ),
                    "raw"         : raw_date,
                    "corrected"   : corrected,
                    "ground_truth": batch["date_str"][i],
                })

    raw_metrics  = condition_satisfaction_rate(raw_preds,  conditions)
    corr_metrics = condition_satisfaction_rate(corr_preds, conditions)
    return raw_metrics, corr_metrics, examples


def print_results_table(
    all_raw  : dict[str, dict[str, float]],
    all_corr : dict[str, dict[str, float]],
) -> None:
    """Print a side-by-side comparison: raw model vs corrected output."""
    cols = ["valid_date_rate", "day_pass_rate", "month_pass_rate",
            "leap_pass_rate", "decade_pass_rate", "full_pass_rate"]
    short = ["valid%", "day%", "month%", "leap%", "decade%", "FULL%"]

    w = 94
    print("\n" + "=" * w)
    print("TEST SET EVALUATION RESULTS  (raw model output | +day-correction)")
    print("=" * w)
    hdr = f"  {'Model':<16}" + "".join(f"  {s:>10}" for s in short)
    print(hdr + "   ||" + "".join(f"  {s:>10}" for s in short))
    print("-" * w)
    for name in all_raw:
        r = all_raw[name]
        c = all_corr[name]
        row_r = "".join(f"  {r.get(k,0)*100:>9.1f}%" for k in cols)
        row_c = "".join(f"  {c.get(k,0)*100:>9.1f}%" for k in cols)
        print(f"  {name:<16}{row_r}   ||{row_c}")
    print("=" * w)
    print("  Left columns = raw model output.  Right columns = after day-of-week correction.")


def save_csv(
    all_raw  : dict[str, dict[str, float]],
    all_corr : dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Save both raw and corrected evaluation results to CSV."""
    path = output_dir / "evaluation_report.csv"
    fields = ["model", "mode",
              "valid_date_rate", "day_pass_rate", "month_pass_rate",
              "leap_pass_rate", "decade_pass_rate", "full_pass_rate"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in all_raw:
            for mode, src in [("raw", all_raw[name]), ("corrected", all_corr[name])]:
                w.writerow({"model": name, "mode": mode,
                            **{k: f"{v:.4f}" for k, v in src.items()}})
    print(f"Evaluation CSV → {path}")


def save_examples(all_examples: list[dict], output_dir: Path) -> None:
    """Save example predictions (raw + corrected) to text file for the report."""
    path = output_dir / "example_outputs.txt"
    with open(path, "w") as f:
        for ex in all_examples:
            f.write(f"Model       : {ex['model']}\n")
            f.write(f"Condition   : {ex['condition']}\n")
            f.write(f"Raw output  : {ex['raw']}\n")
            f.write(f"Corrected   : {ex['corrected']}\n")
            f.write(f"Ground truth: {ex['ground_truth']}\n")
            f.write("\n")
    print(f"Examples    → {path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Evaluate all trained models.")
    p.add_argument("--data_path",  default="../data/data.txt")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    set_seeds(SEED)
    args       = parse_args()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = DateTokenizer()
    _, _, test_loader = get_dataloaders(
        args.data_path, tokenizer, batch_size=args.batch_size, seed=SEED
    )

    model_names = ["cgan", "seq2seq", "autoregressive", "cvae"]
    all_raw  : dict[str, dict[str, float]] = {}
    all_corr : dict[str, dict[str, float]] = {}
    all_examples: list[dict] = []

    for name in model_names:
        model = load_model(name, device)
        if model is None:
            continue
        raw_m, corr_m, examples = run_model_eval(
            model, name, test_loader, tokenizer, device
        )
        all_raw[name]  = raw_m
        all_corr[name] = corr_m
        all_examples.extend(examples)

    if all_raw:
        print_results_table(all_raw, all_corr)
        save_csv(all_raw, all_corr, output_dir)
        save_examples(all_examples, output_dir)
    else:
        print("No models evaluated. Run training first.")


if __name__ == "__main__":
    main()