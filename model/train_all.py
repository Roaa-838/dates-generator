"""
train_all.py — Master training script with MLflow tracking + resume support.

Features
--------
* MLflow experiment tracking: every epoch logs train_loss, val metrics, and
  hyperparameters. Runs are identified by model name, a fixed run_id stored
  in weights/<model>_run_id.txt so disconnected training resumes the same run.
* Resume from checkpoint: full checkpoint (model + optimizer + scheduler +
  epoch + best_score) is saved every epoch so training continues exactly
  where it left off after any disconnection.
* Best-model saving by full_pass_rate (NOT last epoch).
* All random seeds fixed globally for reproducibility.
* Gradient clipping on all non-GAN models (max_norm=1.0).
* tqdm progress bars, device-agnostic (CUDA / CPU).

Usage
-----
    python train_all.py --data_path ../data/data.txt
    python train_all.py --data_path ../data/data.txt --models seq2seq autoregressive
    python train_all.py --data_path ../data/data.txt --resume      # auto-resume (default)
    python train_all.py --data_path ../data/data.txt --no_resume   # force restart
    mlflow ui --port 5000                                           # view dashboard
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not installed. Run:  pip install mlflow")
    print("         Training continues without MLflow tracking.\n")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.tokenizer import DateTokenizer
from utils.dataset import get_dataloaders
from utils.metrics import condition_satisfaction_rate, log_metrics_to_csv
from models.cgan import ConditionalWGANGP
from models.seq2seq_transformer import Seq2SeqDateTransformer, NoamScheduler
from models.autoregressive import AutoregressiveDateTransformer
from models.cvae import ConditionalVAE


# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

SEED        : int  = 42
WEIGHTS_DIR : Path = Path(__file__).parent / "weights"
LOG_CSV     : Path = WEIGHTS_DIR / "training_log.csv"
MLFLOW_URI  : str  = Path(__file__).parent.parent.joinpath("mlruns").as_uri()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = SEED) -> None:
    """Set all random seeds for full experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# MLflow helpers
# ─────────────────────────────────────────────────────────────────────────────

def mlflow_setup(experiment_name: str = "date_generation") -> None:
    """Configure MLflow tracking URI and experiment."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)


def get_saved_run_id(model_name: str) -> str | None:
    """
    Read a previously saved MLflow run_id from disk.
    Returns None if no run has been started yet.
    This is what allows resumed training to append to the same MLflow run
    instead of creating a new one after a disconnection.
    """
    p = WEIGHTS_DIR / f"{model_name}_run_id.txt"
    return p.read_text().strip() if p.exists() else None


def save_run_id(model_name: str, run_id: str) -> None:
    """Persist the MLflow run_id so the next resume can find the same run."""
    (WEIGHTS_DIR / f"{model_name}_run_id.txt").write_text(run_id)


def log_epoch_metrics(
    epoch       : int,
    train_loss  : float,
    val_loss    : float,
    metrics     : dict[str, float],
    extra       : dict | None = None,
) -> None:
    """Log one epoch's scalars to the currently active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_metrics(
        {
            "train_loss"      : train_loss,
            "val_loss"        : val_loss,
            "valid_date_rate" : metrics.get("valid_date_rate",  0.0),
            "day_pass_rate"   : metrics.get("day_pass_rate",    0.0),
            "month_pass_rate" : metrics.get("month_pass_rate",  0.0),
            "leap_pass_rate"  : metrics.get("leap_pass_rate",   0.0),
            "decade_pass_rate": metrics.get("decade_pass_rate", 0.0),
            "full_pass_rate"  : metrics.get("full_pass_rate",   0.0),
            **(extra or {}),
        },
        step=epoch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model      : nn.Module,
    path       : Path,
    epoch      : int,
    best_score : float,
    optimizer  : torch.optim.Optimizer | None = None,
    scheduler                                  = None,
) -> None:
    """
    Save a full training checkpoint every epoch.

    Format:
        { 'epoch', 'best_score', 'model', 'optimizer'?, 'scheduler'? }

    This is distinct from the best-model weights file so that the best model
    is never overwritten by a later (worse) checkpoint save.
    """
    ckpt: dict = {"epoch": epoch, "best_score": best_score,
                  "model": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        with contextlib.suppress(Exception):
            ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    model      : nn.Module,
    path       : Path,
    device     : torch.device,
    optimizer  : torch.optim.Optimizer | None = None,
    scheduler                                  = None,
) -> tuple[int, float]:
    """
    Load a full checkpoint; return (start_epoch, best_score).

    If the file does not exist training starts from epoch 1.
    Also handles bare state-dicts saved by older code versions.
    """
    if not path.exists():
        return 1, -1.0

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer  and "optimizer"  in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler  and "scheduler"  in ckpt:
            with contextlib.suppress(Exception):
                scheduler.load_state_dict(ckpt["scheduler"])
        start  = ckpt.get("epoch", 0) + 1
        best   = ckpt.get("best_score", -1.0)
    else:                                  # bare state-dict fallback
        model.load_state_dict(ckpt)
        start, best = 1, -1.0

    print(f"  [Resume] epoch {start}, best_score={best:.4f}")
    return start, best


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_onehot(ids: Tensor, vocab: int = 14) -> Tensor:
    """LongTensor (B, T) → FloatTensor (B, T, vocab) one-hot."""
    return F.one_hot(ids.clamp(0, vocab - 1), num_classes=vocab).float()


@torch.no_grad()
def evaluate(
    model      : nn.Module,
    name       : str,
    loader     : torch.utils.data.DataLoader,
    tokenizer  : DateTokenizer,
    device     : torch.device,
) -> dict[str, float]:
    """Inference over a DataLoader → condition_satisfaction_rate dict."""
    model.eval()
    preds  : list[str]                       = []
    conds  : list[tuple[str, str, str, str]] = []

    for batch in loader:
        day    = batch["day_id"].to(device)
        month  = batch["month_id"].to(device)
        leap   = batch["leap_id"].to(device)
        decade = batch["decade_id"].to(device)

        if name == "cgan":
            ids = model.generate(day, month, leap, decade, device=device)
        elif name in ("seq2seq", "autoregressive"):
            ids = model.generate(day, month, leap, decade)
        else:
            ids = model.generate(day, month, leap, decade, sample=True)

        for i, tok in enumerate(ids.cpu().tolist()):
            preds.append(tokenizer.decode_output(tok) or "")
            conds.append((batch["day_cond"][i], batch["month_cond"][i],
                          batch["leap_cond"][i], batch["decade_cond"][i]))

    return condition_satisfaction_rate(preds, conds)


def _print_row(tag: str, epoch: int, total: int, loss: float,
               m: dict[str, float], is_best: bool) -> None:
    """Print a compact per-epoch summary line."""
    print(
        f"  {tag} {epoch:3d}/{total} | loss={loss:.4f} | "
        f"full={m['full_pass_rate']:.3f} "
        f"day={m['day_pass_rate']:.3f} "
        f"mon={m['month_pass_rate']:.3f} "
        f"leap={m['leap_pass_rate']:.3f} "
        f"dec={m['decade_pass_rate']:.3f}"
        + (" ★" if is_best else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Conditional WGAN-GP
# ─────────────────────────────────────────────────────────────────────────────

def train_cgan(
    train_loader : torch.utils.data.DataLoader,
    val_loader   : torch.utils.data.DataLoader,
    tokenizer    : DateTokenizer,
    device       : torch.device,
    epochs       : int   = 100,
    latent_dim   : int   = 100,
    n_critic     : int   = 5,
    lambda_gp    : float = 10.0,
    resume       : bool  = True,
) -> ConditionalWGANGP:
    """Train Conditional WGAN-GP with MLflow tracking and checkpoint resume."""
    print("\n" + "=" * 62)
    print("Model 1 — Conditional WGAN-GP")
    print("=" * 62)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    ckpt      = WEIGHTS_DIR / "cgan_checkpoint.pt"
    best_path = WEIGHTS_DIR / "cgan_gen.pt"

    model = ConditionalWGANGP(latent_dim=latent_dim).to(device)
    opt_G = Adam(model.generator.parameters(),     lr=1e-4, betas=(0.0, 0.9))
    opt_D = Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

    start, best = (load_checkpoint(model, ckpt, device, opt_G)
                   if resume else (1, -1.0))
    run_id = get_saved_run_id("cgan") if resume else None

    ctx = (mlflow.start_run(run_id=run_id, run_name="cgan")
           if MLFLOW_AVAILABLE else contextlib.nullcontext())

    with ctx as run:
        if MLFLOW_AVAILABLE:
            save_run_id("cgan", run.info.run_id)
            if start == 1:
                mlflow.log_params({"model": "cgan", "epochs": epochs,
                                   "latent_dim": latent_dim, "n_critic": n_critic,
                                   "lambda_gp": lambda_gp, "seed": SEED})

        for epoch in range(start, epochs + 1):
            model.train()
            tot_g = tot_d = 0.0
            n = 0
            pbar = tqdm(train_loader, desc=f"WGAN-GP {epoch}/{epochs}", leave=False)
            for batch in pbar:
                day    = batch["day_id"].to(device)
                month  = batch["month_id"].to(device)
                leap   = batch["leap_id"].to(device)
                decade = batch["decade_id"].to(device)
                real   = to_onehot(batch["target"].to(device))
                B      = day.size(0)
                cond   = model.embedder(day, month, leap, decade)

                # Critic steps
                for _ in range(n_critic):
                    z      = torch.randn(B, latent_dim, device=device)
                    d_loss = model.critic_loss(real, cond.detach(), z, lambda_gp)
                    opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                tot_d += d_loss.item()

                # Generator step
                z      = torch.randn(B, latent_dim, device=device)
                g_loss = model.generator_loss(cond, z)
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()
                tot_g += g_loss.item()
                model.anneal_temperature()
                n += 1
                pbar.set_postfix(G=f"{g_loss.item():.3f}",
                                 D=f"{d_loss.item():.3f}",
                                 τ=f"{model.generator.tau:.3f}")

            avg_g  = tot_g / max(n, 1)
            avg_d  = tot_d / max(n, 1)
            m      = evaluate(model, "cgan", val_loader, tokenizer, device)
            is_best = m["full_pass_rate"] > best
            if is_best:
                best = m["full_pass_rate"]
                torch.save(model.state_dict(), best_path)

            save_checkpoint(model, ckpt, epoch, best, opt_G)
            log_epoch_metrics(epoch, avg_g, avg_d, m,
                              {"g_loss": avg_g, "d_loss": avg_d,
                               "tau": model.generator.tau})
            log_metrics_to_csv(LOG_CSV, epoch, avg_g, avg_d, m, "cgan")
            _print_row("WGAN-GP", epoch, epochs, avg_g, m, is_best)

    print(f"  Best WGAN-GP  full_pass_rate: {best:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Seq2Seq Transformer
# ─────────────────────────────────────────────────────────────────────────────

def train_seq2seq(
    train_loader : torch.utils.data.DataLoader,
    val_loader   : torch.utils.data.DataLoader,
    tokenizer    : DateTokenizer,
    device       : torch.device,
    epochs       : int  = 50,
    resume       : bool = True,
) -> Seq2SeqDateTransformer:
    """Train Encoder-Decoder Transformer with MLflow + resume."""
    print("\n" + "=" * 62)
    print("Model 2 — Encoder-Decoder Seq2Seq Transformer")
    print("=" * 62)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    ckpt      = WEIGHTS_DIR / "seq2seq_checkpoint.pt"
    best_path = WEIGHTS_DIR / "seq2seq.pt"

    model     = Seq2SeqDateTransformer().to(device)
    optimizer = Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=400)

    start, best = (load_checkpoint(model, ckpt, device, optimizer)
                   if resume else (1, -1.0))
    run_id = get_saved_run_id("seq2seq") if resume else None

    ctx = (mlflow.start_run(run_id=run_id, run_name="seq2seq")
           if MLFLOW_AVAILABLE else contextlib.nullcontext())

    with ctx as run:
        if MLFLOW_AVAILABLE:
            save_run_id("seq2seq", run.info.run_id)
            if start == 1:
                mlflow.log_params({"model": "seq2seq", "epochs": epochs,
                                   "d_model": 128, "nhead": 4,
                                   "label_smoothing": 0.1, "seed": SEED})

        for epoch in range(start, epochs + 1):
            model.train()
            tot = 0.0; n = 0
            pbar = tqdm(train_loader, desc=f"Seq2Seq {epoch}/{epochs}", leave=False)
            for batch in pbar:
                loss = model.compute_loss(
                    batch["day_id"].to(device), batch["month_id"].to(device),
                    batch["leap_id"].to(device), batch["decade_id"].to(device),
                    batch["target"].to(device),
                )
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step(); optimizer.step()
                tot += loss.item(); n += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg     = tot / max(n, 1)
            m       = evaluate(model, "seq2seq", val_loader, tokenizer, device)
            is_best = m["full_pass_rate"] > best
            if is_best:
                best = m["full_pass_rate"]
                torch.save(model.state_dict(), best_path)

            save_checkpoint(model, ckpt, epoch, best, optimizer)
            log_epoch_metrics(epoch, avg, 0.0, m)
            log_metrics_to_csv(LOG_CSV, epoch, avg, 0.0, m, "seq2seq")
            _print_row("Seq2Seq", epoch, epochs, avg, m, is_best)

    print(f"  Best Seq2Seq  full_pass_rate: {best:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Autoregressive Transformer
# ─────────────────────────────────────────────────────────────────────────────

def train_autoregressive(
    train_loader : torch.utils.data.DataLoader,
    val_loader   : torch.utils.data.DataLoader,
    tokenizer    : DateTokenizer,
    device       : torch.device,
    epochs       : int  = 50,
    resume       : bool = True,
) -> AutoregressiveDateTransformer:
    """Train Decoder-Only Autoregressive Transformer with MLflow + resume."""
    print("\n" + "=" * 62)
    print("Model 3 — Decoder-Only Autoregressive Transformer")
    print("=" * 62)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    ckpt      = WEIGHTS_DIR / "autoregressive_checkpoint.pt"
    best_path = WEIGHTS_DIR / "autoregressive.pt"

    model     = AutoregressiveDateTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start, best = (load_checkpoint(model, ckpt, device, optimizer, scheduler)
                   if resume else (1, -1.0))
    run_id = get_saved_run_id("autoregressive") if resume else None

    ctx = (mlflow.start_run(run_id=run_id, run_name="autoregressive")
           if MLFLOW_AVAILABLE else contextlib.nullcontext())

    with ctx as run:
        if MLFLOW_AVAILABLE:
            save_run_id("autoregressive", run.info.run_id)
            if start == 1:
                mlflow.log_params({"model": "autoregressive", "epochs": epochs,
                                   "d_model": 128, "num_layers": 4,
                                   "lr": 3e-4, "seed": SEED})

        for epoch in range(start, epochs + 1):
            model.train()
            tot = 0.0; n = 0
            pbar = tqdm(train_loader, desc=f"AR-Trans {epoch}/{epochs}", leave=False)
            for batch in pbar:
                loss = model.compute_loss(
                    batch["day_id"].to(device), batch["month_id"].to(device),
                    batch["leap_id"].to(device), batch["decade_id"].to(device),
                    batch["target"].to(device),
                )
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot += loss.item(); n += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg     = tot / max(n, 1)
            m       = evaluate(model, "autoregressive", val_loader, tokenizer, device)
            is_best = m["full_pass_rate"] > best
            if is_best:
                best = m["full_pass_rate"]
                torch.save(model.state_dict(), best_path)

            save_checkpoint(model, ckpt, epoch, best, optimizer, scheduler)
            log_epoch_metrics(epoch, avg, 0.0, m)
            log_metrics_to_csv(LOG_CSV, epoch, avg, 0.0, m, "autoregressive")
            _print_row("AR-Trans", epoch, epochs, avg, m, is_best)

    print(f"  Best AR-Trans full_pass_rate: {best:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — Conditional VAE
# ─────────────────────────────────────────────────────────────────────────────

def train_cvae(
    train_loader     : torch.utils.data.DataLoader,
    val_loader       : torch.utils.data.DataLoader,
    tokenizer        : DateTokenizer,
    device           : torch.device,
    epochs           : int   = 50,
    kl_anneal_epochs : int   = 10,
    beta_max         : float = 1.0,
    resume           : bool  = True,
) -> ConditionalVAE:
    """Train Conditional VAE with KL annealing, MLflow + resume."""
    print("\n" + "=" * 62)
    print("Model 4 — Conditional VAE (CVAE)")
    print("=" * 62)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    ckpt      = WEIGHTS_DIR / "cvae_checkpoint.pt"
    best_path = WEIGHTS_DIR / "cvae.pt"

    model     = ConditionalVAE().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start, best = (load_checkpoint(model, ckpt, device, optimizer, scheduler)
                   if resume else (1, -1.0))
    # Restore beta annealing position after resume
    model.beta = min(beta_max, beta_max * (start - 1) / kl_anneal_epochs)
    run_id = get_saved_run_id("cvae") if resume else None

    ctx = (mlflow.start_run(run_id=run_id, run_name="cvae")
           if MLFLOW_AVAILABLE else contextlib.nullcontext())

    with ctx as run:
        if MLFLOW_AVAILABLE:
            save_run_id("cvae", run.info.run_id)
            if start == 1:
                mlflow.log_params({"model": "cvae", "epochs": epochs,
                                   "latent_dim": 64, "beta_max": beta_max,
                                   "kl_anneal_epochs": kl_anneal_epochs,
                                   "lr": 3e-4, "seed": SEED})

        for epoch in range(start, epochs + 1):
            beta       = min(beta_max, beta_max * epoch / kl_anneal_epochs)
            model.beta = beta
            model.train()
            tot = tot_r = tot_kl = 0.0; n = 0
            pbar = tqdm(train_loader, desc=f"CVAE {epoch}/{epochs}", leave=False)
            for batch in pbar:
                loss, recon, kl = model.compute_loss(
                    batch["day_id"].to(device), batch["month_id"].to(device),
                    batch["leap_id"].to(device), batch["decade_id"].to(device),
                    batch["target"].to(device),
                )
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot += loss.item(); tot_r += recon.item(); tot_kl += kl.item()
                n += 1
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 recon=f"{recon.item():.3f}",
                                 kl=f"{kl.item():.3f}", β=f"{beta:.2f}")

            scheduler.step()
            avg   = tot    / max(n, 1)
            avg_r = tot_r  / max(n, 1)
            avg_k = tot_kl / max(n, 1)
            m       = evaluate(model, "cvae", val_loader, tokenizer, device)
            is_best = m["full_pass_rate"] > best
            if is_best:
                best = m["full_pass_rate"]
                torch.save(model.state_dict(), best_path)

            save_checkpoint(model, ckpt, epoch, best, optimizer, scheduler)
            log_epoch_metrics(epoch, avg, 0.0, m,
                              {"recon": avg_r, "kl": avg_k, "beta": beta})
            log_metrics_to_csv(LOG_CSV, epoch, avg, 0.0, m, "cvae")
            _print_row("CVAE    ", epoch, epochs, avg, m, is_best)

    print(f"  Best CVAE     full_pass_rate: {best:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train conditional date generation models.")
    p.add_argument("--data_path",   default="../data/data.txt")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--epochs_gan",  type=int, default=100)
    p.add_argument("--epochs_seq",  type=int, default=50)
    p.add_argument("--epochs_ar",   type=int, default=50)
    p.add_argument("--epochs_cvae", type=int, default=50)
    p.add_argument(
        "--models", nargs="+",
        choices=["cgan", "seq2seq", "autoregressive", "cvae", "all"],
        default=["all"],
    )
    p.add_argument("--resume",    dest="resume", action="store_true",  default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false",
                   help="Force restart from epoch 1 (ignore checkpoints)")
    return p.parse_args()


def main() -> None:
    """Main training entry point."""
    set_seeds(SEED)
    args   = parse_args()
    device = get_device()
    WEIGHTS_DIR.mkdir(exist_ok=True)
    mlflow_setup()

    tokenizer = DateTokenizer()
    train_loader, val_loader, _ = get_dataloaders(
        args.data_path, tokenizer, batch_size=args.batch_size, seed=SEED
    )

    to_train = set(args.models)
    if "all" in to_train:
        to_train = {"cgan", "seq2seq", "autoregressive", "cvae"}

    t0 = time.time()
    if "cgan"           in to_train:
        train_cgan(train_loader, val_loader, tokenizer, device,
                   epochs=args.epochs_gan, resume=args.resume)
    if "seq2seq"        in to_train:
        train_seq2seq(train_loader, val_loader, tokenizer, device,
                      epochs=args.epochs_seq, resume=args.resume)
    if "autoregressive" in to_train:
        train_autoregressive(train_loader, val_loader, tokenizer, device,
                             epochs=args.epochs_ar, resume=args.resume)
    if "cvae"           in to_train:
        train_cvae(train_loader, val_loader, tokenizer, device,
                   epochs=args.epochs_cvae, resume=args.resume)

    print(f"\n✅ Done in {(time.time()-t0)/60:.1f} min")
    print(f"   Weights  → {WEIGHTS_DIR}")
    print(f"   CSV log  → {LOG_CSV}")
    if MLFLOW_AVAILABLE:
        print(f"   MLflow   → mlflow ui --port 5000  (tracking at {MLFLOW_URI})")


if __name__ == "__main__":
    main()