"""
PyTorch Dataset and DataLoader utilities for conditional date generation.

Handles:
  - Parsing data.txt
  - Train / validation / test splitting (85 / 10 / 5) with shuffle + seed
  - WeightedRandomSampler to mitigate leap-year imbalance
  - Padding output sequences to fixed length for batching
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

from .tokenizer import DateTokenizer, PAD_ID, MAX_OUTPUT_LEN
from .date_validator import parse_date_string


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DateDataset(Dataset):
    """
    PyTorch Dataset for conditional date generation.

    Each item is a dict:
      {
        'day_id'    : int tensor (0-6)
        'month_id'  : int tensor (0-11)
        'leap_id'   : int tensor (0-1)
        'decade_id' : int tensor (0-40)
        'target'    : LongTensor of shape (MAX_OUTPUT_LEN,)  — padded with PAD_ID
        'target_len': int  — actual sequence length (including SOS & EOS)
      }

    Parameters
    ----------
    lines : list[str]
        Pre-split lines from data.txt (already shuffled & split by caller).
    tokenizer : DateTokenizer
    """

    def __init__(self, lines: list[str], tokenizer: DateTokenizer) -> None:
        self.tokenizer = tokenizer
        self.samples: list[dict] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                day_id, month_id, leap_id, decade_id = tokenizer.encode_input(line)
                # Extract the date part (last token after the 4 bracketed conditions)
                date_part = line.split("]")[-1].strip()
                if not date_part:
                    continue
                token_ids = tokenizer.encode_output(date_part)
                self.samples.append({
                    "day_id": day_id,
                    "month_id": month_id,
                    "leap_id": leap_id,
                    "decade_id": decade_id,
                    "token_ids": token_ids,
                    # Store raw strings for condition-satisfaction evaluation
                    "day_cond": line.split("]")[0].lstrip("[").strip()[:3],
                    "month_cond": line.split("]")[1].lstrip(" [").strip()[:3],
                    "leap_cond": line.split("]")[2].lstrip(" [").strip(),
                    "decade_cond": line.split("]")[3].lstrip(" [").strip(),
                    "date_str": date_part,
                })
            except (ValueError, IndexError):
                continue  # skip malformed lines

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        token_ids: list[int] = s["token_ids"]
        seq_len: int = len(token_ids)

        # Pad to MAX_OUTPUT_LEN
        padded = token_ids + [PAD_ID] * (MAX_OUTPUT_LEN - seq_len)
        target = torch.tensor(padded, dtype=torch.long)

        return {
            "day_id":     torch.tensor(s["day_id"],     dtype=torch.long),
            "month_id":   torch.tensor(s["month_id"],   dtype=torch.long),
            "leap_id":    torch.tensor(s["leap_id"],    dtype=torch.long),
            "decade_id":  torch.tensor(s["decade_id"],  dtype=torch.long),
            "target":     target,
            "target_len": torch.tensor(seq_len, dtype=torch.long),
            # Strings kept for metric computation (not fed to GPU)
            "day_cond":    s["day_cond"],
            "month_cond":  s["month_cond"],
            "leap_cond":   s["leap_cond"],
            "decade_cond": s["decade_cond"],
            "date_str":    s["date_str"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────────

def get_weighted_sampler(dataset: DateDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that over-samples minority conditions.

    Weight for each sample = 1 / (count of its leap_id class).
    Leap years (~24% of all years) are the primary imbalance factor.
    Secondary: edge decades (180, 220) have fewer total dates.

    Parameters
    ----------
    dataset : DateDataset

    Returns
    -------
    WeightedRandomSampler
        replacement=True so minority classes are seen more often.
    """
    leap_ids = np.array([s["leap_id"] for s in dataset.samples])
    decade_ids = np.array([s["decade_id"] for s in dataset.samples])

    # Count per leap class
    leap_counts = np.bincount(leap_ids)
    leap_weights = 1.0 / leap_counts[leap_ids]  # higher weight for rarer class

    # Mild extra weight for edge decades (indices 0=180 and 40=220)
    edge_mask = (decade_ids == 0) | (decade_ids == 40)
    decade_weights = np.where(edge_mask, 2.0, 1.0)

    combined = leap_weights * decade_weights
    # Normalise to [0, 1] range — PyTorch will scale internally but cleaner
    combined = combined / combined.sum() * len(combined)

    sample_weights = torch.from_numpy(combined).float()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    filepath: str | Path,
    tokenizer: DateTokenizer,
    batch_size: int = 256,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Parse data.txt and return (train_loader, val_loader, test_loader).

    Split ratios:  85% train / 10% val / 5% test.
    Data is shuffled with the given seed BEFORE splitting.
    Train loader uses WeightedRandomSampler for leap-year balance.
    Val / test loaders use sequential ordering.

    Parameters
    ----------
    filepath : str or Path
    tokenizer : DateTokenizer
    batch_size : int
    train_ratio, val_ratio : float
        Must sum to ≤ 1.0; remainder becomes test set.
    seed : int
        For reproducibility (shuffling before split).
    num_workers : int

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
    """
    filepath = Path(filepath)
    lines = filepath.read_text().splitlines()

    # Shuffle with seed before splitting
    rng = random.Random(seed)
    rng.shuffle(lines)

    n = len(lines)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_lines = lines[:n_train]
    val_lines = lines[n_train: n_train + n_val]
    test_lines = lines[n_train + n_val:]

    train_ds = DateDataset(train_lines, tokenizer)
    val_ds = DateDataset(val_lines, tokenizer)
    test_ds = DateDataset(test_lines, tokenizer)

    sampler = get_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Dataset split — train: {len(train_ds):,}  "
        f"val: {len(val_ds):,}  test: {len(test_ds):,}"
    )
    return train_loader, val_loader, test_loader