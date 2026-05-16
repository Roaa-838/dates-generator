"""
Evaluation metrics for conditional date generation.

THE PRIMARY METRIC is condition_satisfaction_rate — NOT accuracy.
Multiple valid dates can satisfy the same conditions, so accuracy would
incorrectly penalise valid-but-different outputs.

For each generated date we check the four conditions independently:
  day  → does the date fall on the required day-of-week?
  month → does the date occur in the required month?
  leap  → is the year a (non-)leap year as required?
  decade → does the year fall in the required decade?

The overall metric is full_pass_rate: all four conditions satisfied simultaneously.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .date_validator import check_conditions, parse_date_string


def condition_satisfaction_rate(
    predictions: list[str],
    conditions: list[tuple[str, str, str, str]],
) -> dict[str, float]:
    """
    Compute per-condition and full-pass satisfaction rates over a batch.

    Parameters
    ----------
    predictions : list[str]
        Generated date strings, e.g. ['3-12-1962', '10-1-1810', ...].
        May include None or malformed strings (counted as failures).
    conditions : list[tuple[str, str, str, str]]
        Each element: (day_cond, month_cond, leap_cond, decade_cond).
        e.g. ('WED', 'DEC', 'False', '196').

    Returns
    -------
    dict[str, float]
        Keys:
          'day_pass_rate'    – fraction where generated day-of-week matches
          'month_pass_rate'  – fraction where month matches
          'leap_pass_rate'   – fraction where leap-year status matches
          'decade_pass_rate' – fraction where decade matches
          'valid_date_rate'  – fraction that are parseable valid dates
          'full_pass_rate'   – fraction where ALL 4 conditions pass
    """
    assert len(predictions) == len(conditions), (
        f"predictions ({len(predictions)}) and conditions ({len(conditions)}) must be same length"
    )

    counters: dict[str, int] = {
        "valid": 0, "day": 0, "month": 0, "leap": 0, "decade": 0, "all_pass": 0,
    }
    n = len(predictions)

    for pred, (day_c, month_c, leap_c, decade_c) in zip(predictions, conditions):
        if pred is None:
            continue
        parsed = parse_date_string(pred)
        if parsed is None:
            continue
        d, m, y = parsed
        result = check_conditions(d, m, y, day_c, month_c, leap_c, decade_c)
        for key in counters:
            if result.get(key, False):
                counters[key] += 1

    denom = max(n, 1)
    return {
        "valid_date_rate":   counters["valid"]    / denom,
        "day_pass_rate":     counters["day"]      / denom,
        "month_pass_rate":   counters["month"]    / denom,
        "leap_pass_rate":    counters["leap"]      / denom,
        "decade_pass_rate":  counters["decade"]   / denom,
        "full_pass_rate":    counters["all_pass"] / denom,
    }


def log_metrics_to_csv(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, float],
    model_name: str = "",
) -> None:
    """
    Append one row of training metrics to a CSV file (creates it if absent).

    Parameters
    ----------
    log_path : Path
        Path to the CSV log file.
    epoch : int
    train_loss : float
    val_loss : float
    metrics : dict[str, float]
        Output of condition_satisfaction_rate().
    model_name : str
        Optional label to distinguish models when multiple share one log.
    """
    fieldnames = [
        "model", "epoch", "train_loss", "val_loss",
        "valid_date_rate", "day_pass_rate", "month_pass_rate",
        "leap_pass_rate", "decade_pass_rate", "full_pass_rate",
    ]
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model":            model_name,
            "epoch":            epoch,
            "train_loss":       f"{train_loss:.6f}",
            "val_loss":         f"{val_loss:.6f}",
            "valid_date_rate":  f"{metrics.get('valid_date_rate', 0):.4f}",
            "day_pass_rate":    f"{metrics.get('day_pass_rate', 0):.4f}",
            "month_pass_rate":  f"{metrics.get('month_pass_rate', 0):.4f}",
            "leap_pass_rate":   f"{metrics.get('leap_pass_rate', 0):.4f}",
            "decade_pass_rate": f"{metrics.get('decade_pass_rate', 0):.4f}",
            "full_pass_rate":   f"{metrics.get('full_pass_rate', 0):.4f}",
        })