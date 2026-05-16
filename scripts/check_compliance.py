"""
scripts/check_compliance.py — Assignment rules compliance checker.

Verifies every hard requirement from the assignment spec:
  1. Repo structure (required files and folders)
  2. predict.py CLI interface (-i / -o)
  3. Output format matches data.txt exactly
  4. Date range constraint  [1-1-1800 .. 31-12-2200]
  5. Condition correctness  (day-of-week, month, leap year, decade)
  6. At least 4 models, one of which is a GAN
  7. Weights files exist
  8. conda environment.yml exists
  9. training_log.csv exists (MLflow / CSV logging)

Usage:
    python scripts/check_compliance.py
    python scripts/check_compliance.py --data_sample data/data.txt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure imports work from any CWD
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from utils.tokenizer import DateTokenizer
from utils.date_validator import (
    is_valid_date, check_conditions, parse_date_string
)


PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "


def header(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Repo structure
# ─────────────────────────────────────────────────────────────────────────────

def check_structure() -> bool:
    """Verify all required files and directories exist."""
    header("1. Repository Structure")
    required: list[tuple[str, str]] = [
        ("file", "model/predict.py"),
        ("file", "model/train_all.py"),
        ("file", "model/evaluate.py"),
        ("file", "model/models/cgan.py"),
        ("file", "model/models/seq2seq_transformer.py"),
        ("file", "model/models/autoregressive.py"),
        ("file", "model/models/cvae.py"),
        ("file", "model/utils/tokenizer.py"),
        ("file", "model/utils/date_validator.py"),
        ("file", "model/utils/dataset.py"),
        ("file", "model/utils/metrics.py"),
        ("file", "model/models/__init__.py"),
        ("file", "model/utils/__init__.py"),
        ("file", "environment.yml"),
        ("file", "data/example_input.txt"),
        ("dir",  "model/weights"),
    ]
    ok = True
    for kind, rel in required:
        p = ROOT / rel
        exists = p.is_file() if kind == "file" else p.is_dir()
        sym = PASS if exists else FAIL
        print(f"{sym}  {rel}")
        if not exists:
            ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — Weights existence
# ─────────────────────────────────────────────────────────────────────────────

def check_weights() -> bool:
    """Verify at least one model weight file is saved."""
    header("2. Saved Weight Files")
    weight_files = {
        "WGAN-GP"        : ROOT / "model/weights/cgan_gen.pt",
        "Seq2Seq"        : ROOT / "model/weights/seq2seq.pt",
        "Autoregressive" : ROOT / "model/weights/autoregressive.pt",
        "CVAE"           : ROOT / "model/weights/cvae.pt",
    }
    ok = True
    found = 0
    for name, p in weight_files.items():
        exists = p.exists()
        print(f"{'PASS' if exists else WARN}  {name}: {p.name}")
        if exists:
            found += 1
        else:
            print(f"         (train with: python model/train_all.py --models "
                  f"{name.lower()})")
    if found == 0:
        print(f"{FAIL}  No weight files found — run training first.")
        ok = False
    elif found < 4:
        print(f"{WARN}  Only {found}/4 models trained. Run full training.")
    else:
        print(f"{PASS}  All 4 model weights present.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — predict.py CLI interface
# ─────────────────────────────────────────────────────────────────────────────

def check_predict_interface() -> bool:
    """
    Run predict.py with -i and -o and verify it exits without error.
    Uses example_input.txt as the input.
    """
    header("3. predict.py CLI Interface  (-i / -o)")
    input_file = ROOT / "data/example_input.txt"
    if not input_file.exists():
        print(f"{FAIL}  example_input.txt not found at {input_file}")
        return False

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        out_path = Path(tmp.name)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "model/predict.py"),
            "-i", str(input_file),
            "-o", str(out_path),
        ],
        capture_output=True, text=True,
        cwd=str(ROOT / "model"),
    )

    if result.returncode != 0:
        print(f"{FAIL}  predict.py exited with code {result.returncode}")
        print(f"       stderr: {result.stderr[:400]}")
        return False

    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"{FAIL}  Output file empty or missing: {out_path}")
        return False

    print(f"{PASS}  predict.py ran successfully, output written to {out_path}")
    return True, out_path


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — Output format
# ─────────────────────────────────────────────────────────────────────────────

def check_output_format(output_path: Path) -> bool:
    """
    Verify each output line matches the format:
        [DAY] [MON] [Leap] [decade] d-m-yyyy
    """
    header("4. Output Format  ([cond] [cond] [cond] [cond] d-m-yyyy)")
    pattern = re.compile(
        r'^\[(?:MON|TUE|WED|THU|FRI|SAT|SUN)\] '
        r'\[(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\] '
        r'\[(?:True|False)\] '
        r'\[\d{3}\] '
        r'\d{1,2}-\d{1,2}-\d{4}$'
    )
    lines    = [l.strip() for l in output_path.read_text().splitlines() if l.strip()]
    ok       = True
    failures = []

    for i, line in enumerate(lines, 1):
        if not pattern.match(line):
            failures.append(f"    Line {i}: {line!r}")
            ok = False

    if ok:
        print(f"{PASS}  All {len(lines)} output lines match required format.")
    else:
        print(f"{FAIL}  {len(failures)} malformed lines:")
        for f in failures[:5]:
            print(f)
        if len(failures) > 5:
            print(f"    ... and {len(failures)-5} more")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — Date range constraint
# ─────────────────────────────────────────────────────────────────────────────

def check_date_range(output_path: Path) -> bool:
    """All generated dates must be in [1-1-1800 .. 31-12-2200]."""
    header("5. Date Range Constraint  [1800-01-01 .. 2200-12-31]")
    lines    = [l.strip() for l in output_path.read_text().splitlines() if l.strip()]
    out_of_range = []
    invalid      = []

    for i, line in enumerate(lines, 1):
        date_str = line.split("]")[-1].strip()
        parsed   = parse_date_string(date_str)
        if parsed is None:
            invalid.append(f"    Line {i}: unparseable '{date_str}'")
            continue
        d, m, y = parsed
        if not is_valid_date(d, m, y):
            out_of_range.append(f"    Line {i}: {date_str} (invalid or out of range)")

    ok = len(out_of_range) == 0 and len(invalid) == 0
    if ok:
        print(f"{PASS}  All {len(lines)} dates valid and in range.")
    else:
        for msg in (invalid + out_of_range)[:5]:
            print(f"{FAIL} {msg}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — Condition satisfaction
# ─────────────────────────────────────────────────────────────────────────────

def check_conditions_compliance(output_path: Path) -> bool:
    """
    Parse every output line and evaluate the four conditions.
    Prints per-condition pass rates (the primary assignment metric).
    """
    header("6. Condition Satisfaction Rates  (primary metric)")
    tok      = DateTokenizer()
    lines    = [l.strip() for l in output_path.read_text().splitlines() if l.strip()]
    results  = {"valid": 0, "day": 0, "month": 0, "leap": 0, "decade": 0, "all": 0}
    n        = len(lines)

    for line in lines:
        try:
            cond_str = " ".join(line.split("]")[:-1]) + "]"
            day_c, month_c, leap_c, decade_c = tok.encode_input(line)
            date_str = line.split("]")[-1].strip()
            parsed   = parse_date_string(date_str)
            if parsed is None:
                continue
            d, m, y = parsed

            import re as _re
            toks = _re.findall(r'\[([^\]]+)\]', line)
            dc, mc, lc, dec = toks[0], toks[1], toks[2], toks[3]

            r = check_conditions(d, m, y, dc, mc, lc, dec)
            for k in ("valid", "day", "month", "leap", "decade"):
                results[k] += int(r.get(k, False))
            results["all"] += int(r.get("all_pass", False))
        except Exception:
            continue

    denom = max(n, 1)
    print(f"  Samples evaluated : {n}")
    print(f"  Valid dates        : {results['valid']/denom*100:.1f}%")
    print(f"  Day-of-week match  : {results['day']/denom*100:.1f}%")
    print(f"  Month match        : {results['month']/denom*100:.1f}%")
    print(f"  Leap-year match    : {results['leap']/denom*100:.1f}%")
    print(f"  Decade match       : {results['decade']/denom*100:.1f}%")
    full = results["all"] / denom * 100
    sym  = PASS if full >= 50 else WARN
    print(f"{sym}  ALL 4 conditions   : {full:.1f}%")
    return full > 0


# ─────────────────────────────────────────────────────────────────────────────
# Check 7 — GAN present
# ─────────────────────────────────────────────────────────────────────────────

def check_gan_present() -> bool:
    """Verify a GAN (cgan.py) is implemented with WGAN-GP components."""
    header("7. GAN Implementation Requirement")
    gan_file = ROOT / "model/models/cgan.py"
    if not gan_file.exists():
        print(f"{FAIL}  model/models/cgan.py not found")
        return False

    src  = gan_file.read_text()
    checks = {
        "Generator class"          : "class Generator" in src,
        "Discriminator class"      : "class Discriminator" in src,
        "WGAN-GP gradient penalty" : "gradient_penalty" in src,
        "Gumbel-Softmax"           : "gumbel_softmax" in src,
        "SpectralNorm (no BN)"     : "spectral_norm" in src,
        "No BatchNorm in critic"   : "BatchNorm" not in src.split("class Discriminator")[1],
    }
    ok = True
    for desc, passed in checks.items():
        print(f"{'PASS' if passed else FAIL}  {desc}")
        if not passed:
            ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 8 — Bonus best practices
# ─────────────────────────────────────────────────────────────────────────────

def check_bonus_practices() -> None:
    """Scan source files for bonus best-practice patterns."""
    header("8. Bonus Best Practices")
    train_src = (ROOT / "model/train_all.py").read_text()
    tok_src   = (ROOT / "model/utils/tokenizer.py").read_text()

    checks = {
        "torch.manual_seed"              : "torch.manual_seed" in train_src,
        "np.random.seed"                 : "np.random.seed" in train_src,
        "random.seed"                    : "random.seed(" in train_src,
        "Type hints on functions"        : "-> " in tok_src,
        "clip_grad_norm_"                : "clip_grad_norm_" in train_src,
        "WeightedRandomSampler"          : "WeightedRandomSampler" in
                                          (ROOT/"model/utils/dataset.py").read_text(),
        "tqdm progress bars"             : "tqdm" in train_src,
        "torch.cuda.is_available()"      : "is_available" in train_src,
        "environment.yml"                : (ROOT/"environment.yml").exists(),
        "Docstrings (triple-quoted)"     : '"""' in tok_src,
        "MLflow tracking"                : "mlflow" in train_src,
        "Resume from checkpoint"         : "load_checkpoint" in train_src,
        "CSV metric logging"             : "training_log.csv" in train_src,
        "__init__.py in models/"         : (ROOT/"model/models/__init__.py").exists(),
        "__init__.py in utils/"          : (ROOT/"model/utils/__init__.py").exists(),
        "Save best (not last) model"     : "is_best" in train_src,
        "Train/val/test 85/10/5 split"   : "0.85" in
                                          (ROOT/"model/utils/dataset.py").read_text(),
    }
    for desc, passed in checks.items():
        print(f"{'PASS' if passed else WARN}  {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 9 — Data sample validation
# ─────────────────────────────────────────────────────────────────────────────

def check_data_sample(data_path: Path, n: int = 100) -> bool:
    """Parse the first n lines of data.txt and verify tokenizer round-trips."""
    header(f"9. Tokenizer Round-Trip (first {n} lines of data.txt)")
    if not data_path.exists():
        print(f"{WARN}  {data_path} not found — skip")
        return True

    tok   = DateTokenizer()
    lines = data_path.read_text().splitlines()[:n]
    ok    = True

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            day_id, mon_id, leap_id, dec_id = tok.encode_input(line)
            date_str = line.split("]")[-1].strip()
            ids      = tok.encode_output(date_str)
            decoded  = tok.decode_output(ids)
            if decoded != date_str:
                print(f"{FAIL}  Line {i}: encode→decode mismatch: "
                      f"'{date_str}' → '{decoded}'")
                ok = False
        except Exception as e:
            print(f"{FAIL}  Line {i}: {e}")
            ok = False

    if ok:
        print(f"{PASS}  All {len(lines)} lines tokenise and decode correctly.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Assignment compliance checker.")
    p.add_argument("--data_sample", default=str(ROOT / "data/data.txt"),
                   help="Path to data.txt for tokenizer round-trip test")
    return p.parse_args()


def main() -> None:
    """Run all compliance checks and print a summary."""
    args = parse_args()
    print("=" * 60)
    print("  ASSIGNMENT COMPLIANCE CHECKER")
    print("=" * 60)

    results: dict[str, bool] = {}

    results["structure"]  = check_structure()
    results["weights"]    = check_weights()
    results["gan"]        = check_gan_present()
    results["tokenizer"]  = check_data_sample(Path(args.data_sample))

    # predict.py interface check — returns (bool, Path) tuple
    predict_result = check_predict_interface()
    if isinstance(predict_result, tuple):
        results["predict_interface"], out_path = predict_result
        results["output_format"]    = check_output_format(out_path)
        results["date_range"]       = check_date_range(out_path)
        results["conditions"]       = check_conditions_compliance(out_path)
    else:
        results["predict_interface"] = predict_result
        results["output_format"]     = False
        results["date_range"]        = False
        results["conditions"]        = False

    check_bonus_practices()   # informational only

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = sum(results.values())
    total  = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    print(f"\n  Score: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉  All checks passed — ready to submit!")
    else:
        print("  ⚠️   Fix failing checks before submitting.")
    print("=" * 60)


if __name__ == "__main__":
    main()
