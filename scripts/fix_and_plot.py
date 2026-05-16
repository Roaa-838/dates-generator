import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load and clean ───────────────────────────────────────────
df = pd.read_csv("model/weights/training_log.csv")

# For seq2seq: keep only the last 50 epochs (the fixed run)
# Drop rows where loss is NaN or absurdly large (>1000)
df_clean = df[df["train_loss"].notna() & (df["train_loss"] < 1000)].copy()

# For each model keep only the last N epochs in case of duplicate runs
cleaned_parts = []
for model in df_clean["model"].unique():
    m = df_clean[df_clean["model"] == model].copy()
    # Re-index epochs from 1
    m = m.reset_index(drop=True)
    m["epoch"] = range(1, len(m) + 1)
    cleaned_parts.append(m)

df_clean = pd.concat(cleaned_parts, ignore_index=True)
df_clean.to_csv("model/weights/training_log_clean.csv", index=False)
print("Cleaned CSV saved.")

# ── Plot each model ──────────────────────────────────────────
Path("eval_results").mkdir(exist_ok=True)

for model in df_clean["model"].unique():
    m = df_clean[df_clean["model"] == model]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{model.upper()} — Training Curves", fontsize=13, fontweight="bold")

    # Left: Loss
    ax1.plot(m["epoch"], m["train_loss"], color="steelblue", linewidth=1.8)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Right: Condition rates
    ax2.plot(m["epoch"], m["full_pass_rate"],   label="full (all 4)",  linewidth=2,   color="royalblue")
    ax2.plot(m["epoch"], m["day_pass_rate"],    label="day",           linewidth=1.5, color="darkorange", linestyle="--")
    ax2.plot(m["epoch"], m["month_pass_rate"],  label="month",         linewidth=1.5, color="green")
    ax2.plot(m["epoch"], m["leap_pass_rate"],   label="leap",          linewidth=1.5, color="red",    linestyle=":")
    ax2.plot(m["epoch"], m["decade_pass_rate"], label="decade",        linewidth=1.5, color="purple", linestyle="-.")
    ax2.axhline(y=1/7, color="gray", linestyle="--", alpha=0.5, label="random day (1/7)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Condition Satisfaction Rates")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Rate")
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"eval_results/{model}_training.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

print("\nAll plots saved to eval_results/")