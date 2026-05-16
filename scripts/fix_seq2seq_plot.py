import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

Path("eval_results").mkdir(exist_ok=True)

df = pd.read_csv("model/weights/training_log.csv")

# Keep only seq2seq rows where loss < 10 (the fixed run, not the NaN/millions run)
seq = df[(df["model"] == "seq2seq") & (df["train_loss"] < 10)].copy()
seq = seq.reset_index(drop=True)
seq["epoch"] = range(1, len(seq) + 1)

print(f"Seq2seq clean rows: {len(seq)}")
print(seq[["epoch","train_loss","full_pass_rate","day_pass_rate","month_pass_rate"]].head(10))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("SEQ2SEQ — Training Curves", fontsize=13, fontweight="bold")

ax1.plot(seq["epoch"], seq["train_loss"], color="steelblue", linewidth=1.8)
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_ylim(0.85, 1.30)
ax1.grid(True, alpha=0.3)

ax2.plot(seq["epoch"], seq["full_pass_rate"],   label="full (all 4)",  color="royalblue",  linewidth=2)
ax2.plot(seq["epoch"], seq["day_pass_rate"],    label="day",           color="darkorange", linewidth=1.5, linestyle="--")
ax2.plot(seq["epoch"], seq["month_pass_rate"],  label="month",         color="green",      linewidth=1.5)
ax2.plot(seq["epoch"], seq["leap_pass_rate"],   label="leap",          color="red",        linewidth=1.5, linestyle=":")
ax2.plot(seq["epoch"], seq["decade_pass_rate"], label="decade",        color="purple",     linewidth=1.5, linestyle="-.")
ax2.axhline(y=1/7, color="gray", linestyle="--", alpha=0.5, label="random day (1/7)")
ax2.set_ylim(-0.05, 1.05)
ax2.set_title("Condition Satisfaction Rates")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Rate")
ax2.legend(fontsize=8, loc="center right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eval_results/seq2seq_training_FIXED.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eval_results/seq2seq_training_FIXED.png")