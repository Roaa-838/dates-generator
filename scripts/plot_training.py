import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model/weights/training_log.csv")
for model in df["model"].unique():
    m = df[df["model"] == model]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(m["epoch"], m["train_loss"]); ax1.set_title(f"{model} - Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(m["epoch"], m["full_pass_rate"], label="full")
    ax2.plot(m["epoch"], m["day_pass_rate"],  label="day")
    ax2.plot(m["epoch"], m["month_pass_rate"],label="month")
    ax2.legend(); ax2.set_title(f"{model} - Condition Rates"); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(f"eval_results/{model}_training.png", dpi=150)
    plt.close()
    print(f"Saved {model}_training.png")