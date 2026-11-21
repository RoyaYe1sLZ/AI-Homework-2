"""从逐轮日志绘制训练曲线。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(run_dir):
    with (run_dir / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    best_epoch = int(min(rows, key=lambda row: float(row["val_loss"]))["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy (%)")):
        for split, label, color in [("train", "Training", "#146954"), ("val", "Validation", "#b6562a")]:
            values = [float(row[f"{split}_{metric}"]) for row in rows]
            if metric == "accuracy":
                values = [value * 100 for value in values]
            ax.plot(epochs, values, label=label, color=color, linewidth=1.8)
        ax.axvline(best_epoch, color="0.5", linestyle=":", label=f"Selected epoch {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=title, xlim=(1, max(epochs)))
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(run_dir / "training_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "formal")
    plot_history(parser.parse_args().run_dir)
