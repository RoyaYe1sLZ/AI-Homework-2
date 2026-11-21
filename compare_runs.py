"""复核两次实验的逐图预测，将改对、改错的图片逐张配对。"""
import csv
import hashlib
import json
import math
from pathlib import Path

from data import CLASSES


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def review(run):
    manifest = run / "manifest.csv"
    history = read_csv(run / "history.csv")
    config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "test_metrics.json").read_text())
    expected = {r["path"]: r["class_name"] for r in read_csv(manifest) if r["split"] == "test"}
    predictions = read_csv(run / "test_predictions.csv")
    best = min(history, key=lambda row: float(row["val_loss"]))
    assert int(best["epoch"]) == metrics["epoch"]
    assert len(history) == config["epochs"] == 25
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == config["manifest_sha256"]
    assert len(predictions) == len(expected) == metrics["total"]
    assert {r["path"]: r["actual"] for r in predictions} == expected
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    correct = {}
    for row in predictions:
        probs = [float(row[f"p_{name}"]) for name in CLASSES]
        assert all(0 <= p <= 1 for p in probs) and abs(sum(probs) - 1) < 1e-6
        true = CLASSES.index(row["actual"])
        pred = probs.index(max(probs))
        assert CLASSES[pred] == row["predicted"]
        matrix[true][pred] += 1
        correct[row["path"]] = pred == true
    assert matrix == metrics["confusion_matrix"]
    macro_f1 = sum(2 * matrix[i][i] / (sum(matrix[i]) + sum(row[i] for row in matrix))
                   for i in range(len(CLASSES))) / len(CLASSES)
    assert math.isclose(macro_f1, metrics["macro_f1"])
    assert math.isclose(sum(correct.values()) / len(correct), metrics["accuracy"])
    return config, best, metrics, correct


def plot_comparison(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for name, label, color in [("formal", "Fixed LR", "#9b552e"), ("plateau", "Plateau", "#176c57")]:
        history = read_csv(root / name / "history.csv")
        config = json.loads((root / name / "config.json").read_text())
        epochs = [int(row["epoch"]) for row in history]
        loss = [float(row["val_loss"]) for row in history]
        rates = [float(row.get("learning_rate", config["learning_rate"])) for row in history]
        axes[0].plot(epochs, loss, label=label, color=color)
        best = loss.index(min(loss))
        axes[0].scatter(epochs[best], loss[best], color=color)
        axes[1].step(epochs, rates, where="post", color=color, label=label)
    axes[0].set(ylabel="Validation cross-entropy")
    axes[1].set(ylabel="Learning rate", yscale="log")
    for ax in axes:
        ax.set(xlabel="Epoch", xlim=(1, 25))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.savefig(root / "plateau" / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "runs"
    baseline, optimized = [review(root / name) for name in ("formal", "plateau")]
    for key in ("manifest_sha256", "learning_rate", "batch_size", "epochs", "seed", "optimizer", "device"):
        assert baseline[0][key] == optimized[0][key]
    old, new = baseline[3], optimized[3]
    fixed = sum(not old[path] and new[path] for path in old)
    lost = sum(old[path] and not new[path] for path in old)
    print(f"原错现对 {fixed} 张，原对现错 {lost} 张；净变化 {fixed - lost:+d} 张")
    for name, (_, best, metrics, _) in zip(("基线", "调度"), (baseline, optimized)):
        print(name, best, metrics)
    plot_comparison(root)
