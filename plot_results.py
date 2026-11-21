"""用正式实验的测试记录生成报告插图。"""
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from compare_runs import read_csv, review
from data import CLASSES, DATA_DIR
from preprocess import load_image

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = RUNS / "plateau" / "figures"
NAMES = dict(zip(CLASSES, ("雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香")))
GREEN, ORANGE = "#13756a", "#aa542c"


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def confusion(baseline, optimized):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), layout="constrained")
    fig.suptitle("同一批测试图片的分类去向", fontsize=18)
    for ax, metrics, name in zip(axes, (baseline, optimized), ("固定学习率", "动态学习率")):
        matrix = np.array(metrics["confusion_matrix"])
        fractions = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(fractions, cmap="BuGn", vmin=0, vmax=1)
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f"{matrix[i,j]}\n{fractions[i,j]:.0%}", ha="center", va="center",
                    color="white" if fractions[i,j] > .6 else "#203632", fontsize=10)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=NAMES.values(), yticklabels=NAMES.values(),
               xlabel="预测类别", ylabel="真实类别", title=f"{name}   ·   {metrics['correct']} / {metrics['total']} 张正确")
    save(fig, "01_confusion.png")


def class_scores(baseline, optimized):
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    x = np.arange(len(CLASSES))
    for offset, metrics, color, name in [(-.18, baseline, "#889a9b", "固定学习率"),
                                          (.18, optimized, GREEN, "动态学习率")]:
        bars = ax.bar(x + offset, [metrics["class_f1"][c] for c in CLASSES], .34, color=color, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.set(xticks=x, xticklabels=NAMES.values(), ylim=(0, 1.06), ylabel="F1", title="五类花卉的 F1 对照")
    ax.legend(ncols=2, loc="upper center", frameon=False)
    save(fig, "02_class_f1.png")


def photo(ax, row):
    ax.imshow(load_image(DATA_DIR / row["path"]).permute(1, 2, 0).numpy())
    ax.axis("off")


def examples(rows):
    selected = [next(r for r in rows if r["actual"] == c) for c in CLASSES]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5), layout="constrained")
    fig.suptitle("五类花卉预测   ·   每个真实类别按测试清单选第一张", fontsize=16)
    for i, row in enumerate(selected):
        photo(axes[0,i], row)
        pred = row["predicted"]
        axes[0,i].set_title(f"真实  {NAMES[row['actual']]}\n预测  {NAMES[pred]}", fontsize=11)
        ax = axes[1,i]
        values = [100 * float(row["p_"+c]) for c in CLASSES]
        bars = ax.barh(range(5), values, color=[GREEN if c == pred else "#aabfbd" for c in CLASSES])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
        ax.set(yticks=range(5), yticklabels=NAMES.values(), xlim=(0,140), xticks=[])
        ax.invert_yaxis()
        ax.tick_params(length=0, labelsize=9)
        ax.spines[:].set_visible(False)
    save(fig, "03_predictions.png")
    return selected


def mistakes(rows):
    errors = [r for r in rows if r["actual"] != r["predicted"]]
    pairs = Counter((r["actual"], r["predicted"]) for r in errors)
    selected = []
    fig, axes = plt.subplots(2, 3, figsize=(9, 7), layout="constrained")
    fig.suptitle("六种最常见错判方向及其高置信度样例", fontsize=16)
    for ax, ((actual, pred), count) in zip(axes.flat, pairs.most_common(6)):
        row = max((r for r in errors if (r["actual"], r["predicted"]) == (actual,pred)),
                  key=lambda r: float(r["p_"+pred]))
        selected.append(row)
        photo(ax, row)
        ax.set_title(f"{NAMES[actual]} → {NAMES[pred]}  ·  共 {count} 张\n此图预测概率 {float(row['p_'+pred]):.1%}", fontsize=11)
    save(fig, "04_errors.png")
    return selected


def credits(groups):
    source = (DATA_DIR / "LICENSE.txt").read_text()
    entries = {line.split(" CC-BY by ")[0]: line for line in source.splitlines() if " CC-BY by " in line}
    lines = ["展示图片来源与摄影者署名", "许可 CC BY 2.0   https://creativecommons.org/licenses/by/2.0/",
             "展示图经过缩放与中心裁剪；预测分数及标注来自本实验。"]
    for name, rows in groups:
        lines += ["", name]
        for row in rows:
            key = "/".join(Path(row["path"]).parts[-2:])
            lines.append(entries[key])
    (OUT / "IMAGE_CREDITS.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    for family in ("Heiti TC", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if family in font_manager.get_font_names():
            plt.rcParams["font.family"] = family
            break
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, optimized = [review(RUNS / r)[2] for r in ("formal", "plateau")]
    rows = read_csv(RUNS / "plateau" / "test_predictions.csv")
    confusion(baseline, optimized)
    class_scores(baseline, optimized)
    credits([("03_predictions.png", examples(rows)), ("04_errors.png", mistakes(rows))])
