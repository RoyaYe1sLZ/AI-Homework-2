"""花卉识别实验命令行入口。"""
import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from data import CLASSES, DATA_DIR, read_samples

SEED = 2026
FIELDS = ["path", "class_name", "label", "source_split", "sha256"]


def collect_unique(data_dir):
    groups = defaultdict(list)
    for source in ("train", "val"):
        for path, label in read_samples(data_dir, source):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups[digest].append(dict(path=path.relative_to(data_dir).as_posix(),
                                       class_name=CLASSES[label], label=label,
                                       source_split=source, sha256=digest))

    kept, excluded = [], []
    for group in groups.values():
        group.sort(key=lambda row: row["path"])
        if len({row["label"] for row in group}) > 1:
            excluded.extend(dict(row, reason="label_conflict", kept_path="") for row in group)
        else:
            first, *copies = group
            kept.append(first)
            excluded.extend(dict(row, reason="duplicate", kept_path=first["path"]) for row in copies)
    return kept, excluded


def assign_splits(rows):
    rng = random.Random(SEED)
    for row in rows:
        row["split"] = row["source_split"]
    for name in CLASSES:
        candidates = sorted((row for row in rows if row["source_split"] == "train"
                             and row["class_name"] == name), key=lambda row: row["path"])
        n_test = round(len(candidates) * 0.15)
        for row in rng.sample(candidates, n_test):
            row["split"] = "test"


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["path"]))


def prepare(data_dir, run_dir):
    kept, excluded = collect_unique(data_dir)
    assign_splits(kept)
    if len({row["sha256"] for row in kept}) != len(kept):
        raise ValueError("划分清单中仍有相同内容的图片")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "excluded.csv", FIELDS + ["reason", "kept_path"], excluded)
    write_csv(run_dir / "manifest.csv", FIELDS + ["split"], kept)
    counts = Counter((row["class_name"], row["split"]) for row in kept)
    print("类别          train  val  test")
    for name in CLASSES:
        print(f"{name:<12} {counts[name,'train']:>5} {counts[name,'val']:>4} {counts[name,'test']:>5}")
    print(f"保留 {len(kept)} 张，排除 {len(excluded)} 张。清单目录：{run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    default_run = Path(__file__).resolve().parent / "runs" / "formal"
    for name, help_text in [("prepare", "去重并生成划分清单"), ("train", "从随机初始化训练 CNN"),
                            ("evaluate", "在测试集评估最佳模型")]:
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--data-dir", type=Path, default=DATA_DIR)
        sub.add_argument("--run-dir", type=Path, default=default_run)
        if name == "train":
            sub.add_argument("--lr-schedule", choices=["constant", "plateau"], default="constant")
    prediction = commands.add_parser("predict", help="加载最佳权重并预测一张图片")
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--run-dir", type=Path, default=default_run)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_dir, args.run_dir)
    elif args.command == "train":
        from training import train
        train(args.data_dir, args.run_dir, SEED, args.lr_schedule)
    elif args.command == "evaluate":
        from evaluation import evaluate
        evaluate(args.data_dir, args.run_dir)
    else:
        from inference import load_model, predict
        try:
            model, epoch = load_model(args.run_dir)
            probabilities = predict(model, args.image)
        except (OSError, ValueError) as error:
            parser.exit(1, f"无法完成预测：{error}\n")
        print(f"权重轮次：{epoch}，预测类别：{max(probabilities, key=probabilities.get)}")
        for name, probability in probabilities.items():
            print(f"{name:<12} {probability:.4%}")
