"""准备识别模型和测试批次，检查正式评估的输入和输出。"""
import argparse
import csv
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader

from data import CLASSES, DATA_DIR
from inference import load_model
from preprocess import FlowerDataset, MANIFEST
from pathlib import Path


def prepare_evaluation(data_dir, run_dir):
    manifest = run_dir / "manifest.csv"
    model, epoch = load_model(run_dir, manifest=manifest)
    dataset = FlowerDataset("test", data_dir, manifest)
    batches = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    return model, epoch, batches


@torch.inference_mode()
def check_batch(data_dir, run_dir):
    model, epoch, batches = prepare_evaluation(data_dir, run_dir)
    images, labels = next(iter(batches))
    probabilities = model(images).softmax(dim=1)
    assert probabilities.shape == (len(labels), len(CLASSES))
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(len(labels)))
    counts = Counter(row["class_name"] for row in batches.dataset.rows)
    print(f"模型轮次 {epoch}，设备 {next(model.parameters()).device}，清单哈希核对通过")
    print("测试集类别数目", {name: counts[name] for name in CLASSES})
    print(f"共 {len(batches.dataset)} 张，{len(batches)} 批")
    print(f"首批输入 {tuple(images.shape)}，输出 {tuple(probabilities.shape)}，概率核对通过")


@torch.inference_mode()
def evaluate(data_dir, run_dir):
    model, epoch, batches = prepare_evaluation(data_dir, run_dir)
    rows = batches.dataset.rows
    matrix = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    predictions = []
    for images, labels in batches:
        probabilities = model(images).softmax(dim=1)
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(len(labels)))
        predicted = probabilities.argmax(dim=1)
        for truth, guess, probs in zip(labels.tolist(), predicted.tolist(), probabilities.tolist()):
            sample = rows[len(predictions)]
            assert int(sample["label"]) == truth
            matrix[truth, guess] += 1
            predictions.append(dict(path=sample["path"], actual=CLASSES[truth], predicted=CLASSES[guess],
                                    **{f"p_{name}": prob for name, prob in zip(CLASSES, probs)}))

    assert len(predictions) == len(rows)
    hits = matrix.diag()
    # F1 = 2 TP / (真实样本数 + 预测样本数)
    f1 = 2 * hits.double() / (matrix.sum(1) + matrix.sum(0)).clamp_min(1)
    metrics = dict(epoch=epoch, classes=list(CLASSES), total=len(rows), correct=int(hits.sum()),
                   accuracy=int(hits.sum()) / len(rows), macro_f1=float(f1.mean()),
                   class_f1=dict(zip(CLASSES, f1.tolist())), confusion_matrix=matrix.tolist())
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"正确 {metrics['correct']}/{len(rows)}，准确率 {metrics['accuracy']:.2%}，宏平均 F1 {metrics['macro_f1']:.4f}")
    print(matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=MANIFEST.parent)
    args = parser.parse_args()
    check_batch(args.data_dir, args.run_dir)
