"""训练和验证四层花卉 CNN。"""
import csv
import hashlib
import json
import math
import platform
import tempfile
from time import perf_counter

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from data import CLASSES
from model import FlowerCNN
from preprocess import FlowerDataset, load_image


def run_epoch(model, batches, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            count += len(labels)
            total_loss += loss.detach() * len(labels)
            total_correct += (logits.detach().argmax(1) == labels).sum()
    mean_loss = total_loss.item() / count
    if not math.isfinite(mean_loss):
        raise ValueError("本轮损失不是有限值")
    return mean_loss, total_correct.item() / count


def smoke_check(dataset, device, criterion):
    model = FlowerCNN().to(device)
    before = model.classifier.weight.detach().clone()
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
    loss, _ = run_epoch(model, [batch], device, criterion, torch.optim.Adam(model.parameters(), lr=0.001))
    delta = (model.classifier.weight.detach() - before).abs().max().item()
    assert delta > 0
    image = load_image(dataset.data_dir / dataset.rows[0]["path"]).unsqueeze(0).to(device)
    with tempfile.TemporaryFile() as tmp, torch.inference_mode():
        torch.save(model.state_dict(), tmp)
        tmp.seek(0)
        restored = FlowerCNN().to(device).eval()
        restored.load_state_dict(torch.load(tmp, map_location=device, weights_only=True))
        expected = model.eval()(image)
        actual = restored(image)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (1, 5) and actual.isfinite().all()
    return dict(batch_size=8, loss=loss, max_weight_change=delta, reload_prediction_matches=True)


def train(data_dir, run_dir, seed, lr_schedule="constant"):
    manifest = run_dir / "manifest.csv"
    train_set = FlowerDataset("train", data_dir, manifest)
    val_set = FlowerDataset("val", data_dir, manifest)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    config = dict(seed=seed, epochs=25, batch_size=32, learning_rate=0.001, optimizer="Adam",
                  device=str(device), python=platform.python_version(), torch=str(torch.__version__),
                  torchvision=torchvision.__version__, classes=CLASSES,
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    config["lr_schedule"] = lr_schedule
    if lr_schedule == "plateau":
        config["scheduler_options"] = dict(mode="min", factor=0.5, patience=2, min_lr=1e-5)
    with (run_dir / "config.json").open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    torch.manual_seed(seed)
    smoke = smoke_check(train_set, device, criterion)
    (run_dir / "smoke.json").write_text(json.dumps(smoke, indent=2))
    print("小批量检查通过", smoke, flush=True)

    torch.manual_seed(seed)
    model = FlowerCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = None
    if lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["scheduler_options"])
    generator = torch.Generator().manual_seed(seed)
    training = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, generator=generator)
    validation = DataLoader(val_set, batch_size=config["batch_size"])
    best = float("inf")
    with (run_dir / "history.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds", "learning_rate"])
        for epoch in range(1, config["epochs"] + 1):
            start = perf_counter()
            train_loss, train_acc = run_epoch(model, training, device, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, validation, device, criterion)
            if val_loss < best:
                checkpoint = dict(model_state=model.state_dict(), epoch=epoch, val_loss=val_loss,
                                  classes=list(CLASSES), manifest_sha256=config["manifest_sha256"])
                temporary = run_dir / "best.pt.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(run_dir / "best.pt")
                best = val_loss
            learning_rate = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step(val_loss)
            seconds = perf_counter() - start
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, seconds, learning_rate])
            file.flush()
            print(f"{epoch:02d}/25 训练 {train_loss:.4f}/{train_acc:.2%}  验证 {val_loss:.4f}/{val_acc:.2%}  {seconds:.1f}s", flush=True)
