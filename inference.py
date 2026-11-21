"""加载已训练的花卉模型，预测单张图片。"""
import hashlib

import torch

from data import CLASSES
from model import FlowerCNN
from preprocess import load_image


def load_model(run_dir, *, manifest=None):
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("权重保存的类别顺序与当前代码不一致")
    if manifest is not None:
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != checkpoint["manifest_sha256"]:
            raise ValueError("当前划分清单与训练权重记录的清单不一致")
    model = FlowerCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["epoch"]


@torch.inference_mode()
def predict(model, image_path):
    batch = load_image(image_path).unsqueeze(0)
    logits = model(batch)
    probabilities = logits.softmax(dim=1)[0]
    return dict(zip(CLASSES, probabilities.tolist()))
