"""四个卷积块组成的五类花卉分类网络。"""
import torch
from torch import nn

from data import CLASSES


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class FlowerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(256, len(CLASSES))

    def forward(self, images):
        features = self.features(images)
        pooled = self.flatten(self.pool(features))
        return self.classifier(self.dropout(pooled))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from preprocess import FlowerDataset

    torch.manual_seed(2026)
    model = FlowerCNN().eval()
    images, labels = next(iter(DataLoader(FlowerDataset("train"), batch_size=32)))
    print("输入", tuple(images.shape))

    def show_shape(layer, inputs, output):
        print(layer.__class__.__name__, tuple(output.shape))

    for layer in [*model.features, model.pool, model.flatten, model.dropout, model.classifier]:
        layer.register_forward_hook(show_shape)
    with torch.inference_mode():
        logits = model(images)
    assert logits.shape == (len(labels), len(CLASSES)) and logits.isfinite().all()
    print("可训练参数", sum(p.numel() for p in model.parameters() if p.requires_grad))
