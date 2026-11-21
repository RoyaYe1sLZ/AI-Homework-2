"""固定图像预处理与按清单取样的 PyTorch 数据集。"""
import csv
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data import CLASSES, DATA_DIR

IMAGE_SIZE = 128
MANIFEST = Path(__file__).resolve().parent / "runs" / "formal" / "manifest.csv"
resize = transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR)
center_crop = transforms.CenterCrop(IMAGE_SIZE)
to_tensor = transforms.ToTensor()


def load_image(path):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return to_tensor(center_crop(resize(rgb)))


class FlowerDataset(Dataset):
    def __init__(self, split, data_dir=DATA_DIR, manifest=MANIFEST):
        self.data_dir = Path(data_dir)
        with Path(manifest).open(encoding="utf-8", newline="") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"清单中找不到集合 {split!r}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return load_image(self.data_dir / row["path"]), int(row["label"])


if __name__ == "__main__":
    dataset = FlowerDataset("train")
    path = dataset.data_dir / dataset.rows[0]["path"]
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        resized = resize(rgb)
        cropped = center_crop(resized)
        print(f"{path.name}  原图 {rgb.size} → 缩放 {resized.size} → 裁剪 {cropped.size}")
        print(f"裁剪后左上角 RGB：{cropped.getpixel((0, 0))}")
    tensor, label = dataset[0]
    print(f"张量 {tuple(tensor.shape)}，{tensor.dtype}，类别 {CLASSES[label]} ({label})")
    print("左上角归一化 RGB：", tensor[:, 0, 0].tolist())
    images, labels = next(iter(DataLoader(dataset, batch_size=32)))
    print(f"一批图片 {tuple(images.shape)}，一批标签 {tuple(labels.shape)}")
