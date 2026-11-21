from pathlib import Path

from PIL import Image


DATA_DIR = Path(__file__).resolve().parent / "flower_data"
CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")


def read_samples(data_dir, split):
    """按固定类别顺序读取路径，每个样本记录为 (图片路径, 类别编号)。"""
    samples = []
    for label, class_name in enumerate(CLASSES):
        folder = data_dir / split / class_name
        paths = sorted(folder.glob("*.jpg"))
        if not paths:
            raise ValueError(f"目录中没有找到 JPG 图片：{folder}")
        samples.extend((path, label) for path in paths)
    return samples


if __name__ == "__main__":
    train = read_samples(DATA_DIR, "train")
    val = read_samples(DATA_DIR, "val")

    print("类别编号  类别          train  val")
    for label, class_name in enumerate(CLASSES):
        n_train = sum(target == label for _, target in train)
        n_val = sum(target == label for _, target in val)
        print(f"{label:>4}      {class_name:<12} {n_train:>4}  {n_val:>3}")
    print(f"合计：训练 {len(train)} 张，验证 {len(val)} 张")

    path, label = train[0]
    with Image.open(path) as image:
        image.load()
        print(f"样本：{path.relative_to(DATA_DIR)}")
        print(f"标签：{label}（{CLASSES[label]}），宽×高：{image.width}×{image.height}，模式：{image.mode}")
