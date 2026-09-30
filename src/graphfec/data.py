"""Dataset preparation, fixed holdouts and deterministic evaluation order."""

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def stratified_split(labels, fraction, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, val = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            raise ValueError("Every class needs at least two training examples")
        n = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        val.extend(indices[:n].tolist())
        train.extend(indices[n:].tolist())
    return sorted(train), sorted(val)


def prepare(cfg, download=False, classes_file=None):
    """Write a portable manifest; image data stay in their original directories."""
    root = Path(cfg.data_root)
    root.mkdir(parents=True, exist_ok=True)
    if cfg.dataset == "cifar100":
        training = datasets.CIFAR100(root, train=True, download=download)
        testing = datasets.CIFAR100(root, train=False, download=download)
        classes = training.classes
        source = [[i, int(y)] for i, y in enumerate(training.targets)]
        test = [[i, int(y)] for i, y in enumerate(testing.targets)]
    elif cfg.dataset == "synthetic":
        classes = [str(i) for i in range(cfg.num_classes)]
        source = [[i, i % cfg.num_classes] for i in range(cfg.synthetic_train)]
        test = [[i, i % cfg.num_classes] for i in range(cfg.synthetic_test)]
    else:
        if cfg.dataset == "tinyimagenet" and download and not (root / "train").exists():
            archive = root / "tiny-imagenet-200.zip"
            urllib.request.urlretrieve("https://cs231n.stanford.edu/tiny-imagenet-200.zip", archive)
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    relative = Path(info.filename)
                    if relative.parts[0] != "tiny-imagenet-200":
                        raise ValueError("Unexpected Tiny ImageNet archive layout")
                    target = (root / Path(*relative.parts[1:])).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError("Archive path escapes data_root")
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zf.read(info))
            archive.unlink()
        available = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
        classes = (
            Path(classes_file).read_text().split() if classes_file else available[: cfg.num_classes]
        )
        if len(classes) != cfg.num_classes or len(set(classes)) != len(classes):
            raise ValueError(f"Expected {cfg.num_classes} distinct class folders")
        source, test = [], []
        for label, cls in enumerate(classes):
            files = sorted(
                p
                for p in (root / "train" / cls).rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            source.extend([[p.relative_to(root).as_posix(), label] for p in files])
        if cfg.dataset == "tinyimagenet":
            mapping = {cls: i for i, cls in enumerate(classes)}
            lines = (root / "val" / "val_annotations.txt").read_text().splitlines()
            for line in sorted(lines):
                name, cls, *_ = line.split()
                if cls in mapping:
                    test.append([f"val/images/{name}", mapping[cls]])
        else:
            for label, cls in enumerate(classes):
                files = sorted(
                    p
                    for p in (root / "val" / cls).rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                test.extend([[p.relative_to(root).as_posix(), label] for p in files])
        if {y for _, y in source} != set(range(cfg.num_classes)):
            raise ValueError("Training images are missing for selected classes")
        if {y for _, y in test} != set(range(cfg.num_classes)):
            raise ValueError("Test images are missing for selected classes")
    train, val = stratified_split([y for _, y in source], cfg.val_fraction, cfg.split_seed)
    manifest = {
        "dataset": cfg.dataset,
        "classes": classes,
        "split_seed": cfg.split_seed,
        "val_fraction": cfg.val_fraction,
        "train": [source[i] for i in train],
        "val": [source[i] for i in val],
        "test": test,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {path}: train={len(train)}, val={len(val)}, test={len(test)}")


class ManifestDataset(Dataset):
    def __init__(self, cfg, split, augment=False, limit=None):
        self.cfg, self.split = cfg, split
        manifest = json.loads((Path(cfg.data_root) / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["dataset"] != cfg.dataset
            or len(manifest["classes"]) != cfg.num_classes
            or manifest["split_seed"] != cfg.split_seed
            or manifest["val_fraction"] != cfg.val_fraction
        ):
            raise ValueError("Config and dataset manifest disagree; run prepare with this config")
        self.rows = manifest[split]
        self.identifiers = list(range(len(self.rows)))
        if limit is not None and limit < 1:
            raise ValueError("Sample limit must be positive")
        if limit is not None and limit < len(self.rows):
            selected = np.random.default_rng(cfg.split_seed).choice(
                len(self.rows), limit, replace=False
            )
            self.rows = [self.rows[i] for i in sorted(selected)]
            self.identifiers = sorted(selected)
        self.cifar = (
            datasets.CIFAR100(cfg.data_root, train=split != "test")
            if cfg.dataset == "cifar100"
            else None
        )
        ops = []
        if augment:
            ops = (
                [transforms.RandomCrop(cfg.image_size, padding=4)]
                if cfg.image_size <= 64
                else [transforms.RandomResizedCrop(cfg.image_size)]
            )
            ops.append(transforms.RandomHorizontalFlip())
        else:
            ops = [
                transforms.Resize(
                    cfg.image_size if cfg.image_size <= 64 else round(cfg.image_size * 256 / 224)
                ),
                transforms.CenterCrop(cfg.image_size),
            ]
        self.transform = transforms.Compose(
            ops
            + [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        identifier, label = self.rows[index]
        if self.cfg.dataset == "synthetic":
            # A deterministic color/pattern task for functional checks only.
            seed = self.cfg.split_seed + identifier + (1000000 if self.split == "test" else 0)
            rng = torch.Generator().manual_seed(seed)
            x = torch.randn(3, self.cfg.image_size, self.cfg.image_size, generator=rng) * 0.1
            x[label % 3] += 0.5 + label / self.cfg.num_classes
        else:
            if self.cifar is not None:
                img, _ = self.cifar[identifier]
            else:
                with Image.open(Path(self.cfg.data_root) / identifier) as source:
                    img = source.convert("RGB")
            x = self.transform(img)
        return x, label, self.identifiers[index]


def loader(cfg, split, augment=False, limit=None, batch_size=None):
    dataset = ManifestDataset(cfg, split, augment=augment, limit=limit)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=augment,
        num_workers=cfg.workers,
        generator=torch.Generator().manual_seed(cfg.seed),
        pin_memory=cfg.device.startswith("cuda"),
    )
