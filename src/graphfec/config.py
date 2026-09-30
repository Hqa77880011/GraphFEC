"""Experiment settings shared by every command."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    dataset: str = "cifar100"
    data_root: str = "data/cifar100"
    run_dir: str = "runs/cifar100"
    backbone: str = "resnet18"
    image_size: int = 32
    num_classes: int = 100
    split_stage: int = 2
    seed: int = 17
    split_seed: int = 2026
    val_fraction: float = 0.1
    batch_size: int = 128
    workers: int = 0
    threads: int = 4
    device: str = "auto"
    clean_epochs: int = 200
    clean_lr: float = 0.1
    recovery_epochs: int = 120
    finetune_epochs: int = 20
    recovery_lr: float = 0.0003
    weight_decay: float = 0.0001
    calibration_samples: int = 4096
    saliency_samples: int = 1024
    packets: int = 20
    parity: int = 4
    neighbors: int = 8
    beta: float = 0.95
    balance: float = 0.1
    sparsity: int = 8
    eta: float = 0.001
    ridge: float = 0.000001
    cache_size: int = 256
    bottleneck: int = 16
    parity_precision: str = "uint8"
    train_channel: str = "mixed"
    loss_rates: tuple = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3)
    lambda_l1: float = 1.0
    lambda_cos: float = 1.0
    lambda_kd: float = 1.0
    temperature: float = 2.0
    synthetic_train: int = 48
    synthetic_test: int = 24

    def validate(self):
        if self.dataset not in {"synthetic", "cifar100", "tinyimagenet", "imagenet100"}:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        if not 1 <= self.packets <= 255 or not 0 <= self.parity <= self.packets:
            raise ValueError("Require 1 <= packets <= 255 and 0 <= parity <= packets")
        if self.packets + self.parity > 255:
            raise ValueError("GF(256) supports at most 255 total packets")
        if not 0 < self.val_fraction < 1 or not 0 <= self.beta < 1:
            raise ValueError("Require 0 < val_fraction < 1 and 0 <= beta < 1")
        if self.train_channel not in {"mixed", "bernoulli", "gilbert"}:
            raise ValueError("train_channel must be mixed, bernoulli or gilbert")
        if not self.loss_rates or any(not 0 <= p <= 0.8 for p in self.loss_rates):
            raise ValueError("Loss rates must lie in [0, 0.8] for the four-packet Gilbert model")
        if self.parity_precision not in {"uint8", "float32"}:
            raise ValueError("parity_precision must be uint8 or float32")
        for name in (
            "batch_size",
            "threads",
            "clean_epochs",
            "recovery_epochs",
            "calibration_samples",
            "saliency_samples",
            "neighbors",
            "sparsity",
            "bottleneck",
            "num_classes",
            "image_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.finetune_epochs < 0 or self.workers < 0 or self.cache_size < 0:
            raise ValueError("Epoch, worker and cache counts cannot be negative")
        if self.ridge <= 0 or self.eta < 0 or self.temperature <= 0:
            raise ValueError("Require ridge > 0, eta >= 0 and temperature > 0")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path, overrides=()):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, value = item.split("=", 1)
        try:
            values[key] = json.loads(value)
        except json.JSONDecodeError:
            values[key] = value
    return Config(**values).validate()
