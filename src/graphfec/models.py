"""Backbones with explicit edge/cloud boundaries."""

from collections import OrderedDict

import torch
from torch import nn
from torchvision import models


class SplitClassifier(nn.Module):
    def __init__(self, edge, cloud, tune_names):
        super().__init__()
        self.edge = edge
        self.cloud = cloud
        self.tune_names = tune_names

    def forward(self, images):
        return self.cloud(self.edge(images))

    def enable_final_stage(self):
        """Train only the final feature stage and classifier; keep edge statistics fixed."""
        for name in self.tune_names:
            module = self.cloud.get_submodule(name)
            module.requires_grad_(True)
            module.train()


def build_model(cfg):
    """Build a fresh classifier without downloading pretrained weights."""
    if cfg.backbone == "resnet18":
        if cfg.split_stage not in (1, 2, 3):
            raise ValueError("ResNet split_stage must be 1, 2 or 3")
        net = models.resnet18(weights=None, num_classes=cfg.num_classes)
        net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        stages = [(f"layer{i}", getattr(net, f"layer{i}")) for i in range(1, 5)]
        edge = nn.Sequential(
            OrderedDict(
                [("conv1", net.conv1), ("bn1", net.bn1), ("relu", net.relu)]
                + stages[: cfg.split_stage]
            )
        )
        cloud = nn.Sequential(
            OrderedDict(
                stages[cfg.split_stage :]
                + [("pool", net.avgpool), ("flatten", nn.Flatten()), ("classifier", net.fc)]
            )
        )
        tune_names = ("layer4", "classifier")
    elif cfg.backbone == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=None, num_classes=cfg.num_classes)
        edge = net.features[:7]
        cloud = nn.Sequential(
            OrderedDict(
                [
                    ("features", net.features[7:]),
                    ("pool", net.avgpool),
                    ("flatten", nn.Flatten()),
                    ("classifier", net.classifier),
                ]
            )
        )
        tune_names = ("features.16", "classifier")
    elif cfg.backbone == "convnext_tiny":
        net = models.convnext_tiny(weights=None, num_classes=cfg.num_classes)
        edge = net.features[:4]
        cloud = nn.Sequential(
            OrderedDict(
                [
                    ("features", net.features[4:]),
                    ("pool", net.avgpool),
                    ("classifier", net.classifier),
                ]
            )
        )
        tune_names = ("features.7", "classifier")
    elif cfg.backbone == "tiny":
        edge = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2))
        cloud = nn.Sequential(
            OrderedDict(
                [
                    ("features", nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())),
                    ("pool", nn.AdaptiveAvgPool2d(1)),
                    ("flatten", nn.Flatten()),
                    ("classifier", nn.Linear(16, cfg.num_classes)),
                ]
            )
        )
        tune_names = ("features", "classifier")
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")
    return SplitClassifier(edge, cloud, tune_names)


def setup(cfg):
    """Seed training and select the requested device."""
    import random

    import numpy as np

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
