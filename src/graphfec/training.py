"""Clean training, train-only calibration, and task-aware recovery training."""

import copy
import csv
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .channels import sample_mask
from .config import Config
from .data import loader
from .graph import ChannelStatistics
from .models import build_model, setup
from .system import VARIANTS, Protection


def save_checkpoint(path, cfg, model, **extra):
    payload = {"version": 1, "config": cfg.to_dict(), "model": model.state_dict(), **extra}
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    cfg = Config(**payload["config"]).validate()
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return cfg, model, payload


def check_compatible(cfg, saved):
    for name in (
        "dataset",
        "backbone",
        "image_size",
        "num_classes",
        "split_stage",
        "split_seed",
        "val_fraction",
    ):
        if getattr(cfg, name) != getattr(saved, name):
            raise ValueError(f"Checkpoint and config disagree on {name}")


@torch.no_grad()
def validation(model, batches, device, protection=None, seed=17001):
    model.eval()
    if protection is not None:
        protection.eval()
    correct, total = 0, 0
    rng = np.random.default_rng(seed)
    for images, labels, _ in batches:
        images, labels = images.to(device), labels.to(device)
        if protection is None:
            logits = model(images)
        else:
            z = model.edge(images)
            mask = sample_mask(len(z), protection.total_packets, 0.1, "bernoulli", rng).to(device)
            logits = model.cloud(protection(z, mask))
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return correct / total


def train_clean(cfg):
    device = setup(cfg)
    root = Path(cfg.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg).to(device)
    train, val = loader(cfg, "train", augment=True), loader(cfg, "val")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.clean_lr, momentum=0.9, weight_decay=cfg.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.clean_epochs)
    best = -1.0
    with (root / "clean_history.csv").open("w", newline="", encoding="utf-8") as file:
        history = csv.writer(file)
        history.writerow(["epoch", "train_ce", "val_accuracy"])
        for epoch in range(cfg.clean_epochs):
            model.train()
            loss_sum, count = 0.0, 0
            for images, labels, _ in train:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite clean training loss")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
                count += len(labels)
            score = validation(model, val, device)
            schedule.step()
            history.writerow([epoch + 1, loss_sum / count, score])
            file.flush()
            if score > best:
                best = score
                save_checkpoint(
                    root / "clean.pt",
                    cfg,
                    model,
                    epoch=epoch + 1,
                    val_accuracy=score,
                    method="clean",
                )
            print(f"clean epoch {epoch + 1}/{cfg.clean_epochs}: val_accuracy={score:.4f}")


def calibrate(cfg, clean_path=None):
    device = setup(cfg)
    root = Path(cfg.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved_cfg, model, _ = load_checkpoint(clean_path or root / "clean.pt", device)
    check_compatible(cfg, saved_cfg)
    stats = None
    shape = None
    with torch.no_grad():
        for images, _, _ in loader(cfg, "train", limit=cfg.calibration_samples):
            z = model.edge(images.to(device))
            if stats is None:
                stats = ChannelStatistics(z.shape[1], cfg.beta)
                shape = list(z.shape[1:])
            stats.update(z)
    model.requires_grad_(False)
    saliency = torch.zeros(shape[0], device=device)
    saliency_count = 0
    for images, labels, _ in loader(cfg, "val", limit=cfg.saliency_samples):
        with torch.no_grad():
            z = model.edge(images.to(device))
        z.requires_grad_(True)
        loss = F.cross_entropy(model.cloud(z), labels.to(device), reduction="sum")
        (gradient,) = torch.autograd.grad(loss, z)
        saliency += (gradient * z).detach().abs().mean((2, 3)).sum(0)
        saliency_count += len(z)
    profile = {
        "correlation": stats.correlation(),
        "correlation_static": stats.correlation(False),
        "low": stats.low,
        "high": stats.high,
        "mean": (stats.total / stats.count).float(),
        "saliency": (saliency / saliency_count).cpu(),
        "shape": shape,
        "calibration_examples": stats.count,
        "saliency_examples": saliency_count,
    }
    save_checkpoint(
        root / "profile.pt",
        cfg,
        model,
        profile=profile,
        method="profile",
        profile_id=str(uuid.uuid4()),
    )
    print(f"Calibrated {stats.count} training examples; split shape={shape}")


def task_loss(recovered, clean, logits, teacher_logits, labels, cfg, task_only=False):
    ce = F.cross_entropy(logits, labels)
    if task_only:
        return ce
    l1 = F.l1_loss(recovered, clean)
    cosine = (1 - F.cosine_similarity(recovered.flatten(1), clean.flatten(1), dim=1)).mean()
    temperature = cfg.temperature
    kd = F.kl_div(
        F.log_softmax(logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
    return ce + cfg.lambda_l1 * l1 + cfg.lambda_cos * cosine + cfg.lambda_kd * temperature**2 * kd


def train_recovery(cfg, variants):
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"{variant} is not a trainable variant")
        device = setup(cfg)
        root = Path(cfg.run_dir)
        saved_cfg, model, payload = load_checkpoint(root / "profile.pt", device)
        check_compatible(cfg, saved_cfg)
        # Calibration settings are part of the profile, not silently overridden.
        if cfg.beta != saved_cfg.beta:
            raise ValueError("beta changed: recalibrate the profile")
        profile = payload["profile"]
        protection = Protection(cfg, profile, variant).to(device)
        teacher = copy.deepcopy(model.cloud).requires_grad_(False).eval()
        model.requires_grad_(False)
        model.enable_final_stage()
        final_parameters = [p for p in model.parameters() if p.requires_grad]
        model.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [{"params": protection.parameters()}, {"params": final_parameters}],
            lr=cfg.recovery_lr,
            weight_decay=cfg.weight_decay,
        )
        epochs = cfg.recovery_epochs + cfg.finetune_epochs
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        train, val = loader(cfg, "train", augment=True), loader(cfg, "val")
        rng = np.random.default_rng(cfg.seed)
        best = -1.0
        with (root / f"{variant}_history.csv").open("w", newline="", encoding="utf-8") as file:
            history = csv.writer(file)
            history.writerow(["epoch", "phase", "train_loss", "val_accuracy"])
            for epoch in range(epochs):
                model.eval()
                tune = epoch >= cfg.recovery_epochs
                if tune:
                    model.enable_final_stage()
                protection.train()
                loss_sum, count = 0.0, 0
                for images, labels, _ in train:
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        clean = model.edge(images)
                        teacher_logits = teacher(clean)
                    channel = (
                        rng.choice(["bernoulli", "gilbert"])
                        if cfg.train_channel == "mixed"
                        else cfg.train_channel
                    )
                    rate = float(rng.choice(cfg.loss_rates))
                    mask = sample_mask(len(clean), protection.total_packets, rate, channel, rng).to(
                        device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    recovered = protection(clean, mask)
                    logits = model.cloud(recovered)
                    loss = task_loss(
                        recovered,
                        clean,
                        logits,
                        teacher_logits,
                        labels,
                        cfg,
                        task_only=variant == "task_only",
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {variant} training loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(labels)
                    count += len(labels)
                score = validation(model, val, device, protection)
                schedule.step()
                phase = "finetune" if tune else "frozen"
                history.writerow([epoch + 1, phase, loss_sum / count, score])
                file.flush()
                if score > best:
                    best = score
                    save_checkpoint(
                        root / f"{variant}.pt",
                        cfg,
                        model,
                        method=variant,
                        profile=profile,
                        protection=protection.state_dict(),
                        profile_id=str(uuid.uuid4()),
                        epoch=epoch + 1,
                        val_accuracy=score,
                    )
                print(f"{variant} epoch {epoch + 1}/{epochs} ({phase}): val_accuracy={score:.4f}")
        (root / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
