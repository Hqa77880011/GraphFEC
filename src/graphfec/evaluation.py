"""Common erasure traces, per-example logs, paired comparisons, and local timing."""

import csv
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .channels import load_trace, sample_mask
from .data import loader
from .metrics import stratified_interval
from .models import setup
from .system import BASELINES, Protection
from .training import check_compatible, load_checkpoint
from .transport import HEADER


def load_method(cfg, method, device):
    path = Path(cfg.run_dir) / ("profile.pt" if method in BASELINES else f"{method}.pt")
    saved, model, payload = load_checkpoint(path, device)
    check_compatible(cfg, saved)
    if method not in BASELINES:
        for name in (
            "packets",
            "parity",
            "neighbors",
            "sparsity",
            "balance",
            "eta",
            "ridge",
            "bottleneck",
            "parity_precision",
            "seed",
            "beta",
            "train_channel",
        ):
            if getattr(cfg, name) != getattr(saved, name):
                raise ValueError(
                    f"{method} checkpoint disagrees on {name}; train the new configuration"
                )
    protection = Protection(cfg, payload["profile"], method).to(device)
    if method not in BASELINES:
        protection.load_state_dict(payload["protection"])
    else:
        settings = {
            name: getattr(cfg, name)
            for name in (
                "packets",
                "parity",
                "neighbors",
                "balance",
                "sparsity",
                "eta",
                "ridge",
                "seed",
                "parity_precision",
            )
        }
        payload["profile_id"] = str(
            uuid.uuid5(
                uuid.UUID(payload["profile_id"]),
                json.dumps({"method": method, **settings}, sort_keys=True),
            )
        )
    protection.eval()
    return model, protection, payload


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(
    cfg,
    methods,
    channels=("bernoulli", "gilbert"),
    trace_seeds=(10017,),
    split="test",
    limit=None,
    bootstrap=10000,
    trace=None,
    output=None,
):
    device = setup(cfg)
    output = Path(output or Path(cfg.run_dir) / f"evaluation_{split}")
    output.mkdir(parents=True, exist_ok=True)
    batches = loader(cfg, split, limit=limit)
    n = len(batches.dataset)
    max_packets = 2 * cfg.packets if "duplication" in methods else cfg.packets + cfg.parity
    summaries, comparisons = [], []
    scenario_list = (
        [("trace", -1.0, 0)]
        if trace
        else [
            (channel, rate, seed)
            for channel in channels
            for rate in cfg.loss_rates
            for seed in trace_seeds
        ]
    )
    fields = [
        "method",
        "channel",
        "loss_rate",
        "trace_seed",
        "sample_id",
        "label",
        "prediction",
        "correct",
        "clean_correct",
        "quantized_correct",
        "mae",
        "cosine",
        "erased_packets",
        "total_packets",
        "mask",
    ]
    with (output / "raw.csv").open("w", newline="", encoding="utf-8") as raw_file:
        raw = csv.DictWriter(raw_file, fieldnames=fields)
        raw.writeheader()
        for channel, rate, trace_seed in scenario_list:
            masks = (
                load_trace(trace, n, max_packets)
                if trace
                else sample_mask(n, max_packets, rate, channel, np.random.default_rng(trace_seed))
            )
            outcomes, reference_labels = {}, None
            for method in methods:
                model, protection, payload = load_method(cfg, method, device)
                correct, clean_correct, quantized_correct, maes, cosines, labels_all = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                lost, count = 0, 0
                for images, labels, identifiers in batches:
                    images, labels = images.to(device), labels.to(device)
                    z = model.edge(images)
                    keep = masks[count : count + len(z), : protection.total_packets].to(device)
                    recovered = protection(z, keep)
                    predictions = model.cloud(recovered).argmax(1)
                    clean = model.cloud(z).argmax(1) == labels
                    quantized = (
                        model.cloud(
                            protection.quantizer.decode(protection.quantizer.encode(z))
                        ).argmax(1)
                        == labels
                    )
                    matches = predictions == labels
                    mae = (recovered - z).abs().flatten(1).mean(1)
                    cosine = F.cosine_similarity(recovered.flatten(1), z.flatten(1), dim=1)
                    for i in range(len(z)):
                        raw.writerow(
                            dict(
                                zip(
                                    fields,
                                    [
                                        method,
                                        channel,
                                        rate,
                                        trace_seed,
                                        int(identifiers[i]),
                                        int(labels[i]),
                                        int(predictions[i]),
                                        int(matches[i]),
                                        int(clean[i]),
                                        int(quantized[i]),
                                        float(mae[i]),
                                        float(cosine[i]),
                                        int((~keep[i]).sum()),
                                        protection.total_packets,
                                        "".join("1" if v else "0" for v in keep[i].tolist()),
                                    ],
                                )
                            )
                        )
                    correct.extend(matches.cpu().tolist())
                    clean_correct.extend(clean.cpu().tolist())
                    quantized_correct.extend(quantized.cpu().tolist())
                    maes.extend(mae.cpu().tolist())
                    cosines.extend(cosine.cpu().tolist())
                    labels_all.extend(labels.cpu().tolist())
                    lost += int((~keep).sum())
                    count += len(z)
                low, high = stratified_interval(correct, labels_all, bootstrap)
                wire_bytes = protection.payload_bytes(payload["profile"]["shape"][-2:])
                summaries.append(
                    {
                        "method": method,
                        "channel": channel,
                        "loss_rate": rate,
                        "training_seed": cfg.seed,
                        "trace_seed": trace_seed,
                        "examples": count,
                        "accuracy": float(np.mean(correct)),
                        "accuracy_ci_low": low,
                        "accuracy_ci_high": high,
                        "clean_accuracy": float(np.mean(clean_correct)),
                        "quantized_accuracy": float(np.mean(quantized_correct)),
                        "mae": float(np.mean(maes)),
                        "cosine": float(np.mean(cosines)),
                        "observed_loss": lost / (count * protection.total_packets),
                        "intra_correlation": protection.intra,
                        "data_packets": cfg.packets,
                        "parity_packets": protection.r,
                        "payload_bytes": wire_bytes,
                        "header_bytes": protection.total_packets * HEADER.size,
                        "transmitted_bytes": wire_bytes + protection.total_packets * HEADER.size,
                        "decoder_parameters": sum(
                            p.numel() for p in protection.refiner.parameters()
                        )
                        if protection.refiner is not None
                        else 0,
                    }
                )
                outcomes[method] = np.asarray(correct, dtype=float)
                reference_labels = labels_all
                print(
                    f"{method} {channel} p={rate:g} trace={trace_seed}: accuracy={np.mean(correct):.4f}"
                )
            if "graphfec" in outcomes:
                for method, matches in outcomes.items():
                    if method == "graphfec":
                        continue
                    difference = outcomes["graphfec"] - matches
                    low, high = stratified_interval(difference, reference_labels, bootstrap)
                    comparisons.append(
                        {
                            "reference": "graphfec",
                            "method": method,
                            "channel": channel,
                            "loss_rate": rate,
                            "trace_seed": trace_seed,
                            "gain_pp": float(difference.mean() * 100),
                            "ci_low_pp": low * 100,
                            "ci_high_pp": high * 100,
                            "ci_excludes_zero": low > 0 or high < 0,
                        }
                    )
            raw_file.flush()
    write_csv(output / "summary.csv", summaries)
    if comparisons:
        write_csv(output / "paired_comparisons.csv", comparisons)
    (output / "evaluation.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "split": split,
                "limit": limit,
                "methods": methods,
                "channels": list(channels),
                "trace_seeds": list(trace_seeds),
                "bootstrap_resamples": bootstrap,
                "trace": str(trace) if trace else None,
                "synthetic": cfg.dataset == "synthetic",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark(cfg, methods, samples=100, warmup=10, loss=0.1, channel="bernoulli", output=None):
    if samples < 1 or warmup < 0:
        raise ValueError("samples must be positive and warmup nonnegative")
    device = setup(cfg)
    batches = loader(cfg, "test", batch_size=1)
    dataset = batches.dataset
    masks = sample_mask(
        samples + warmup, 2 * cfg.packets, loss, channel, np.random.default_rng(10017)
    )
    raw, summary = [], []
    for method in methods:
        model, protection, _ = load_method(cfg, method, device)
        timings = []
        for i in range(samples + warmup):
            image, _, _ = dataset[i % len(dataset)]
            image = image[None].to(device)
            keep = masks[i : i + 1, : protection.total_packets].to(device)
            synchronize(device)
            start = time.perf_counter()
            z = model.edge(image)
            synchronize(device)
            edge_end = time.perf_counter()
            wire = protection.encode(z)
            synchronize(device)
            encode_end = time.perf_counter()
            recovered = protection.decode(wire, keep, z.shape[-2:])
            synchronize(device)
            decode_end = time.perf_counter()
            model.cloud(recovered)
            synchronize(device)
            end = time.perf_counter()
            row = {
                "method": method,
                "request": i,
                "warmup": i < warmup,
                "edge_ms": (edge_end - start) * 1000,
                "encode_ms": (encode_end - edge_end) * 1000,
                "decode_ms": (decode_end - encode_end) * 1000,
                "cloud_ms": (end - decode_end) * 1000,
                "total_ms": (end - start) * 1000,
                "erased_packets": int((~keep).sum()),
                "status": "ok",
            }
            raw.append(row)
            if i >= warmup:
                timings.append(row)
        total = np.array([row["total_ms"] for row in timings])
        summary.append(
            {
                "method": method,
                "samples": samples,
                "encode_ms": float(np.mean([r["encode_ms"] for r in timings])),
                "decode_ms": float(np.mean([r["decode_ms"] for r in timings])),
                "median_ms": float(np.median(total)),
                "p95_ms": float(np.quantile(total, 0.95)),
                "p99_ms": float(np.quantile(total, 0.99)),
                "compute_qps": float(1000 / total.mean()),
            }
        )
    output = Path(output or Path(cfg.run_dir) / "benchmark")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "raw.csv", raw)
    write_csv(output / "summary.csv", summary)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "torch": str(torch.__version__),
                "threads": cfg.threads,
                "loss": loss,
                "channel": channel,
                "warmup": warmup,
                "samples": samples,
                "timing_scope": "single-process compute; excludes data loading and networking",
            },
            indent=2,
        )
        + "\n"
    )
    return output
