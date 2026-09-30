"""Command-line workflow from dataset preparation to figures."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .channels import sample_mask
from .config import Config, load_config
from .data import loader, prepare
from .evaluation import benchmark, evaluate, load_method
from .models import setup
from .plotting import plot_results
from .system import BASELINES, VARIANTS
from .training import calibrate, train_clean, train_recovery
from .transport import decode_datagrams, encode_datagrams


def names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


@torch.no_grad()
def infer(cfg, method, image_path=None, loss=0.1, channel="bernoulli", seed=10017, output=None):
    device = setup(cfg)
    model, protection, payload = load_method(cfg, method, device)
    dataset = loader(cfg, "test", limit=1).dataset
    if image_path:
        with Image.open(image_path) as image:
            x = dataset.transform(image.convert("RGB"))
        label = None
    else:
        x, label, _ = dataset[0]
    z = model.edge(x[None].to(device))
    wire = protection.encode(z)
    shape = tuple(z.shape[1:])
    datagrams = encode_datagrams(protection, wire[0], shape, payload["profile_id"])
    keep = sample_mask(1, protection.total_packets, loss, channel, np.random.default_rng(seed))[0]
    received = [packet for packet, present in zip(datagrams, keep) if present]
    restored_wire, restored_mask = decode_datagrams(
        protection, received, shape, payload["profile_id"]
    )
    recovered = protection.decode(restored_wire.to(device), restored_mask.to(device), shape[-2:])
    logits = model.cloud(recovered)
    output = Path(output or Path(cfg.run_dir) / "inference")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "packets.bin").open("wb") as file:
        for packet in datagrams:
            file.write(struct.pack("<I", len(packet)))
            file.write(packet)
    np.savez_compressed(
        output / "activations.npz",
        clean=z.cpu().numpy(),
        recovered=recovered.cpu().numpy(),
        received_mask=keep.numpy(),
    )
    report = {
        "method": method,
        "profile_id": payload["profile_id"],
        "label": label,
        "prediction": int(logits.argmax(1)),
        "clean_prediction": int(model.cloud(z).argmax(1)),
        "erased_packets": int((~keep).sum()),
        "total_packets": protection.total_packets,
        "transmitted_bytes": sum(map(len, datagrams)),
        "mae": float((recovered - z).abs().mean()),
        "channel": channel,
        "requested_loss": loss,
        "trace_seed": seed,
    }
    (output / "prediction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def smoke(output):
    cfg = Config(
        dataset="synthetic",
        data_root=str(Path(output) / "data"),
        run_dir=str(output),
        backbone="tiny",
        image_size=16,
        num_classes=3,
        batch_size=8,
        device="cpu",
        clean_epochs=1,
        recovery_epochs=1,
        finetune_epochs=1,
        calibration_samples=24,
        saliency_samples=6,
        packets=4,
        parity=2,
        neighbors=3,
        sparsity=3,
        bottleneck=4,
        synthetic_train=30,
        synthetic_test=12,
        loss_rates=(0.0, 0.1, 0.3),
    ).validate()
    prepare(cfg)
    train_clean(cfg)
    calibrate(cfg)
    train_recovery(cfg, VARIANTS)
    results = evaluate(cfg, list(VARIANTS) + list(BASELINES), bootstrap=100)
    infer(cfg, "graphfec")
    benchmark(cfg, ["graphfec", "rs", "duplication"], samples=3, warmup=1)
    plot_results([results / "summary.csv"], Path(output) / "figures")
    print("Synthetic functional run complete. These outputs are not benchmark results.")


def main():
    parser = argparse.ArgumentParser(description="GraphFEC implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "train-clean",
        "calibrate",
        "train-recovery",
        "evaluate",
        "infer",
        "benchmark",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, help="JSON experiment configuration")
        command.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a config field; repeat for multiple fields",
        )
        if name == "prepare":
            command.add_argument("--download", action="store_true")
            command.add_argument(
                "--classes", help="Text file listing ImageNet class folders in label order"
            )
        elif name == "calibrate":
            command.add_argument("--clean", help="Clean checkpoint; defaults to run_dir/clean.pt")
        elif name == "train-recovery":
            command.add_argument(
                "--variants", default="graphfec", help="Comma-separated variants or all"
            )
        elif name in ("evaluate", "benchmark"):
            command.add_argument("--methods", default="graphfec,none,rs,rlc,uep")
            command.add_argument("--output")
            if name == "evaluate":
                command.add_argument("--channels", default="bernoulli,gilbert")
                command.add_argument("--trace-seeds", default="10017")
                command.add_argument("--split", choices=("val", "test"), default="test")
                command.add_argument("--limit", type=int)
                command.add_argument("--bootstrap", type=int, default=10000)
                command.add_argument(
                    "--trace", help="Observed 0/1 erasure trace, no rate rescaling"
                )
            else:
                command.add_argument("--samples", type=int, default=100)
                command.add_argument("--warmup", type=int, default=10)
                command.add_argument("--loss", type=float, default=0.1)
                command.add_argument(
                    "--channel", choices=("bernoulli", "gilbert"), default="bernoulli"
                )
        elif name == "infer":
            command.add_argument("--method", default="graphfec", choices=VARIANTS + BASELINES)
            command.add_argument("--image")
            command.add_argument("--loss", type=float, default=0.1)
            command.add_argument("--channel", choices=("bernoulli", "gilbert"), default="bernoulli")
            command.add_argument("--trace-seed", type=int, default=10017)
            command.add_argument("--output")
    plot = commands.add_parser("plot")
    plot.add_argument(
        "--input", nargs="+", required=True, help="One or more evaluation summary.csv files"
    )
    plot.add_argument("--output", required=True)
    quick = commands.add_parser("smoke")
    quick.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    if args.command == "plot":
        plot_results(args.input, args.output)
        return
    if args.command == "smoke":
        smoke(args.output)
        return
    cfg = load_config(args.config, args.set)
    if args.command == "prepare":
        prepare(cfg, args.download, args.classes)
    elif args.command == "train-clean":
        train_clean(cfg)
    elif args.command == "calibrate":
        calibrate(cfg, args.clean)
    elif args.command == "train-recovery":
        train_recovery(cfg, VARIANTS if args.variants == "all" else names(args.variants))
    elif args.command == "evaluate":
        methods = (
            list(VARIANTS) + ["none", "rs", "rlc", "uep"]
            if args.methods == "all"
            else names(args.methods)
        )
        evaluate(
            cfg,
            methods,
            names(args.channels),
            [int(v) for v in names(args.trace_seeds)],
            args.split,
            args.limit,
            args.bootstrap,
            args.trace,
            args.output,
        )
    elif args.command == "benchmark":
        benchmark(
            cfg,
            names(args.methods),
            args.samples,
            args.warmup,
            args.loss,
            args.channel,
            args.output,
        )
    elif args.command == "infer":
        infer(cfg, args.method, args.image, args.loss, args.channel, args.trace_seed, args.output)


if __name__ == "__main__":
    main()
