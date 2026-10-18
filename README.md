# GraphFEC

PyTorch implementation of **GraphFEC: Correlation-Graph-Coded Activation Transmission for Robust Edge-Cloud Split Inference**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Preparation downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), preserves its test split and reserves a fixed validation holdout. `data_root` and `run_dir` in `configs/cifar100.json` control paths. Use repeated `--set key=value` to override them consistently.

```bash
graphfec prepare --config configs/cifar100.json --download
```

For [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), use `configs/tinyimagenet.json` with `prepare --download`. For [ImageNet](https://image-net.org/download.php), obtain its labeled folders and use `configs/imagenet100.json`; `--classes classes.txt` pins the selected class order. Keep the manifest and split seed fixed across methods.

## Training and inference

Fit the clean classifier, calibrate its graph/quantizer, then fit recovery. The saved profile links every downstream checkpoint to its calibration. The config fixes training seed 17; use a new `run_dir` for another seed.

```bash
graphfec train-clean --config configs/cifar100.json
graphfec calibrate --config configs/cifar100.json
graphfec train-recovery --config configs/cifar100.json --variants graphfec
graphfec infer --config configs/cifar100.json --loss 0.1 --channel gilbert --trace-seed 10017
```

## Evaluation

Baselines `none`, `rs`, `rlc` and `uep` use the same calibrated profile. Evaluation records explicit trace seeds and its bootstrap settings.

```bash
graphfec evaluate --config configs/cifar100.json --methods graphfec,none,rs,rlc,uep --trace-seeds 10017,10029,10043
graphfec plot --input runs/cifar100/evaluation_test/summary.csv --output runs/cifar100/figures
```

## Ablations

Each recovery variant needs its own fitting. `all` includes `random`, `cluster`, `no_parity`, `no_refinement`, `neural_only`, `task_only` and `static_graph`. Use separate run directories when changing parity, graph or split settings.

```bash
graphfec train-recovery --config configs/cifar100.json --variants all
graphfec evaluate --config configs/cifar100.json --methods all --trace-seeds 10017,10029,10043 --output runs/cifar100/ablations
graphfec plot --input runs/cifar100/ablations/summary.csv --output runs/cifar100/ablation_figures
```

## Outputs

The run directory contains `clean.pt`, `profile.pt`, `<variant>.pt` and their training histories. Inference writes `prediction.json`, `packets.bin` and `activations.npz`. Evaluation writes per-example `raw.csv`, `summary.csv`, `paired_comparisons.csv` and `evaluation.json`; plots read those summaries. Packet serialization is available through `graphfec.transport`.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
