# GraphFEC

[![CI](https://github.com/Hqa77880011/GraphFEC/actions/workflows/ci.yml/badge.svg)](https://github.com/Hqa77880011/GraphFEC/actions/workflows/ci.yml)

PyTorch implementation of **GraphFEC: Correlation-Graph-Coded Activation Transmission for Robust Edge-Cloud Split Inference**.

GraphFEC protects the intermediate activation of a split classifier. A channel correlation graph guides balanced packet assignment and sparse parity coding. The receiver combines a regularized linear solve with two graph refinement blocks before running the cloud classifier.

The workflow is: prepare data → train the classifier → calibrate the graph and quantizer → train recovery → evaluate and plot. CIFAR-100, Tiny ImageNet-200, and ImageNet-100 configurations are included.

## Installation

Use Python 3.11 or newer. Create and activate a virtual environment:

```bash
git clone https://github.com/Hqa77880011/GraphFEC.git
cd GraphFEC
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install a matching PyTorch and TorchVision pair using the [PyTorch installation selector](https://pytorch.org/get-started/locally/). For the CPU environment used by CI:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
graphfec smoke --output runs/smoke
```

The smoke command checks the complete pipeline on CPU using generated images and a tiny CNN. It runs one epoch per training phase, evaluates the methods, and writes outputs to `runs/smoke/`. The generated data are for functional checks; no dataset download is needed.

`python -m graphfec` is equivalent to `graphfec`. Run the commands below from the repository root. Use `graphfec --help` or, for example, `graphfec evaluate --help` to list options.

Training and calibration save to `run_dir`; inference, evaluation, and timing also accept `--output`. Repeating a command replaces its named output files. Use separate directories for experiments you want to keep. Data and generated outputs are excluded from Git.

## 1. Prepare data

| Configuration | Dataset / backbone | Default split tensor |
|---|---|---|
| `configs/cifar100.json` | CIFAR-100 / ResNet-18 | 128 × 16 × 16 |
| `configs/tinyimagenet.json` | Tiny ImageNet-200 / MobileNetV3-Large | 40 × 8 × 8 |
| `configs/imagenet100.json` | ImageNet-100 / ConvNeXt-Tiny | 192 × 28 × 28 |

### CIFAR-100

The data come from the [CIFAR dataset page](https://www.cs.toronto.edu/~kriz/cifar.html). TorchVision downloads the Python archive:

```bash
graphfec prepare --config configs/cifar100.json --download
```

### Tiny ImageNet

Download from the [Stanford Tiny ImageNet archive](https://cs231n.stanford.edu/tiny-imagenet-200.zip):

```bash
graphfec prepare --config configs/tinyimagenet.json --download
```

For an existing download, extract the archive into `data/tiny-imagenet-200/` and omit `--download`. The loader reads `train/<wnid>/images/`, `val/images/`, and `val/val_annotations.txt` in place.

### ImageNet-100

Obtain ImageNet from the [ImageNet download site](https://www.image-net.org/download.php) and arrange the training and validation images in class folders:

```text
data/imagenet/
  train/n01440764/*.JPEG
  train/...
  val/n01440764/*.JPEG
  val/...
```

```bash
graphfec prepare --config configs/imagenet100.json
```

By default, this selects the first 100 lexicographically sorted training class folders. To use a specific subset, supply `--classes classes.txt`, with one class folder name per line. The supplied order defines the labels. The selected classes and image paths are saved to `data_root/manifest.json`.

To use a different data location, add `--set data_root=/path/to/dataset` to preparation and subsequent commands. Keep the same manifest throughout an experiment so that class labels and holdouts stay consistent.

For all datasets, a fixed, class-stratified 10% holdout of the original training set is used for validation. CIFAR's original test set and the original labeled validation sets of Tiny ImageNet/ImageNet are used as test data. Graph and quantization calibration use only the remaining training partition. Gradient saliency for UEP uses the holdout. The split seed is independent of the model seed and stays fixed across methods.

## 2. Train the clean classifier

```bash
graphfec train-clean --config configs/cifar100.json
```

This trains from scratch with SGD, momentum 0.9 and cosine decay, and saves the best validation checkpoint as `runs/cifar100/clean.pt`. The history is in `clean_history.csv`. The CIFAR ResNet uses a 3 × 3 stride-one stem without max pooling. Training uses random crops and horizontal flips; evaluation uses deterministic resizing/center cropping. All image datasets use ImageNet channel normalization.

Use the other configuration paths for their corresponding dataset and backbone. `device=auto` selects CUDA when available; `--set device=cpu` forces CPU. Full training is intended for a GPU. Batch size is 128 for CIFAR-100 and Tiny ImageNet, and 32 for ConvNeXt. Reduce it with `--set batch_size=16` if needed.

## 3. Calibrate the protection profile

```bash
graphfec calibrate --config configs/cifar100.json
```

This computes global-average-pooled channel statistics, EMA and full-sample absolute Pearson correlations, channel minima/maxima/means, and validation gradient saliency. The result is `profile.pt`, which also contains the clean classifier. Calibration selects at most 4,096 training examples and 1,024 validation examples, deterministically and without augmentation.

The profile fixes the channel graph, packet assignment, and quantization ranges used during recovery training and inference. Calibration must finish before running `train-recovery` or any baseline.

## 4. Train GraphFEC

```bash
graphfec train-recovery --config configs/cifar100.json --variants graphfec
```

The clean backbone stays frozen for 120 epochs. The final cloud feature stage and classifier are then tuned with the recovery module for 20 epochs. AdamW uses learning rate `3e-4`, weight decay `1e-4`, and cosine decay across both phases. Each mini-batch samples a loss rate and either Bernoulli or Gilbert erasures. The objective combines cross-entropy, activation L1, cosine consistency, and temperature-scaled KL distillation from the fixed clean classifier.

Validation uses fixed masks at 10% Bernoulli loss. The best checkpoint is `graphfec.pt`; `graphfec_history.csv` records the phase, training loss, and validation accuracy. Checkpoints contain the classifier, graph statistics, quantizer, packet map, parity weights, refiner weights, configuration, and a profile UUID.

The default CIFAR-100 run produces these checkpoints:

| File | Created by | Used by |
|---|---|---|
| `runs/cifar100/clean.pt` | `train-clean` | `calibrate` |
| `runs/cifar100/profile.pt` | `calibrate` | Recovery training and all baselines |
| `runs/cifar100/graphfec.pt` | `train-recovery --variants graphfec` | GraphFEC inference, evaluation, and timing |
| `runs/cifar100/<variant>.pt` | `train-recovery --variants <variant>` | The corresponding ablation |

## 5. Execute a protected inference

```bash
graphfec infer --config configs/cifar100.json --loss 0.1 --channel gilbert
```

This runs one test image through packet encoding, simulated loss, recovery, and cloud classification. Add `--image path/to/image.jpg` to use your own image, or `--method rs` to select a baseline. Image preprocessing and labels follow the prepared dataset; `prediction` is the class index in `manifest.json`. Set `--trace-seed` to change the sampled erasures.

Files in `runs/cifar100/inference/`:

| File | Contents |
|---|---|
| `prediction.json` | Prediction, clean prediction, label when available, lost packets, bytes and MAE |
| `packets.bin` | Transmitted datagrams, each preceded by a four-byte little-endian length |
| `activations.npz` | Clean and recovered activations and the received-packet mask |

`graphfec.transport` provides the packet serialization API. Each datagram has a 42-byte header containing format version, profile UUID, sample ID, packet index/count, activation shape, payload length, and CRC32. Reported bytes include these headers and exclude the file's four-byte length prefixes. Parsing checks the profile, sample, shape, packet index, and checksum. `infer` runs locally; an application can send the same datagrams through its own transport.

## 6. Compare baselines and evaluate

```bash
graphfec evaluate --config configs/cifar100.json --methods graphfec,none,rs,rlc,uep --trace-seeds 10017,10029,10043
graphfec plot --input runs/cifar100/evaluation_test/summary.csv --output runs/cifar100/figures
```

The default comparison requires `graphfec.pt` and `profile.pt`. To evaluate only the baselines after calibration, use `--methods none,rs,rlc,uep`; they do not require recovery training.

The default loss rates are 0%, 1%, 5%, 10%, 20%, and 30%, under both Bernoulli and Gilbert loss. Gilbert bad-state duration has mean four packets, with stationary initialization and transition probabilities `P(B→G)=0.25`, `P(G→B)=0.25p/(1-p)`. Each example is an independent stationary sequence. Input order and reception-mask prefixes are shared across methods, and evaluation seeds are separate from training seeds.

| Method | Implementation |
|---|---|
| `graphfec` | Learned sparse parity, graph-regularized solve, two graph refinement blocks |
| `none` | Systematic packets only; erased channels filled with training means |
| `rs` | Systematic Vandermonde Reed–Solomon over GF(256), polynomial `0x11D` |
| `rlc` | Fixed dense Gaussian real parity, row normalized; linear solve without graph regularization |
| `uep` | Grad-FEC-style saliency allocation: sort packets by mean absolute gradient × activation, split into high/low halves, allocate parity in proportion to saliency, and code each half with RS |
| `duplication` | A second copy of every data packet; 100% payload redundancy |

RS and UEP solve over GF(256). If the received equations do not determine every packet, uniquely determined packets are retained and unresolved packets use training means. RLC, RS and UEP use the graph interleaver so that the baseline comparison isolates the protection mechanism; packet-assignment alternatives are trained separately below.

Evaluation writes:

| File | Interpretation |
|---|---|
| `raw.csv` | One row per example, method and channel setting; predictions, clean/quantized correctness, MAE, cosine and reception mask |
| `summary.csv` | Accuracy, 95% label-stratified bootstrap CI, activation errors, observed loss, within-packet correlation, actual byte budget and decoder parameter count |
| `paired_comparisons.csv` | GraphFEC minus each method in percentage points, with a paired stratified bootstrap interval |
| `evaluation.json` | Configuration, split, sample limit, trace seeds and bootstrap settings |

Accuracy is stored as a fraction; `gain_pp` is the GraphFEC accuracy gain in percentage points. MAE and cosine compare recovered activations with the unquantized activation, so no-loss MAE includes quantization error. `clean_accuracy` and `quantized_accuracy` use each method's own cloud suffix. Fine-tuned variants can therefore have different clean references.

The default confidence intervals use 10,000 label-stratified bootstrap resamples for each checkpoint and trace. A positive interval in `paired_comparisons.csv` indicates a GraphFEC gain for that setting. Training-seed variation is evaluated with separate runs. Plots show arithmetic means across the supplied run/trace rows; combine files only when their dataset, split, and byte budget match.

Use `--split val` for parameter selection and `--split test` for final evaluation. For a short check, add `--limit 100 --bootstrap 200`. Plotting writes `<channel>_metrics.png` and `.pdf`, plus `component_comparison.png` when 10% Bernoulli results are available.

For a recorded loss trace:

```bash
graphfec evaluate --config configs/cifar100.json --methods graphfec,rs --trace path/to/erasures.csv --output runs/cifar100/trace_evaluation
```

Provide your recorded trace as headerless values `0` (received) and `1` (erased), separated by commas or, for a `.txt` file, whitespace. The trace needs at least `number_of_examples × (M+R)` events, or `number_of_examples × 2M` when duplication is included. It is consumed in example-major packet order at its recorded loss rate. `observed_loss` reports the fraction actually erased; trace rows use `loss_rate=-1` to indicate that no synthetic loss rate was requested.

## 7. Run ablations and sensitivity studies

```bash
graphfec train-recovery --config configs/cifar100.json --variants all
graphfec evaluate --config configs/cifar100.json --methods all --output runs/cifar100/ablations
graphfec plot --input runs/cifar100/ablations/summary.csv --output runs/cifar100/ablation_figures
```

Each variant starts from the same calibrated clean checkpoint and is trained independently:

| Variant | Change from GraphFEC |
|---|---|
| `random` | Seeded random balanced channel assignment |
| `cluster` | Greedy assignment favors grouping correlated channels |
| `no_parity` | Zero parity packets; coarse graph recovery and refiner remain |
| `no_refinement` | Remove the neural refiner; train only parity coefficients during the frozen phase |
| `neural_only` | Replace the matrix solve with diagonal-normalized transpose backprojection; retain parity and refiner |
| `task_only` | Use cross-entropy alone |
| `static_graph` | Use full-sample calibration moments instead of EMA moments |

`all` trains GraphFEC and every listed variant. Evaluation also includes `none`, `rs`, `rlc`, and `uep`. To run a subset, use a comma-separated list such as `--variants random,no_refinement`. Each evaluated variant needs its own checkpoint. Check `transmitted_bytes` when comparing methods: `no_parity` sends fewer bytes and duplication sends twice the data payload.

For sensitivity studies, change one configuration field and use a separate run directory. For example, train a 10% redundancy model while reusing the clean classifier:

```bash
graphfec calibrate --config configs/cifar100.json --clean runs/cifar100/clean.pt --set run_dir=runs/rho10 --set parity=2
graphfec train-recovery --config configs/cifar100.json --set run_dir=runs/rho10 --set parity=2
graphfec evaluate --config configs/cifar100.json --set run_dir=runs/rho10 --set parity=2
```

Use `parity=0,1,2,4,6` in separate runs for 0%, 5%, 10%, 20%, 30% redundancy at `M=20`; `neighbors=4,8,16` for graph sparsity; and `train_channel=bernoulli,gilbert,mixed` for channel mismatch. Train and evaluate a changed profile using the same overrides. Changing `beta` requires recalibration. Changing ResNet `split_stage` to 1 or 3 requires a new clean training and calibration run because the checkpoint structure changes.

To assess training-seed variation, repeat sections 2–6 with `--set seed=29 --set run_dir=runs/cifar100_seed29`, then seed 43. Keep `split_seed` fixed. Pass multiple `summary.csv` paths to `graphfec plot --input ... --output ...` to plot their arithmetic means. Per-seed paired intervals remain available in the individual evaluation folders.

## 8. Measure local execution time

```bash
graphfec benchmark --config configs/cifar100.json --methods graphfec,none,rs,rlc,uep,duplication --samples 100 --warmup 10
```

The benchmark uses batch size one and paired inputs/masks. Synchronized monotonic timing covers edge, encoder, decoder and cloud computation, including any CPU transfers used by GF(256). It excludes image loading, serialization and networking. `benchmark/raw.csv` retains warm-up rows; `summary.csv` reports mean encoding/decoding time, median/P95/P99 compute latency, and serial compute throughput. Increase sample count for meaningful tail estimates. Device and timing scope are recorded in `metadata.json`.

## Important configuration fields

Commands that take `--config` also accept repeated `--set key=value` overrides. Strings may be unquoted; lists use JSON, e.g. `--set 'loss_rates=[0.0,0.1,0.2]'`. Full defaults are in [`src/graphfec/config.py`](src/graphfec/config.py); the dataset JSON files override selected values.

| Field | Default | Meaning |
|---|---|---|
| `data_root`, `run_dir` | Dataset configuration | Dataset location and checkpoint/history directory |
| `device`, `batch_size` | `auto`, 128 | Compute device and training/evaluation batch size; ImageNet-100 uses 32 |
| `clean_epochs`, `clean_lr` | 200, 0.1 | Clean-classifier schedule; Tiny ImageNet and ImageNet-100 use learning rate 0.05 |
| `recovery_epochs`, `finetune_epochs` | 120, 20 | Frozen-backbone and final-cloud-stage training epochs |
| `recovery_lr`, `weight_decay` | 0.0003, 0.0001 | Recovery optimizer settings |
| `packets`, `parity` | 20, 4 | Data and parity packets; redundancy is `R/M` |
| `neighbors`, `beta` | 8, 0.95 | Per-channel top-k graph and EMA coefficient |
| `balance`, `sparsity` | 0.1, 8 | Interleaver balance cost and max nonzeros per parity row |
| `eta`, `ridge` | 0.001, 0.000001 | Packet Laplacian weight and numerical ridge |
| `bottleneck`, `cache_size` | 16, 256 | Refiner hidden width and bounded reception-mask operator cache |
| `parity_precision` | `uint8` | `uint8` for equal payload bytes; `float32` for real-parity experiments |
| `lambda_l1`, `lambda_cos`, `lambda_kd` | 1, 1, 1 | Auxiliary loss weights |
| `temperature` | 2 | Distillation temperature |
| `calibration_samples`, `saliency_samples` | 4096, 1024 | Calibration and validation-saliency limits |
| `train_channel`, `loss_rates` | `mixed`, [0, 0.01, 0.05, 0.1, 0.2, 0.3] | Training channel mixture and loss probabilities; evaluation uses the same rate list |
| `seed`, `split_seed` | 17, 2026 | Training randomness and fixed data partition |
| `workers`, `threads` | 0, 4 | DataLoader workers and CPU PyTorch threads |

## Implementation details

**Graph and packet assignment.** Absolute Pearson correlations are computed from batch first/second moments with EMA updates. The sparse graph takes the union of each channel's top-k edges. Channels are visited in descending weighted degree and assigned by within-packet affinity plus a balance cost. Every packet has capacity `ceil(C / M)`, with zero padding. Calibration runs once before recovery training; deployment uses the fixed graph and packet map.

**Parity precision.** Each real parity row is quantized to uint8 using coefficient-derived bounds: `low = 255 sum(min(gamma, 0))` and `scale = sum(abs(gamma))`. The profile determines these scales, and training uses straight-through rounding. At the same `M` and `R`, GraphFEC, RLC, RS, and UEP have equal payload and header budgets. This quantization adds error to the real-valued recovery equations. `parity_precision=float32` retains real parity values and uses four bytes per parity symbol; byte metrics include the increase.

**Recovery.** The coarse decoder solves `(GᵀMG + eta L + ridge I) U = GᵀMY`. Two refinement blocks combine normalized graph messages and reception masks through a depthwise 3 × 3 convolution and a 1 × 1 bottleneck. Received systematic channels keep their quantized values throughout recovery. Evaluation caches the solved operator for each reception mask, up to `cache_size` entries.

**Transport and timing.** Datagrams represent logical packets, which can exceed a network MTU. Network deployments need to account for fragmentation and transport headers. CRC32 checks corruption. The included benchmark measures local computation; energy and end-to-end network performance require measurements on the target devices.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Tests cover correlation estimates, packet assignment and padding, quantization bounds, erasure recovery, training gradients, dataset labels and holdouts, loss-channel statistics, bootstrap intervals, split-tensor shapes, and packet validation. CI runs the tests and the synthetic workflow on Python 3.11 and 3.13.

## Code map

```text
src/graphfec/
  data.py          dataset manifests and holdouts
  models.py        edge/cloud classifiers
  graph.py         correlation statistics and packet mapping
  codec.py         quantization, parity, solve, graph refinement
  gf256.py         finite-field erasure coding
  system.py        GraphFEC variants and baselines
  channels.py      Bernoulli, Gilbert, trace replay
  transport.py     datagram serialization and checks
  training.py      training and calibration
  evaluation.py    metrics, comparisons and local timing
  plotting.py      figures from measured results
  cli.py           command-line entry points
configs/           dataset configurations
tests/             key numerical and workflow assertions
```

Licensed under the MIT License.
