# GraphFEC

[![CI](https://github.com/Hqa77880011/GraphFEC/actions/workflows/ci.yml/badge.svg)](https://github.com/Hqa77880011/GraphFEC/actions/workflows/ci.yml)

PyTorch implementation of **GraphFEC: Correlation-Graph-Coded Activation Transmission for Robust Edge-Cloud Split Inference**.

GraphFEC protects the intermediate activation of a split classifier. A channel correlation graph guides balanced packet assignment and sparse parity coding. The receiver combines a regularized linear solve with two graph refinement blocks before running the cloud classifier.

The repository includes dataset preparation, clean-model training, calibration, recovery training, baseline and ablation comparisons, packet serialization, evaluation, and plotting. Start with the small CPU workflow below, then follow the numbered sections for an experiment on real data.

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
python -m pip install -e ".[dev]"
pytest -q
graphfec smoke --output runs/smoke
```

The smoke command uses generated color-pattern images and a tiny CNN. It runs one clean epoch, one recovery epoch and one fine-tuning epoch per variant, then exercises every baseline, evaluation, packet roundtrip, local timing, and plotting. Its outputs are functional-check data, not classification benchmark results. No dataset download is needed. CI runs this workflow on Python 3.11 and 3.13.

`python -m graphfec` is equivalent to `graphfec`. All commands below run from the repository root. Outputs are written under `runs/` and excluded from Git.

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

You can also extract the archive into `data/tiny-imagenet-200/` yourself and omit `--download`. The loader reads `train/<wnid>/images/`, `val/images/`, and `val/val_annotations.txt` directly; it does not move images.

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

For all datasets, a fixed, class-stratified 10% holdout of the original training set is used for validation. CIFAR's original test set and the original labeled validation sets of Tiny ImageNet/ImageNet are used as test data. Graph and quantization calibration use only the remaining training partition. Gradient saliency for UEP uses the holdout. The split seed is independent of the model seed and stays fixed across methods.

## 2. Train the clean classifier

```bash
graphfec train-clean --config configs/cifar100.json
```

This trains from scratch with SGD, momentum 0.9 and cosine decay, and saves the best validation checkpoint as `runs/cifar100/clean.pt`. The history is in `clean_history.csv`. The CIFAR ResNet uses a 3 × 3 stride-one stem without max pooling. Training uses random crops and horizontal flips; evaluation uses deterministic resizing/center cropping. All image datasets use ImageNet channel normalization.

Use the other configuration paths for their corresponding dataset and backbone. `device=auto` selects CUDA when available; `--set device=cpu` forces CPU. ConvNeXt defaults to batch size 32 to limit memory use. The full schedules are intended for a GPU; the smoke command is the quick CPU check.

## 3. Calibrate the protection profile

```bash
graphfec calibrate --config configs/cifar100.json
```

This computes global-average-pooled channel statistics, EMA and full-sample absolute Pearson correlations, channel minima/maxima/means, and validation gradient saliency. The result is `profile.pt`, which also contains the clean classifier. Calibration selects at most 4,096 training examples and 1,024 validation examples, deterministically and without augmentation.

The sparse graph keeps the strongest `neighbors` edges per channel and uses the union of both directions. The interleaver visits channels by descending weighted degree, minimizes within-packet affinity plus a balance term, and pads each packet to `ceil(C / M)` channels. Packet-graph neighborhoods initialize the support of each parity row; the coefficients are learned with unit L2 row norm.

## 4. Train GraphFEC

```bash
graphfec train-recovery --config configs/cifar100.json --variants graphfec
```

The clean backbone stays frozen for 120 epochs. The final cloud feature stage and classifier are then tuned with the recovery module for 20 epochs. AdamW uses learning rate `3e-4`, weight decay `1e-4`, and cosine decay across both phases. Each mini-batch samples a loss rate and either Bernoulli or Gilbert erasures. The objective combines cross-entropy, activation L1, cosine consistency, and temperature-scaled KL distillation from the fixed clean classifier.

Validation uses fixed masks at 10% Bernoulli loss. The best checkpoint is `graphfec.pt`; `graphfec_history.csv` records the phase, training loss, and validation accuracy. Checkpoints contain the classifier, graph statistics, quantizer, packet map, parity weights, refiner weights, configuration, and a profile UUID.

## 5. Execute a protected inference

```bash
graphfec infer --config configs/cifar100.json --loss 0.1 --channel gilbert
```

This selects one test image, runs the edge, serializes the packets, applies erasures, parses surviving packets, restores the activation, and runs the cloud. To use your own image, add `--image path/to/image.jpg`. `--method rs` or another method selects a baseline.

Files in `runs/cifar100/inference/`:

| File | Contents |
|---|---|
| `prediction.json` | Prediction, clean prediction, label when available, lost packets, bytes and MAE |
| `packets.bin` | Transmitted datagrams, each preceded by a four-byte little-endian length |
| `activations.npz` | Clean and recovered activations and the received-packet mask |

The packet API is in `graphfec.transport`. Each datagram has a 42-byte header containing format version, profile UUID, sample ID, packet index/count, activation shape, payload length, and CRC32. The file's length prefixes are container metadata and are not included in reported datagram bytes. Parsing rejects mixed profiles, mismatched samples/shapes, duplicate indices and corrupt payloads. The included execution path is local; networking is managed by the application using these datagrams.

## 6. Compare baselines and evaluate

```bash
graphfec evaluate --config configs/cifar100.json --methods graphfec,none,rs,rlc,uep --trace-seeds 10017,10029,10043
graphfec plot --input runs/cifar100/evaluation_test/summary.csv --output runs/cifar100/figures
```

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

Accuracy is stored as a fraction. MAE and cosine compare recovered activations with the unquantized clean activation. A no-loss MAE therefore includes quantization error. `clean_accuracy` and `quantized_accuracy` use the same cloud suffix as that row's method; fine-tuned variants can have different clean references. Bootstrap intervals describe example uncertainty conditional on a trained checkpoint and trace, not uncertainty across training seeds. The plots average the supplied run/trace rows and do not invent confidence bands.

Use `--split val` for parameter selection; keep test data for the final evaluation. `--limit 100 --bootstrap 200` is useful for a short functional run, not a final accuracy estimate. No results or model weights are prefilled in the repository.

For a recorded loss trace:

```bash
graphfec evaluate --config configs/cifar100.json --methods graphfec,rs --trace path/to/erasures.csv
```

Provide headerless values `0` (received) and `1` (erased), separated by commas or, for a `.txt` file, whitespace. The trace needs at least `number_of_examples × (M+R)` events, or `number_of_examples × 2M` when duplication is included. It is consumed in example-major packet order without repetition or loss-rate rescaling; the actual observed loss is reported. No mobile/Wi-Fi trace is bundled.

## 7. Run ablations and sensitivity studies

```bash
graphfec train-recovery --config configs/cifar100.json --variants all
graphfec evaluate --config configs/cifar100.json --methods all
graphfec plot --input runs/cifar100/evaluation_test/summary.csv --output runs/cifar100/figures
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

`all` includes GraphFEC and every listed variant; the evaluation also includes `none`, `rs`, `rlc`, and `uep`. Missing trained checkpoints are an error. The no-parity ablation transmits fewer bytes; duplication transmits more. Their actual budgets are reported rather than represented as matched-budget comparisons.

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

Energy and network throughput require measurements on the target hardware. This code does not infer energy or claim the paper's Jetson/RTX testbed timings from local compute measurements.

## Important configuration fields

Every command accepts repeated `--set key=value` overrides. Strings may be unquoted; lists use JSON, e.g. `--set 'loss_rates=[0.0,0.1,0.2]'`. Unknown fields are rejected. Full defaults are in `src/graphfec/config.py`.

| Field | Default | Meaning |
|---|---|---|
| `packets`, `parity` | 20, 4 | Data and parity packets; redundancy is `R/M` |
| `neighbors`, `beta` | 8, 0.95 | Per-channel top-k graph and EMA coefficient |
| `balance`, `sparsity` | 0.1, 8 | Interleaver balance cost and max nonzeros per parity row |
| `eta`, `ridge` | 0.001, 0.000001 | Packet Laplacian weight and numerical ridge |
| `bottleneck`, `cache_size` | 16, 256 | Refiner hidden width and bounded reception-mask operator cache |
| `parity_precision` | `uint8` | `uint8` for equal payload bytes; `float32` for real-parity experiments |
| `lambda_l1`, `lambda_cos`, `lambda_kd` | 1, 1, 1 | Auxiliary loss weights |
| `temperature` | 2 | Distillation temperature |
| `calibration_samples`, `saliency_samples` | 4096, 1024 | Calibration and validation-saliency limits |
| `seed`, `split_seed` | 17, 2026 | Training randomness and fixed data partition |
| `workers`, `threads` | 0, 4 | DataLoader workers and CPU PyTorch threads |

## Implementation choices

These choices fill details not fixed by the paper and affect interpretation of results:

- **Byte budget.** The real-valued parity equations alone do not specify an eight-bit packet representation. The default implementation quantizes each parity row to uint8 using bounds derived from its coefficients: `low = 255 sum(min(gamma, 0))`, `scale = sum(abs(gamma))`. These scales are fixed by the profile and need no per-sample metadata. Straight-through rounding is used during training. GraphFEC, RLC, RS and UEP then transmit the same data/parity payload bytes and 42-byte headers. Parity quantization introduces noise, so real-valued full-rank exact recovery is a property of the unquantized solver, not a claim about uint8 parity. `parity_precision=float32` removes this rounding but spends four bytes per real parity symbol, and is reported with its larger byte budget.
- **Recovery.** Coarse decoding solves `(GᵀMG + eta L + ridge I) U = GᵀMY`. Received systematic channels are restored exactly to their quantized values after both coarse recovery and refinement. Each refinement block concatenates normalized graph messages with channel-reception masks, then applies a depthwise 3 × 3 convolution and a two-layer 1 × 1 bottleneck. Graph aggregation follows stored sparse edges. The evaluation cache stores the solved linear operator for each mask, bounded by `cache_size`; it is cleared when training mode changes.
- **Calibration.** EMA updates are applied to batch first/second moments, not to sample-specific deployment graphs. The edge backbone is frozen after clean training, so calibration and the interleaver run once before recovery training. The `static_graph` variant uses exact aggregate calibration moments; all deployed graphs are fixed. Per-channel min/max ranges, balance weight, loss weights, bottleneck size and clean-training schedules are explicit implementation conventions.
- **Baselines and neural-only recovery.** UEP implements the saliency-allocation principle with two separately coded groups. Random linear coding uses real Gaussian coefficients. Neural-only uses normalized backprojection of all received equations as its initializer; it does not solve the normal equations. These definitions make the executable comparisons precise.
- **Data and measurements.** ImageNet-100 has an explicit selectable class list because the paper does not supply one. The trace reader preserves measured events rather than rescaling them. Local logical packets can exceed a network MTU; a network application must choose packetization/fragmentation for its transport and count the resulting headers. CRC32 detects corruption; it does not provide authentication or encryption.

Full real-data training, accuracy comparisons, and hardware/network experiments are left to the commands above. The repository's automated validation covers numerical invariants, actual split shapes, and the complete small synthetic workflow.

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
