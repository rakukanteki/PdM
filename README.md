# DR-AAD

Modular Python implementation of **Descriptor-Guided Relational Representation Learning for Acoustic Machine Condition Monitoring**, based on the author's final manuscript supplied on October 31, 2026.

**Implementation status:** this repository was written from the manuscript after its experiments. It is not the recovered historical experiment code. No original checkpoints, split manifests, or prediction files were supplied. The code computes results from audio; it does not contain the paper's result tables as executable outputs. Matching the reported numbers requires independent experiments and verification. See [implementation decisions](docs/IMPLEMENTATION_NOTES.md) before describing this repository as a reproduction.

## What is implemented

- Explicit 16-kHz waveform preparation, log-Mel patches, and the 11 Table I descriptors.
- Residual-SE acoustic encoder, descriptor fusion, and **recording-level reconstruction from acoustic embeddings**, without a direct descriptor-input shortcut.
- Positive-weight within-ID cosine relational alignment on distinct recording bags.
- Normal-only ID classification with mixup, masking, Gaussian augmentation, and early stopping.
- Known-ID and leave-one-ID-out folds with fitting-only normalization and reference estimation.
- Clean normal-only component-rank calibration, two-member ensemble scores, disagreement, and threshold decisions.
- Clean, 20-dB noise, 10-dB noise, gain changes, and low-pass evaluation with fixed clean calibration.
- Unrounded per-recording predictions, per-ID metrics, category macros, review evaluation, and documented timing.
- Three additional normal-only descriptor controls: Isolation Forest, one-class SVM, and nearest-neighbor distance. These are newly specified controls, not the paper's undocumented conventional baselines or reproductions of external neural methods.

## Installation

Use Python 3.10 or newer in a fresh environment. From this directory:

```console
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows; `source .venv/bin/activate` on Linux/macOS), then:

```console
python -m pip install --upgrade pip
python -m pip install -e .
python -m dr_aad --help
```

For a CUDA installation, install a PyTorch build suitable for your GPU/driver before installing this package. `--device cpu` works without a GPU; examples use `cuda` for training speed. All run directories record actual dependency versions and device information. [requirements-tested.txt](requirements-tested.txt) lists the direct dependency versions used for the local smoke tests, not the historical manuscript environment.

## Dataset and split manifest

Obtain the DCASE 2020 Task 2 development audio separately and retain a directory layout such as:

```text
data/dcase2020/
  ToyCar/
    train/normal_id_01_00000000.wav
    test/normal_id_01_00000000.wav
    test/anomaly_id_01_00000000.wav
  ToyConveyor/
  fan/
  pump/
  slider/
  valve/
```

The scanner supports additional parent directories and reads the category from the directory immediately preceding `train` or `test`. It recognizes `normal_id_XX_...wav` and `anomaly_id_XX_...wav`. All training-folder audio must be normal. ID strings retain leading zeros. Dataset audio is excluded by `.gitignore` and is not redistributed.

```console
python -m dr_aad make-manifest --root data/dcase2020 --output data/manifest.csv --seed 42
```

This creates a **new** seeded recording split: approximately 80% fitting, 10% validation, and 10% calibration from each ID's normal training pool; the provided test directory remains the test pool. For small pools, validation has at least one recording, calibration at least two, and fitting at least two. The command rejects byte-identical duplicates. Split sizes and audio SHA-256 hashes are saved. These ratios are implementation choices, not known manuscript split sizes.

An existing manifest may be supplied instead. Required columns are `recording_id,path,category,machine_id,split,label,audio_sha256`. `split` is `fit`, `validation`, `calibration`, or `test`; `label` is 0 for normal and 1 for anomaly. Paths are relative to `--root`. Keep this manifest with your released results.

## Train and evaluate one category

```console
python -m dr_aad train --root data/dcase2020 --manifest data/manifest.csv --cache cache --category ToyCar --protocol known --variant full --config configs/default.json --output runs/ToyCar/known/full --device cuda
python -m dr_aad evaluate --run runs/ToyCar/known/full --root data/dcase2020 --cache cache --output outputs/ToyCar/known/full --conditions clean noise20 noise10 --device cuda
```

Default settings include 300 maximum epochs, patience 20, patch batches 128/256, and seeds 42 and 1337. The validation criterion is normal-recording ID cross-entropy; anomaly labels are not used to select models. Each seed's best model is selected independently.

Training saves `run.json`, `manifest.csv`, `normalization.npz`, per-seed checkpoint/selection files, and per-epoch loss and valid-pair logs. Every run/evaluation needs a new output directory to prevent accidental mixing. Feature-cache contents are keyed by processing configuration and condition and checked against source audio hashes.

For unseen ID `01`:

```console
python -m dr_aad train --root data/dcase2020 --manifest data/manifest.csv --category ToyCar --protocol unseen --heldout 01 --variant full --output runs/ToyCar/unseen_01/full --device cuda
python -m dr_aad evaluate --run runs/ToyCar/unseen_01/full --root data/dcase2020 --output outputs/ToyCar/unseen_01/full --device cuda
```

The held-out ID is excluded from fitting, validation, calibration, normalization, and normal reference estimation. Its test recordings are scored against the seen IDs. Repeat for every ID, using the same manifest.

## Ablations and seed variability

| Variant | Descriptor fusion | Acoustic reconstruction | Within-ID relations |
|---|---|---|---|
| `cnn` | No | No | No |
| `fusion` | Yes | No | No |
| `relational` | Yes | No | Yes |
| `full` | Yes | Yes | Yes |

These follow the cumulative components described in the final PDF. `relational` uses the same fusion pathway as `full`, without reconstruction. They differ from the older draft's fusion-plus-reconstruction comparator.

Train five independent models with `--seeds 42 1337 2024 7 314`. Evaluate each independently using `evaluate --seeds 42`, etc., with a different evaluation directory per seed. Compute the five-seed mean and sample SD of **category-macro** metrics. Use `evaluate --seeds 42 1337` for the separate two-member ensemble. Omitting `--seeds` evaluates all checkpoints in the run as one ensemble; five models do not automatically produce five single-model results.

To schedule all categories, protocols, and variants sequentially:

```console
python scripts/run_experiments.py --root data/dcase2020 --manifest data/manifest.csv --output experiment --device cuda
```

This is a potentially long full training schedule. It does not run during installation or testing. Use its `--help` to restrict categories, variants, protocols, or epochs.

## Metrics, review, and prediction

Evaluation outputs `predictions.csv`, `metrics.json`, `evaluation.json`, and a reusable `calibration.npz`. Binary metrics use `score > threshold` before review; the displayed three-way decision gives review precedence. AUC is threshold-free. pAUC is scikit-learn's standardized partial ROC area at maximum FPR 0.10. Single-class subsets yield undefined metrics (`null` in JSON); they are not silently discarded from macros.

Aggregate evaluations from different categories/folds using their full-precision predictions:

```console
python -m dr_aad summarize --predictions outputs/ToyCar/known/full/predictions.csv outputs/fan/known/full/predictions.csv --output outputs/summary.json
python -m dr_aad review --predictions outputs/ToyCar/known/full/predictions.csv --condition clean --coverage 0.90 --pool global --output outputs/review.json
```

`summarize` averages IDs within categories and then averages the categories present. It reports their number; a subset is not a six-category macro. Review uses `floor(N*(1-coverage))` referrals, increasing absolute score margin or decreasing disagreement, and recording-ID tie order. `--pool id` instead ranks within each ID and reports an ID macro. These review definitions are newly explicit and are not verified against the historical table.

After clean calibration has been saved, predict on a WAV without the training dataset:

```console
python -m dr_aad predict --run runs/ToyCar/known/full --calibration outputs/ToyCar/known/full/calibration.npz --audio sample.wav --machine-id 01 --output outputs/sample.json --device cpu
```

For unseen-ID run models, omit `--machine-id`. Share the run's checkpoint files, `run.json`, `normalization.npz`, and its matching `calibration.npz` to enable inference. Changing model weights requires rebuilding calibration.

## Baseline controls and timing

```console
python -m dr_aad baseline --root data/dcase2020 --manifest data/manifest.csv --category ToyCar --method iforest --output outputs/ToyCar/descriptor_iforest
python -m dr_aad benchmark --run runs/ToyCar/known/full --output outputs/timing.json --device cuda --warmup 10 --repetitions 100 --batch-size 32
```

The benchmark uses synthetic, normalized, device-resident patches and measures sequential ensemble forward passes, probability calculation, and recording pooling. It excludes feature extraction, transfer, and calibration. Raw timings, warm-up, synchronization, memory statistic, and hardware details are recorded. Do not label these measurements end-to-end audio latency.

## Tests

```console
python -m unittest discover -s tests -v
```

Tests cover Table II shapes, exact 313-frame patch construction, descriptor edge cases, SNR, detached relational targets and pair counts, reconstruction isolation, calibration ties, metrics, macro weighting, held-out exclusion, and synthetic known/unseen train-to-inference workflows. The integration test uses a tiny encoder, short generated waveforms, one epoch, and one or two seeds; it verifies execution, not research performance. No dataset download occurs. See [local validation](docs/VALIDATION.md) for the checks performed on this release.

## Source map

```text
src/dr_aad/
  audio.py          waveform IO, resampling, perturbations
  descriptors.py    11 complete-recording descriptors
  features.py       log-Mel, patches, normalization, cache
  data.py           manifests, folds, patch and recording sampling
  model.py          residual-SE encoder and heads
  losses.py         augmentation, mixup, pooling, relational loss
  training.py       normal-only optimization and model selection
  scoring.py        Mahalanobis components and rank calibration
  inference.py      fitted detector and single-file prediction
  evaluation.py     perturbation evaluation and aggregation
  metrics.py        detection and review metrics
  baselines.py      newly specified normal-only descriptor controls
  benchmark.py      documented model-only latency measurement
  config.py         validated serializable configuration
  cli.py            command-line interface
```

## Sharing and citation

See [GitHub and manuscript code availability](docs/CODE_AVAILABILITY.md). `CITATION.cff` provides software author metadata without inventing a repository URL or DOI. Do not upload dataset audio, local caches, or experimental weights without reviewing their release terms. A software license must be selected by the authors before representing the repository as open-source licensed software.
