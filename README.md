# Probe-Bamboo-2026

Reference guide for the repository pipeline to process Wi-Fi probe requests and evaluate three methods:
- `BAMBOO` (main focus of this repo)
- `PF` (state-of-the-art baseline)
- `PINTOR` (state-of-the-art baseline)

The code is designed around **PCAP (`.pcap`) captures** (not `.pcapng` in the main pipeline), with a **one-file-per-device** dataset organization.

## Repository Structure

```text
Probe-Bamboo-2026/
├── main.py                         # Main experiment/training/validation/testing pipeline
├── pcap_processing/
│   ├── parse_pcaps.py              # Extracts hex/binary/dissected CSVs from .pcap files
│   ├── concat_pcap.py              # Utility to merge split captures per device
│   ├── config_pcap_processing.ini  # Input/output dataset paths for PCAP parsing
│   └── utils/                      # Low-level extractors, headers, protocol parsing helpers
├── pre_processing/
│   ├── data_preprocess.py          # Builds interim binary/hex datasets + BAMBOO filters
│   ├── dataset_balancer.py         # Balances per-device samples and creates balanced files
│   ├── device_analyzer.py          # Optional exploratory stats/plots
│   ├── config_preprocessing.ini    # Paths for binary/hex input and interim/filter output
│   └── utils/                      # Concatenation, balancing, cleaning, filter generation helpers
├── modules/
│   ├── bamboo/                     # BAMBOO training core and console/logging utilities
│   ├── *_roc_validation.py         # ROC / tau validation for BAMBOO, PF, PINTOR
│   ├── pf_training.py              # PF training (index selection)
│   ├── pair_generator.py           # Pair generation for positive/negative matching
│   ├── compare_results.py          # Cross-method plotting and summary figures
│   └── utils/                      # Validation, fingerprint, clustering utilities
├── data/                           # Parsed and interim datasets
└── results/                        # Per-cycle outputs and metrics
```

## Data Assumptions

1. Input captures are `.pcap` files.
2. Each device is represented by one capture file named as device label, e.g. `iPhone11_B.pcap`.
3. Labeling is filename-based (`Label`/`label` = basename of the file without extension).

If device traffic is split in multiple files, first merge by device with:
- `pcap_processing/concat_pcap.py`

For this project setup, keep final merged files in `.pcap` format and process those.

## Data Access

The dataset used in this project is available at:

- **Project PCAP data link**: https://polimi365-my.sharepoint.com/:f:/r/personal/10692910_polimi_it/Documents/Probe%20Dataset%202026?csf=1&web=1&e=Rrhqr9

After downloading, place the dataset so that each device is represented by one `.pcap` file, then point `pcap_processing/config_pcap_processing.ini` to the correct paths.

## End-to-End Workflow

1. Parse raw PCAPs into CSV features.
2. Build interim datasets (`binary_U_concatenated.csv`, `hex_full.csv`) and filter masks.
3. Balance data and produce training-ready balanced files.
4. Run `main.py` toggling BAMBOO / PF / PINTOR training-validation-testing stages.

## 1) PCAP Parsing

Configure datasets in:
- `pcap_processing/config_pcap_processing.ini`

Each section defines:
- `raw_path`: folder containing `.pcap` files (one file per device)
- `binary_path`, `hex_path`, `dissected_path`: output folders for extracted CSVs

Run:

```bash
python pcap_processing/parse_pcaps.py
```

This produces per-device CSV files in:
- binary features
- hex/dissected features

## 2) Preprocessing and Filter Generation

Configure:
- `pre_processing/config_preprocessing.ini`

Expected paths:
- `binary_path`: parsed binary CSV root
- `hex_path`: parsed hex CSV root
- `output_path`: interim output folder
- `filters_path` (optional `FILTERS` section): where generated BAMBOO bitmasks are saved

Run:

```bash
python pre_processing/data_preprocess.py
```

Outputs:
- `binary_U_concatenated.csv`
- `binary_0_concatenated.csv` (same as above with `U -> 0`)
- `hex_full.csv`
- `bitmask_patterns_sliding_window.csv` (if `FILTERS` configured)

## 3) Dataset Balancing
This process balances the probe for the various classes (devices) and outputs a dataset with the same number of entires for each device, maintaining the probe differentiation ratio (e.g., different probes for the same device have the original ratio in the output). The number of probes per device can be customized in the script (n_entries_per_devices variable)
Run:

```bash
python pre_processing/dataset_balancer.py
```

Outputs in interim folder:
- `binary_U_balanced.csv`
- `binary_0_balanced.csv` (same as above with `U -> 0`)
- `hex_full_balanced.csv`

## 4) Main Training / Validation / Testing (`main.py`)

`main.py` is controlled by boolean flags near the top:
- training: `to_train_bamboo`, `to_train_pf`
- ROC validation: `to_val_bamboo`, `to_val_pf`, `to_val_pintor`
- DBSCAN validation: `to_val_dbscan_bamboo`, `to_val_dbscan_pf`, `to_val_dbscan_pintor`
- clustering test: `to_test_cluster_bamboo`, `to_test_cluster_pf`, `to_test_cluster_pintor`
- cross-cycle summaries: `auc_comparison`, `dbscan_comparison`

Run:

```bash
python main.py
```

### Split Strategy (Device-Level Rotating Folds)

The main process works at **device-label level** (not packet-level random split):

1. Device labels are read from `binary_U_balanced.csv`.
2. Labels are sorted, then shuffled once with `RANDOM_STATE=42`.
3. For each cycle `t` in `range(N_FOLDS)`, `main.py` computes a rotated split with (to adapt depending your data size):
   - `N_TEST = 20` device labels
   - `N_VAL = 20` device labels
   - `N_TRAIN = 23` device labels
4. The start index is rotated by a fixed step (`step=N_TEST`), so across cycles the role of each device changes (test/val/train) in a deterministic way.

This gives a reproducible cross-validation-like procedure with fixed partition sizes per cycle.

### What Happens Inside a Cycle

For each `cycle_t`:

1. The script creates `results/cycle_t/` and saves `labels.txt` with train/val/test label lists.
2. If training flags are enabled:
   - Training pairs from `train_pairs.csv` file are used for BAMBOO training that produces the `bamboo_output.csv` (best filters/thresholds/confidence).
   - PF training produces `pf/pf_indexes_<bits>bits.csv` for 8/16/32/64 bits.
    
3. If ROC validation flags are enabled:
   - It builds or reuses `validation_pairs.csv` from validation labels. This file contains pairs of probe requests generated from the validation devices.
   - BAMBOO writes ROC data + best taus in `bamboo/`.
   - PF writes ROC data + best taus in `pf/`.
   - PINTOR writes ROC data + best tau files in `pintor/`.
4. If DBSCAN validation is enabled, it builds/reuses `val_combinations.csv`, searches best DBSCAN params, and stores them in each method folder. The file contains a list of 10 combinations of device labels from 1 to N size.
5. If cluster test flags are enabled, it builds/reuses `test_combinations.csv` and runs final clustering methods for testing. DBSCAN and/or groupby method is used on test labels for producing the test results.

### Shared Evaluation Design

- **Same validation/test device sets** are used across BAMBOO, PF, and PINTOR in each cycle.
- Pair generation is label-balanced (`N_PAIRS` positives and negatives per label role).
- Method outputs are stored separately under:
  - `results/cycle_t/bamboo/`
  - `results/cycle_t/pf/`
  - `results/cycle_t/pintor/`

This keeps comparisons fair while preserving each method-specific feature construction and distance metric.

Main inputs expected by default:
- `/data/interim/binary_0_balanced.csv`
- `/data/interim/binary_U_balanced.csv`
- `/data/interim/hex_full_balanced.csv`
- `/data/filters/bitmask_patterns_sliding_window.csv`

Results are written per fold/cycle under:
- `/data/results/cycle_<t>/...`

With subfolders:
- `bamboo/`
- `pf/`
- `pintor/`

## Method Coverage

- `BAMBOO`: full training + ROC validation + DBSCAN/groupby clustering evaluation.
- `PF`: baseline training + validation/test evaluation implemented in same pipeline.
- `PINTOR`: baseline feature preparation + ROC and clustering evaluation implemented in same pipeline.

So the repo is BAMBOO-centric, while still containing reproducible methodology and evaluation flow for PF and PINTOR.

## Practical Notes

- Run scripts from repository root (`Probe-Bamboo-2026/`) so relative paths/imports resolve consistently.
- No dependency manifest is included (`requirements.txt`/`pyproject.toml` absent), so environment setup is manual.
- Main third-party packages used: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scapy`, `bitstring`, `rich`, `cowsay`.
