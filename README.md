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
- `hex_full.csv`
- `bitmask_patterns_sliding_window.csv` (if `FILTERS` configured)

## 3) Dataset Balancing

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
