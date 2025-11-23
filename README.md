# Contract-Level Binary Prediction of Implied Volatility Surfaces Using Transformers (Research Release)

This repository contains the code used in our research on predicting next-day implied volatility movement and magnitude from cross-sectional options data. It includes data-processing utilities, transformer- and LSTM-based models, baselines, and training/evaluation scripts driven by YAML configs. Cite via [CITATION.cff](CITATION.cff); license is [MIT](LICENSE).

## Schema 
- **Task (binary):** predict whether IV will go **up (1)** or **down (0)** the next day.
- **Task (regression):** predict the **absolute IV change |ΔIV|** for the next day.
- **Input:** per-option contract sequences over a window of trading days (shape: `C x T x F`).
- **Targets:**
  - `targets_dir` (binary labels 0/1) for classification configs.
  - `targets_mag` (|ΔIV|) for regression configs.
- **Features:** `Strike`, `DaysToExpiration`, `ImpliedVolatility`, `Delta`, `Gamma`, `Vega`, `OpenInterest`, `Volume`, `SPX`, `LogMoneyness`, `RiskFreeRate`, `ForwardPrice`

As described in the paper, a special dataset is developed: `CrossSectionalDataset`, which enables compatible data operations based on contract counts, using chosen `train_ratio` and `val_ratio` (test receives the remainder).

## Model Variants
Models are built modularly using separate `Backbones` and `Heads` combined via `Wrapper` builders.

### Backbones (`lib/models/backbones.py`)
- **TCTA**: temporal transformer per contract + cross-sectional transformer across contracts;
- **FullTemp TCTA**: uses all temporal embeddings concatenated plus cross-sectional attention.
- **TTA**: temporal-only transformer backbone.
- **LSTM**: temporal LSTM backbone (bi-LSTM by default).

### Heads (`lib/models/heads.py`)
- **BinaryHead**: used for binary classification, of directional movement
- **RegressionHead**: used for regression tasks

### Baselines (`lib/models/baselines_*`)
Beside the models, baseline code is also presented.
- **Baselines (binary):** MajorityTrend, LastDay, BiasedRandom, Random.
- **Baselines (regression):** LastDayAbsChange, MeanAbsChange, ZeroAbsChange, RandomAbsChange.


## Quickstart
Prereqs: Python 3.10+, PyTorch (CUDA optional but recommended), numpy/pandas/sklearn/matplotlib/pyyaml/tqdm.

1) **Train** (binary example)
```bash
python -m scripts.train --config configs/tcta_binary.yaml
```

2) **Evaluate** (uses `best_model.pt` saved per split)
```bash
python -m scripts.evaluate --config configs/tcta_binary.yaml
```

3) **Baselines (binary)**
```bash
python -m scripts.evaluate_baselines_binary
```

4) **Baselines (regression)**
```bash
python -m scripts.evaluate_baselines_reg
```

See `configs/` for ready YAMLs:
- Binary: `tcta_binary.yaml`, `tta_binary.yaml`, `lstm_binary.yaml`, `fulltemp_tcta_binary.yaml`
- Regression: `tcta_reg.yaml`, `tta_reg.yaml`, `lstm_reg.yaml`, `fulltemp_tcta_reg.yaml`

## Configuration Keys (YAML)
- `seed`: global seed.
- `data.dataset_entries_dir`: path to `.npz` entries with `sequences`, targets, `log_moneyness_ref`, `feature_columns`.
- `output.root_dir`: where models/plots/metrics are written.
- `task`: `{type: binary|regression, model_builder: <registry key>, target_key: targets_dir|targets_mag}`.
- `splits`: list of `{train_ratio, val_ratio}`; each defines a CV split.
- `model`: hyperparams passed to builder (`d_model`, `nhead`, `num_layers`, `dropout` and specialized parameters like `bidirectional` for lstm or `seq_length` for fulltemp).
- `training`: batch sizes, epochs, patience, LR, scheduler.

## Package & Entry Points
The code is importable as a library via `lib/`. Key public objects are exported from `lib/__init__.py` (datasets, trainers, evaluators, builders, backbones, heads, baselines).

## Data
CBOE options data is proprietary and cannot be redistributed.  
You must provide your own licensed dataset.

Although we can not release the data, we provide the preprocessing scripts used within our paper experiments: `processing/`.

## License
This project is released under the [MIT License](LICENSE).

## Citation
If you use this code, please cite the accompanying paper (metadata in [CITATION.cff](CITATION.cff)).
