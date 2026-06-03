# SAE-XCrash

**Calibrated and Explainable Gradient Boosting for Road Traffic Crash Severity Prediction:
SHAP Audit and Cross-Jurisdiction Transfer Evaluation**

> Mohammad Alhawarat · Ahmad Alkhatib · Qasem Nijem
> Al-Ahliyya Amman University, Amman, Jordan
> *Submitted to Applied Sciences (MDPI), 2026*

---

## Overview

SAE-XCrash is the companion code repository for the paper above. It addresses three long-standing gaps in ML-based crash severity prediction simultaneously:

1. **Uncalibrated probability scores** — post hoc isotonic calibration reduces ECE by 97.3% (0.2394 → 0.0065) with negligible discrimination loss (ΔAUPRC = −0.0035).
2. **Unaudited SHAP explanations** — a four-step quantitative XAI audit (sanity check, deletion faithfulness, insertion faithfulness, perturbation stability) adapted from computer vision to the tabular imbalanced safety-prediction setting.
3. **Untested cross-jurisdiction generalizability** — the first peer-reviewed systematic US→UK cross-dataset transfer study on crash severity, using a controlled three-tier design (zero-shot, recalibrated, retrained).

---

## Repository Structure

```
sae-xcrash/
├── code/                          # Jupyter notebooks (NB01–NB07)
│   ├── NB01_Data_Protocol.ipynb       # Data loading, preprocessing, split indices
│   ├── NB02_Modeling_Baselines.ipynb  # Model training and benchmarking
│   ├── NB03_Calibration.ipynb         # Probability calibration and metrics
│   ├── NB04_XAI_Audit.ipynb           # Four-step SHAP audit
│   ├── NB05_Transfer_STATS19.ipynb    # US→UK cross-jurisdiction transfer
│   ├── NB06_SOTA_Comparison.ipynb     # CatBoost and SOTA baselines
│   └── NB07_Reduced_Transfer.ipynb    # Reduced feature zero-shot transfer
├── models/
│   ├── calibrators/               # Isotonic calibrators trained on US-Accidents
│   └── calibrators_s19/           # Isotonic calibrators trained on UK STATS19
├── split_indices/                 # Published temporal split row indices
├── environment.yml                # Conda environment (Python 3.12)
├── experiment_config.yaml         # All seeds, hyperparameters, and audit thresholds
└── README.md
```

---

## Datasets

| Dataset | Records | License | Source |
|---|---|---|---|
| US-Accidents (Moosavi et al., 2019) | ~7.0 M (after filtering) | CC BY 4.0 | https://smoosavi.org/datasets/us_accidents |
| UK STATS19 (Dept. for Transport) | ~1,010,000 (2016–2022) | Open Government Licence v3.0 | https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data |

> **Important note on US-Accidents severity field:** All studies using temporal splits must account for the Severity 3 reclassification that began in 2019. This repository uses Severity 4 only as the positive class. Studies using the conventional {3, 4}→Severe mapping show training prevalence of ~24.7% but 2023 test-year prevalence of just 2.9% — a data quality issue not described in the peer-reviewed literature.

### Temporal Splits

| Dataset | Train | Validation | Test |
|---|---|---|---|
| US-Accidents | 2016–2021 (5,549,870 rows) | 2022 (1,268,806 rows) | 2023 (166,552 rows) |
| UK STATS19 | 2016–2020 (699,060 rows) | 2021 (106,004 rows) | 2022 (205,185 rows) |

All split row indices are published in `split_indices/` for exact reproducibility.

---

## Key Results

### Primary Benchmark — US-Accidents (Test Year 2023)

| Model | AUROC | AUPRC | ECE (uncal.) | ECE (cal.) | Brier (cal.) |
|---|---|---|---|---|---|
| Logistic Regression | 0.7081 | 0.0584 | 0.3327 | 0.0055 | 0.0279 |
| Random Forest | 0.7686 | 0.0838 | 0.3480 | 0.0059 | 0.0275 |
| **XGBoost** | **0.8215** | **0.1306** | 0.2394 | **0.0065** | **0.0266** |
| LightGBM | 0.8193 | 0.1311 | 0.2226 | 0.0065 | 0.0267 |
| MLP | 0.7974 | 0.1097 | 0.0414 | 0.0043 | 0.0271 |

Primary metric: AUPRC (no-skill baseline = 0.027; models achieve ~5× lift).
All AUPRC values reported with 95% bootstrap CIs (1,000 replications).

### XAI Audit Summary — XGBoost, US-Accidents

| Audit Step | Result | Threshold | Outcome |
|---|---|---|---|
| Sanity check (JSD, top-10 features) | 0.6266 | > 0.20 | PASS |
| Deletion faithfulness (per-k gap) | SHAP > Random at all k | > 0.05/step | PASS |
| Insertion faithfulness (per-k gap) | SHAP > Random at all k | > 0.05/step | PASS |
| Stability (σ = 0.01) | Mean cos = 0.731, low-stability = 34.9% | < 40% | PASS |
| Stability (σ = 0.05) | Low-stability = 54.3% | < 40% | FAIL |

SHAP explanations are reliable at σ = 0.01 but degrade at realistic noise levels (σ = 0.05).

### Cross-Jurisdiction Transfer — UK STATS19 (Test Year 2022)

| Configuration | AUROC | AUPRC | ECE (cal.) |
|---|---|---|---|
| XGBoost zero-shot (US model, 22 harmonized features) | 0.5182 | 0.2562 | 0.0688 |
| Recalibrated zero-shot (STATS19 isotonic) | 0.5182 | 0.2562 | 0.0088 |
| XGBoost retrained on STATS19 | 0.6432 | 0.3608 | 0.0115 |
| LightGBM retrained on STATS19 | 0.6514 | 0.3692 | 0.0099 |

Zero-shot transfer fails (AUROC ≈ random) due to spatial memorization — the dominant US features (geohash5, state encoding) have no UK equivalent. Temporal features transfer robustly: Spearman ρ = 0.810 (p = 0.015) across 8 common features.

---

## Reproducibility

All experiments use **strictly temporal splits** — no random train/test shuffling. Published split indices in `split_indices/` allow exact row-level reproduction. All random seeds are fixed at `seed = 42`.

### Setup

```bash
git clone https://github.com/malhawarat/sae-xcrash.git
cd sae-xcrash
conda env create -f environment.yml
conda activate sae-xcrash
```

### Environment

Key dependencies (from `environment.yml`):

| Package | Version |
|---|---|
| Python | 3.12 |
| XGBoost | 3.2.0 |
| LightGBM | 4.6.0 |
| SHAP | 0.51.0 |
| pandas | 2.2.2 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| Optuna | 4.9.0 |
| CatBoost | 1.2.10 |
| scipy | 1.16.3 |

All experiments were run on Google Colab with an NVIDIA A100 GPU (40 GB HBM2).
Training time: ~18 min for LightGBM on the full US-Accidents dataset; the XAI stability audit on the 5,000-instance subsample takes ~10 min.

### Configuration

All hyperparameters, audit thresholds, split definitions, and seeds are documented in `experiment_config.yaml`. Key settings:

- Global seed: `42`
- Optuna trials: `100` (US-Accidents), `50` (STATS19)
- SHAP subsample: `50,000` (stratified)
- Stability perturbations: `10` per instance, σ = `0.01` × feature std
- Bootstrap replications: `1,000` (95% CI)

---

## Trained Model Artifacts

Pre-trained calibrators are available in `models/`:

- `models/calibrators/` — isotonic regression calibrators trained on the US-Accidents 2022 validation window (one per model: XGBoost, LightGBM, Random Forest, Logistic Regression, MLP)
- `models/calibrators_s19/` — isotonic regression calibrators trained on the UK STATS19 2021 validation window

Base models (XGBoost: 38.2 MB, LightGBM: 49.7 MB) are too large for GitHub. The notebooks load them directly from Google Drive.

---

## Citation

If you use this code or results from this paper, please cite:

```bibtex
@article{alhawarat2026saexcrash,
  title     = {Calibrated and Explainable Gradient Boosting for Road Traffic Crash
               Severity Prediction: {SHAP} Audit and Cross-Jurisdiction Transfer Evaluation},
  author    = {Alhawarat, Mohammad and Alkhatib, Ahmad and Nijem, Qasem},
  journal   = {Applied Sciences},
  publisher = {MDPI},
  year      = {2026},
  note      = {Submitted. Code: https://github.com/malhawarat/sae-xcrash}
}
```

> This entry will be updated with volume, issue, pages, and DOI upon formal publication.

---

## License

This repository is licensed under the [MIT License](LICENSE).

The datasets used in this study are subject to their own licenses:
- US-Accidents: [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- UK STATS19: [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

---

## Authors

| Name | Role | Affiliation | Email |
|---|---|---|---|
| Mohammad Alhawarat *(corresponding)* | Methodology, Software, Writing | Dept. of Data Science & AI, Faculty of IT, Al-Ahliyya Amman University | m.hawarat@ammanu.edu.jo |
| Ahmad Alkhatib | Conceptualization, Validation, Data Curation | Dept. of Software Engineering, Faculty of IT, Al-Ahliyya Amman University | a.khatib@ammanu.edu.jo |
| Qasem Nijem | Writing — Review & Editing | Dept. of Software Engineering, Faculty of IT, Al-Ahliyya Amman University | q.nijem@ammanu.edu.jo |
