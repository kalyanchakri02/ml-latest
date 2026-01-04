# GPU Fault Remediation Classification (Rules vs ML) — CatBoost PoC

This repository contains a proof-of-concept experiment that compares a **rule-based (SWE) remediation engine** against a **machine-learned classifier (CatBoost)** for predicting **repair actions** on GPU fleet telemetry.

The experiment synthesizes realistic (but synthetic) GPU telemetry signals, defines an independent probabilistic ground truth for “what action should be taken,” and evaluates which approach better predicts that ground truth.

---

## What this code does

### Goal
Predict an **operational remediation action** from heterogeneous telemetry:

| Label | Action |
|---:|---|
| 0 | No Action |
| 1 | Reboot |
| 2 | Reseat |
| 3 | RMA |

### Compared approaches
1. **SWE Rules Baseline (No Training)**  
   Deterministic thresholds and conditions on telemetry (ECC, PCIe width, retimer errors, XID codes, thermal).
2. **ML (CatBoost) using Raw + Engineered Features**  
   A supervised multiclass classifier trained with class imbalance handling and early stopping.

> Why this is meaningful: the **ground truth is not generated from the SWE rules**, so ML is not “learning the rules.” Both systems are evaluated fairly against the same held-out test set.

---

## Experiment design (high level)

### Step 1 — Latent fault domains (hidden)
Each sample is assigned a hidden “fault domain” (not given to the model):
- healthy, software, signal, thermal, silicon  
with realistic class imbalance (e.g., silicon is rare).

### Step 2 — Synthetic telemetry generation
Telemetry signals are generated from the latent domain with noise/overlap to mimic real fleet behavior:
- Thermal: `temp_c`, `fan_rpm`, `is_active_cooling`
- Power: `voltage`
- Interconnect: `pcie_width`, `pcie_gen`, `retimer_errors`
- Reliability: `error_count_24h`, `unfixable_ecc_errors`
- Logs: `xid_code`
- Context: `node_age_days`, `prior_recovery_failures`
- Categorical: `gpu_model`

### Step 3A — Ground truth (synthetic, independent of rules)
The “should-do” action is generated probabilistically from latent domains using `domain_to_action_probs`.
A small amount of label noise is injected to simulate telemetry gaps and operator variance.

Output label: `action_true`

### Step 3B — SWE rule engine baseline
A deterministic rules function predicts an action using thresholds and conditions on telemetry.

Output prediction: `action_rules`

### Step 4 — Feature engineering
Adds derived features intended to increase separability:
- log transforms (error counters, node age)
- thermal headroom, cooling mismatch
- voltage droop severity
- PCIe bandwidth proxy + degradation flags
- silicon risk score & signal integrity score
- XID semantic indicators
- recovery pressure metrics

Output dataset: `df_full`

### Step 5 — Train/Val/Test split
Splits are stratified on `action_true`:
- Train 70%
- Validation 15%
- Test 15%

Rules and ML are both evaluated on the same **test split**.

### Step 6–8 — Train, evaluate, summarize
Evaluates:
- Accuracy
- Balanced Accuracy (important under imbalance)
- Macro F1 (treats classes equally)
- Weighted F1 (frequency-weighted)

Also plots:
- Confusion matrix
- Top feature importances (CatBoost)

---

## Requirements

### Python packages
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `catboost`

Install:
```bash
pip install -U numpy pandas scikit-learn matplotlib catboost
