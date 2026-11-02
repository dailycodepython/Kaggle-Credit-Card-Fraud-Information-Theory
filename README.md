# Kaggle-Credit-Card-Fraud-Information-Theory
Production-ready, structurally robust machine learning pipeline for Kaggle S6E11 using Shannon Entropy, Lorentz geometric damping, and relativistic physics analogies with an optimized Hill-Climbing GBDT Triple Stack (XGBoost, LightGBM, CatBoost) to eliminate leaderboard overfitting

This repository contains a high-performance, structurally robust Machine Learning pipeline designed for the **Kaggle Playground Series (Season 6, Episode 11)** competition: *Predicting Credit Card Fraud*.

The core architecture builds upon information theory and physical analogies, completely avoiding standard polynomial feature brute-forcing. Instead, it extracts signal from fraud transaction dynamics using **Shannon Entropy, Relativistic Physics analogies, and Lorentz geometric damping**, stabilized via an optimized Triple GBDT Stacking architecture.

---


## 🏗️ System Architecture & Data Flow

```text
[Raw Transaction Data]
   │
   ├──► Lorentz Amount Damping ──► [lorentz_compressed_amount] ┐
   │                                                            │
   ├──► Relativistic Time Scale ──► [velocity_damping_factor]  ├─► [Feature Interactions & Synergy]
   │                                                            │
   └──► Behavioral Quantization ──► [shannon_entropy_behavior] ┘
                                                │
                                                ▼
                                  [X_train_final / X_test_final]
                                                │
                                 ┌──────────────┼──────────────┐
                                 ▼              ▼              ▼
                            [LightGBM]     [CatBoost]      [XGBoost]
                            (Leaf-wise)    (Symmetric)    (Level-wise)
                                 │              │              │
                                 └──────────────┼──────────────┘
                                                ▼
                                  [Out-of-Fold (OOF) Matrices]
                                                │
                                                ▼
                                  [Masaya Hill Climb Optimizer]
                                                │
                                                ▼
                         [submission_shannon_entropy_non_linear_...]
```
---

## 🔬 Mathematical Engineering Framework

### 1. Geometric Shannon Entropy (`shannon_entropy_behavior`)
To capture systemic anomalies and behavioral chaos, continuous behavioral features (`txn_count_last_24h`, `distance_from_home_km`, `account_balance_usd`, etc.) are standardized and dynamically mapped onto a 10-bin localized coordinate space. Row-wise Shannon Entropy is calculated to expose structural deviation:

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

When a fraudulent operator breaks standard user interaction patterns, the localized configuration chaos triggers an explicit spike in the structural entropy metric.

### 2. Lorentz Financial Amount Damping (`lorentz_compressed_amount`)
Standard log transforms (`log1p`) maintain a relatively linear sensitivity to massive scale variances. To secure the trees from overfitting on extreme transaction volume anomalies in the private test partition, amounts are processed via a relativistic Lorentz attenuation function:

$$f(x) = \frac{x}{\sqrt{1 + \left(\frac{x}{x_{\text{median}}}\right)^2}}$$

### 3. Relativistic Velocity Factor (`velocity_damping_factor`)
Fraud velocity attacks (rapid consecutive card testing) scale non-linearly. The temporal intervals (`hours_since_last_txn`) are mapped onto a damping factor that squashes immediate, tight bursts closely to $1.0$, while naturally decaying typical human operational delays toward $0.0$:

$$g(t) = \frac{1}{\sqrt{1 + \frac{t_{\text{minutes}}}{60}}}$$

---

## 📈 Optimized Ensemble Recipe

Rather than relying on uncalibrated, intuitive blending ratios (e.g., $0.4 / 0.3 / 0.3$), the pipeline implements a localized forward-selection **Hill Climbing Blending Engine** (Masaya Kawamata style) strictly calculated across out-of-fold validation spaces. 

### Final Optimization Blueprint:
* **Validation Baseline (Equal Shares):** `0.946233` AUC
* **Post-Optimization Master Cross-Validation:** **`0.946310` AUC**

### Meta-Model Weight Allocations:
* 🔹 **XGBoost** (Level-wise, heavily regularized L1/L2): **`0.4837`**
* 🔹 **LightGBM** (Leaf-wise, high-exposure localization): **`0.3400`**
* 🔹 **CatBoost** (Symmetric-tree, native category encodings): **`0.1763`**

---

## ⚙️ How to Execute

1. Place your native `train.csv` and `test.csv` in the root working directory.
2. Execute the notebook `submission_shannon_entropy_non_linear_scaled_triple_stack.ipynb` sequentially.
3. The script will automatically parse behavioral features, execute high-speed GPU training across 5 stratified folds, trigger optimal blending parameters, and export the deployment file.

**Output Submission Target File Name:** `submission_shannon_entropy_non_linear_scaled_triple_stack.csv`
