# 🧮 Credit Scoring Framework in Python

A **modular and automated Python framework** for credit scoring model development — including data preprocessing, optimal monotonic binning, Weight of Evidence (WoE) transformation, logistic regression training, and interpretable scorecard generation.

This repository provides a **ready-to-use foundation** for building, validating, and deploying explainable credit scoring systems.

## 📁 Project Structure

```text
credit-scoring-framework-python/
│
├── data.csv                  # Example dataset
├── scorecard.py              # Core framework (Scorecard class)
└── scorecardNotebook.ipynb   # Notebook example and demonstration


## ⚙️ Features

- **Automated Data Preprocessing** — handles feature cleaning, missing value encoding, and transformations.  
- **Optimal Monotonic Binning** — uses entropy-based algorithms via `optbinning` to ensure monotonicity and interpretability.  
- **WoE Transformation & IV Calculation** — converts variables into Weight of Evidence scale and computes Information Value for feature selection.  
- **Logistic Regression Model** — built with scikit-learn to produce interpretable credit scorecards.  
- **Score Scaling** — converts model outputs into score contributions using PDO, base odds, and base score.  
- **Performance Evaluation** — computes AUC, cross-validation stability, and OOT performance metrics.

---

## 🧠 Methodology Overview

1. **Data Preparation**  
   - Input data with binary performance target (good/bad).  
   - Split into in-time and OOT samples for validation.

2. **Feature Binning & WoE Encoding**  
   - Apply quantile or optimal binning per feature.  
   - Compute WoE and IV per bin to assess predictive power.

3. **Model Training**  
   - Logistic regression on WoE-transformed features.  
   - Evaluate AUC stability across validation folds.

4. **Scorecard Generation**  
   - Transform model coefficients into score contributions:  
     \[
     \text{Score} = C - M \times \ln\left(\frac{P(\text{bad})}{1 - P(\text{bad})}\right)
     \]
   - Output JSON mapping bins → scores.

5. **Evaluation**  
   - Cross-validation AUC, in-time vs. OOT comparison.  
   - PSI / IV drift checks for stability.


## 🚀 Quick Start

### 1️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn optbinning tqdm

### 2️⃣ Run the example notebook

Open **`scorecardNotebook.ipynb`** to explore a full demonstration of:

- Data preprocessing  
- Optimal binning visualization  
- Scorecard generation  
- AUC evaluation and interpretation


### 3️⃣ (Optional) Run the framework directly

```bash
python scorecard.py
