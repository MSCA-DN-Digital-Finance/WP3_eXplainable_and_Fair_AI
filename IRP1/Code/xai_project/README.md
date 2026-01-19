# Explainable AI for Loan Granting Models

This repository contains the implementation and analysis for an **Explainable Artificial Intelligence (XAI)** assignment conducted as part of the MSCA Digital Finance doctoral training. The work was completed in the context of the course **“The Need for Explainable AI”**, part of the MSCA-DN Digital Finance program.

➡️ **Course assignment description:**  
https://github.com/MSCA-DN-Digital-Finance/Courses/tree/main/Cohort%201%20(2024.01-2027.12)/MSCA_DF_11_The%20Need%20for%20Explainable%20AI

The goal of the project is to train a supervised machine learning model for **loan approval decisions** and to analyze its behavior using **post-hoc explainability methods**, with a focus on **SHAP (SHapley Additive exPlanations)**. The project emphasizes interpretability, feature attribution, and critical assessment of model-driven financial decisions.

---

## Project Structure

```text
xai_project/
├── data/
│   └── README.md                  # Instructions for obtaining the dataset externally
├── results/
│   └── eval_*.json                 # Logged evaluation results from multiple runs
├── 251014_data_preparation.ipynb   # Data cleaning, train/test split
├── 251014_scm.ipynb                # Structural causal model (SCM) with neural mechanisms
├── 251014_xgb_training.ipynb       # XGBoost baseline training and evaluation
├── utils.py                        # Shared helper functions
├── shap_dependence_fico_n.png      # SHAP dependence plot (FICO score)
├── shap_dependence_loan_amnt.png   # SHAP dependence plot (loan amount)
├── shap_dependence_revenue.png     # SHAP dependence plot (revenue / income)
├── shap_summary_bar.png            # Global SHAP feature importance (bar plot)
├── shap_summary_beeswarm.png       # SHAP summary (beeswarm)
├── environment.yml                 # Conda environment specification
├── CITATION.cff                    # Citation metadata for academic reuse
├── LICENSE                         # Apache License 2.0
├── README.md                       # Project documentation
└── 260119_xai_assignment_mathis_jander.pdf  # Final assignment report (PDF)

```

## Workflow Overview

### Data Preparation & Feature Engineering
- Implemented in `251014_data_preparation.ipynb`
- Lending Club loan data is cleaned, standardized, and transformed into a domain-informed feature set
- Engineered variables reflect credit-relevant concepts (e.g. income, loan amount, debt-to-income)
- A fixed train/test split is created to ensure consistent comparison across models

### Supervised Learning Baseline (SML)
- Implemented in `251014_xgb_training.ipynb`
- A tuned **XGBoost classifier** is trained to predict loan default under purely observational assumptions
- Hyperparameters are selected via randomized search with cross-validation
- Serves as a strong black-box baseline for predictive performance (ROC AUC, PR-AUC, F1)

### Structural Causal Model with Neural Mechanisms
- Implemented in `251014_scm.ipynb`
- A **structural causal model (SCM)** is specified via an explicit directed acyclic graph (DAG)
- Each causal mechanism is parameterized by a **feed-forward neural network (FFNN)**
- Continuous nodes are trained via regression (MSE), the binary outcome via classification (BCE)
- Supports both:
  - *Factual inference* (using observed intermediate variables)
  - *From-roots inference* (recomputing endogenous variables via the DAG)

### Comparative Evaluation
- Predictive performance of SCM+FFNN is benchmarked against XGBoost
- Metrics include ROC AUC, PR-AUC, and F1 score at fixed and optimal thresholds
- Results highlight the trade-off between predictive accuracy and causal transparency

### Interventional Analysis (Explainable-by-Design)
- The SCM natively supports **interventions via Pearl’s `do`-operator**
- Policy-relevant “what-if” analyses are performed (e.g. changing income or loan amount)
- Effects are propagated through intermediate variables (e.g. debt-to-income), enabling causal tracing
- Demonstrates capabilities fundamentally unavailable to post-hoc XAI methods

### Post-hoc Explainability for Comparison
- **SHAP** is applied to the XGBoost baseline for associational explanations
- Generated artifacts include:
  - Global feature importance (`shap_summary_bar.png`)
  - Distribution of SHAP values (`shap_summary_beeswarm.png`)
  - Feature dependence plots (income, loan amount, FICO score)
- Used to illustrate limitations of post-hoc explanations under correlated and engineered features


---

## Environment & Reproducibility

The project uses **conda** for environment management.

To recreate the environment, run:

```bash
conda env create -f environment.yml
conda activate xai_project
```
All experiments were executed within the same environment to ensure consistency across training, evaluation, and explainability steps.

---

## Notes

- This project is intended for **educational and research purposes**
- It does **not** constitute a production-ready credit decision system
- Results should be interpreted with caution, particularly in real-world lending contexts

---

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.


## Citation

If you use this code in academic work, please cite it appropriately.
A `CITATION.cff` file is provided.
