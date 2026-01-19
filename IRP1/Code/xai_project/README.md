# Explainable AI for Loan Granting Models

This repository contains the implementation and analysis for an **Explainable Artificial Intelligence (XAI)** assignment conducted as part of the MSCA Digital Finance doctoral training. The work was completed in the context of the course **“The Need for Explainable AI”**, part of the MSCA-DN Digital Finance program.

➡️ **Course assignment description:**  
https://github.com/MSCA-DN-Digital-Finance/Courses/tree/main/Cohort%201%20(2024.01-2027.12)/MSCA_DF_11_The%20Need%20for%20Explainable%20AI

The goal of the project is to train a supervised machine learning model for **loan approval decisions** and to analyze its behavior using **post-hoc explainability methods**, with a focus on **SHAP (SHapley Additive exPlanations)**. The project emphasizes interpretability, feature attribution, and critical assessment of model-driven financial decisions.

---

## Project Structure

```text
xai_project/
├── __pycache__/
├── data/
│   ├── LC_loans_granting_model_dataset.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── results/
│   ├── eval_20251014_115109.json
│   ├── eval_20251014_115147.json
│   ├── eval_20251014_115236.json
│   ├── eval_20251014_115320.json
│   ├── eval_20251014_123329.json
│   ├── eval_20251014_142017.json
│   ├── eval_20251014_150637.json
│   ├── eval_20251014_154532.json
│   └── eval_20251014_161610.json
├── 251014_data_preparation.ipynb
├── 251014_scm.ipynb
├── 251014_xgb_training.ipynb
├── environment.yml
├── shap_dependence_fico_n.png
├── shap_dependence_loan_amnt.png
├── shap_dependence_revenue.png
├── shap_summary_bar.png
├── shap_summary_beeswarm.png
└── utils.py
```

## Workflow Overview

### Data Preparation
- Implemented in `251014_data_preparation.ipynb`
- Raw loan data is cleaned, encoded, and split into training and test sets
- Processed datasets are stored as NumPy arrays to ensure reproducibility

### Model Training
- Implemented in `251014_xgb_training.ipynb`
- An **XGBoost classifier** is trained to predict loan approval outcomes
- Multiple evaluation runs are executed and logged for comparison

### Evaluation and Logging
- Model performance metrics are stored as timestamped JSON files in `results/`
- Enables systematic comparison across runs and hyperparameter settings

### Explainability Analysis (XAI)
- **SHAP** is used to analyze feature importance and both local and global effects
- Generated outputs include:
  - Global feature importance: `shap_summary_bar.png`
  - Distribution of SHAP values: `shap_summary_beeswarm.png`
  - Feature dependence plots (e.g. FICO score, loan amount, revenue)

### Structural / Causal Exploration
- Implemented in `251014_scm.ipynb`
- Explores structural or causal perspectives on model behavior
- Used to contrast purely predictive explanations with more causal reasoning

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
