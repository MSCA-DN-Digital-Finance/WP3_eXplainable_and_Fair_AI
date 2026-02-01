# Candidate-Driven Anchors for Green Claim Detection

This repository contains the implementation and experiments for a PhD-level course assignment on **Explainable Artificial Intelligence (XAI)**, focusing on the **Anchors** explanation method applied to transformer-based text classification.

The project investigates a key limitation of Anchors—**high computational cost**—and proposes a **candidate-driven anchor search strategy** that substantially reduces runtime while preserving explanation quality. The approach is evaluated on a green marketing claim detection task using a fine-tuned DistilBERT model.

---

## Project Structure

```text
.
├── 01_train_distilbert_binary.py
├── 02_confusion_matrix_distilbert.py
├── 03_anchors_beamsearch_spacy.py
├── 04_tfidf_contrast_candidates.py
├── 05_candidate_anchor_precisionB_spacy.py
├── 06_inspect_no_candidate_matches.py
├── 07_compare_anchor_methods.py
├── requirements.txt
└── results/
```

---

## Dataset

The experiments use the **Green Claims Dataset** introduced in:

> Woloszyn et al., *Towards Automatic Green Claim Detection*,  
> FIRE 2021  
> DOI: https://doi.org/10.1145/3503162.3503163  
> Dataset: https://zenodo.org/records/5764913

The dataset contains **773 tweets** from the cosmetics and electronics domains, annotated with:
- a **binary label** (green claim vs. non-green claim),
- a **multi-class label** (explicit green claim, implicit green claim, non-green claim).

Only the **binary labels** are used in this project.

The dataset is expected at:

```text
data/green_claims.csv
```

with the following columns:
- `tweet`: the tweet text,
- `label_binary`: binary label (0 = non-green, 1 = green claim).

---

## Environment Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy language model:
```bash
python -m spacy download en_core_web_md
```

---

## Running the Experiments

### 1. Train the classification model

Fine-tune DistilBERT for binary green claim classification:

```bash
python 01_train_distilbert_binary.py
```

This script:
- performs a stratified 80/20 train–test split,
- fine-tunes `distilbert-base-uncased`,
- saves the trained model to `distilbert-green-claim-binary-best/`,
- exports `train_split.csv` and `test_split.csv`.

Model checkpoints are not versioned and can be reproduced by running the training script.

---

### 2. Evaluate classification performance

Compute the confusion matrix on the test set:

```bash
python 02_confusion_matrix_distilbert.py
```

---

### 3. Beam-search Anchors baseline

Run the beam-search-based anchor discovery method inspired by the original Anchors algorithm:

```bash
python 03_anchors_beamsearch_spacy.py
```

**Outputs:**
- `beam_precisionB_spacy.csv`
- `beam_best_anchor_precisionB.csv`
- `anchors_metrics_beam.json`
- `anchors_runtime_beam.txt`

---

### 4. Candidate generation via contrastive TF-IDF

Extract candidate anchor phrases using contrastive TF-IDF between green and non-green tweets:

```bash
python 04_tfidf_contrast_candidates.py \
  --input data/green_claims.csv \
  --text_col tweet \
  --label_col label_binary \
  --top_k 80
```

**Outputs:**
- `tfidf_contrast_candidates.csv`
- `topK_candidates.txt`

---

### 5. Candidate-driven Anchors

Evaluate Anchors using the pre-selected candidate phrases:

```bash
python 05_candidate_anchor_precisionB_spacy.py
```

**Outputs:**
- `candidate_precisionB_spacy.csv`
- `candidate_best_anchor_precisionB.csv`
- `anchors_metrics_candidate_first.json`
- `anchors_runtime_candidate_first.txt`

---

### 6. Inspect tweets without candidate matches

Identify tweets for which no candidate anchor matches were found:

```bash
python 06_inspect_no_candidate_matches.py
```

---

### 7. Compare anchor search strategies

Compare beam-search and candidate-driven Anchors on a per-instance basis:

```bash
python 07_compare_anchor_methods.py
```

**Outputs:**
- `anchor_comparison.csv`,
- printed summary statistics comparing anchor overlap, precision, and coverage.

---

## Reproducibility Notes

- All scripts use fixed random seeds to ensure reproducibility.
- The `requirements.txt` file contains only the libraries required to run the experiments.
- Runtime may vary depending on hardware and available acceleration (CPU vs. GPU/MPS).

---

## License

This project is developed for **educational and research purposes** as part of a PhD-level course assignment on Explainable Artificial Intelligence.  
The dataset is subject to the original authors’ license and should be cited accordingly.
