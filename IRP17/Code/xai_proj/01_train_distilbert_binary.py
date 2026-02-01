import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# HuggingFace datasets and transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -------------------------
# Configuration
# -------------------------
SEED = 42
MODEL_NAME = "distilbert-base-uncased"

# -------------------------
# Metric computation
# -------------------------
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for binary classification.
    Metrics are computed on the validation/test set at evaluation time.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    prec, rec, _, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }

# -------------------------
# Main training pipeline
# -------------------------
def main():
    # Ensure reproducibility
    set_seed(SEED)

    # -------------------------
    # Load and preprocess data
    # -------------------------
    df = pd.read_csv("green_claims.csv")
    df = df.dropna(subset=["tweet", "label_binary"]).copy()

    # Map string labels to binary integers
    label_map = {
        "not_green": 0,
        "green_claim": 1
    }
    df["label_binary"] = df["label_binary"].map(label_map).astype(int)

    # -------------------------
    # Train / test split
    # -------------------------
    # Stratified split to preserve class distribution
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label_binary"]
    )

    # Save splits for later reuse (e.g. Anchors experiments)
    train_df.to_csv("train_split.csv", index=False)
    test_df.to_csv("test_split.csv", index=False)

    # Convert pandas DataFrames to HuggingFace Datasets
    train_ds = Dataset.from_pandas(
        train_df[["tweet", "label_binary"]]
        .rename(columns={"label_binary": "labels"})
    )
    test_ds = Dataset.from_pandas(
        test_df[["tweet", "label_binary"]]
        .rename(columns={"label_binary": "labels"})
    )

    # -------------------------
    # Tokenization
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        """
        Tokenize input tweets with truncation.
        """
        return tokenizer(
            batch["tweet"],
            truncation=True,
            max_length=128
        )

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    # Dynamic padding for efficient batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------------
    # Model initialization
    # -------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # -------------------------
    # Training configuration
    # -------------------------
    args = TrainingArguments(
        output_dir="models/distilbert-green-claim-binary",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    # -------------------------
    # Trainer setup
    # -------------------------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # -------------------------
    # Training and evaluation
    # -------------------------
    trainer.train()
    metrics = trainer.evaluate()
    print("Test metrics:", metrics)

    # -------------------------
    # Save best model and tokenizer
    # -------------------------
    trainer.save_model("models/distilbert-green-claim-binary-best")
    tokenizer.save_pretrained("models/distilbert-green-claim-binary-best")

# Entry point
if __name__ == "__main__":
    main()
