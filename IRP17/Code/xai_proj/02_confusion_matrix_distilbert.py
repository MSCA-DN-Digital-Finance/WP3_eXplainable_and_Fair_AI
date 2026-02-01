import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# HuggingFace datasets and transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)

# -------------------------
# Configuration
# -------------------------
SEED = 42
MODEL_PATH = "models/distilbert-green-claim-binary-best"
DATA_PATH = "data/green_claims.csv"

# -------------------------
# Main evaluation pipeline
# -------------------------
def main():
    # -------------------------
    # Load and preprocess data
    # -------------------------
    # Load the full dataset (same preprocessing as during training)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["tweet", "label_binary"]).copy()

    # Map string labels to binary integers
    label_map = {"not_green": 0, "green_claim": 1}
    df["label_binary"] = df["label_binary"].map(label_map)

    # Drop rows with unmapped labels (safety check)
    df = df.dropna(subset=["label_binary"]).copy()
    df["label_binary"] = df["label_binary"].astype(int)

    # -------------------------
    # Recreate the SAME train/test split
    # -------------------------
    # Important: use the same random seed and stratification
    # to ensure consistency with training
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label_binary"],
    )

    # Convert test set to HuggingFace Dataset
    test_ds = Dataset.from_pandas(
        test_df[["tweet", "label_binary"]]
        .rename(columns={"label_binary": "labels"})
    )

    # -------------------------
    # Load trained model and tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # Tokenization function
    def tokenize(batch):
        """
        Tokenize tweet text for DistilBERT.
        """
        return tokenizer(
            batch["tweet"],
            truncation=True,
            max_length=128
        )

    test_ds = test_ds.map(tokenize, batched=True)

    # Dynamic padding for efficient batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------------
    # Prediction
    # -------------------------
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
    )

    # Run prediction on the test set
    out = trainer.predict(test_ds)
    y_true = out.label_ids
    y_pred = np.argmax(out.predictions, axis=1)

    # -------------------------
    # Evaluation metrics
    # -------------------------
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["True not_green", "True green_claim"],
        columns=["Pred not_green", "Pred green_claim"],
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    # Detailed classification metrics
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["not_green", "green_claim"]
        )
    )

    # -------------------------
    # Save results
    # -------------------------
    cm_df.to_csv("results/confusion_matrix_distilbert.csv", index=True)
    print("\nSaved: confusion_matrix_distilbert.csv")

# Entry point
if __name__ == "__main__":
    main()
