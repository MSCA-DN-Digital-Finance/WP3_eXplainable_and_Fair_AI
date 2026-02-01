import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# MODEL CONFIGURATION
# -------------------------
MODEL_PATH = "models/distilbert-green-claim-binary-best"

# Use Apple's Metal (MPS) if available; otherwise fallback to CPU.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Batch size for faster inference.
BATCH_SIZE = 32

# Load tokenizer + model once (reused for all predictions).
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


@torch.no_grad()
def predict_label(texts):
    """
    Predict labels for a list of texts using the fine-tuned DistilBERT model.

    Returns:
        list[int]: model predictions (0/1) for each input text
    """
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        # Tokenize exactly as in training/evaluation (truncation to 128).
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # DistilBERT does not use token_type_ids; safe to drop if present.
        enc.pop("token_type_ids", None)

        # Move tensors to device and run forward pass.
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits

        # Convert logits to class predictions (argmax).
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    return preds


# -------------------------
# SCRIPT CONFIG
# -------------------------
TEST_FILE = "data/test_split.csv"
CANDIDATE_FILE = "results/topK_candidates.txt"

# If True, restrict to correctly classified green claims (label=1 and pred=1),
# i.e., the same population used for anchor explanations.
ONLY_TP_GREEN = True

# -------------------------
# MATCHING UTILITIES
# -------------------------
# Regular expressions used to normalize tweets for matching.
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
HTML_AMP_RE = re.compile(r"&amp;")


def clean_for_match(s: str) -> str:
    """
    Normalize tweet text for consistent candidate phrase matching.

    Steps:
      - replace HTML ampersand encoding
      - remove URLs and user mentions
      - convert hashtags "#word" -> "word"
      - lowercase
      - keep only alphabetic characters + spaces
      - collapse whitespace
    """
    s = str(s)
    s = HTML_AMP_RE.sub(" and ", s)
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(r"\1", s)
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def phrase_in_clean(clean_text: str, cand: str) -> bool:
    """
    Check whether candidate phrase appears as a whole-word match in clean_text.

    This avoids false matches such as:
      cand="eco" matching "economy"
    """
    return re.search(r"\b" + re.escape(cand) + r"\b", clean_text) is not None


# -------------------------
# MAIN
# -------------------------
def main():
    # Load test split. Drop rows missing required fields.
    df = pd.read_csv(TEST_FILE).dropna(subset=["tweet", "label_binary"]).copy()

    # Compute model predictions on raw tweets.
    df["pred"] = predict_label(df["tweet"].astype(str).tolist())

    # Optional: restrict to true positives (same set as Anchors evaluation).
    if ONLY_TP_GREEN:
        df = df[(df["label_binary"] == 1) & (df["pred"] == 1)].copy()

    # Load candidate phrases.
    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        candidates = [l.strip() for l in f if l.strip()]

    # Precompute normalized tweet text for matching.
    df["tweet_clean"] = df["tweet"].astype(str).map(clean_for_match)

    no_match_rows = []
    match_counts = []

    # For each tweet, count how many candidates appear.
    for idx, row in df.iterrows():
        clean = row["tweet_clean"]
        matches = [c for c in candidates if phrase_in_clean(clean, c)]

        match_counts.append(len(matches))

        # Record tweets where *no* candidate appears.
        if len(matches) == 0:
            no_match_rows.append({
                "test_index": idx,
                "tweet": row["tweet"],
                "label_binary": row["label_binary"]
            })

    no_match_df = pd.DataFrame(no_match_rows)

    # Print summary diagnostics.
    print("=== Candidate Matching Diagnostics ===")
    print(f"Total evaluated tweets: {len(df)}")
    print(f"Tweets with NO candidate match: {len(no_match_df)}")
    print(f"Fraction with no match: {len(no_match_df) / len(df):.3f}")

    # Print a small sample for quick inspection.
    print("\nFirst 5 tweets with no candidate match:\n")
    for t in no_match_df["tweet"].head(5):
        print("-", t)

    # Save for reproducibility / appendix analysis.
    out_path = "results/tweets_with_no_candidate_match.csv"
    no_match_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
