import argparse
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Text normalization helpers
# -------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")   # keep hashtag word, drop '#'
HTML_AMP_RE = re.compile(r"&amp;")

def clean_text(s: str) -> str:
    """
    Normalize tweet text for TF-IDF computation.
    This cleaning is intentionally lightweight and preserves lexical content
    relevant for candidate extraction.
    """
    s = str(s)
    s = HTML_AMP_RE.sub(" and ", s)
    s = URL_RE.sub(" <URL> ", s)      # normalize URLs
    s = MENTION_RE.sub(" <USER> ", s) # normalize mentions
    s = HASHTAG_RE.sub(r"\1", s)      # "#EarthDay" -> "EarthDay"
    s = s.lower()
    # keep alphabetic tokens and special markers (<URL>, <USER>)
    s = re.sub(r"[^a-z\s<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------
# MAIN
# -------------------------
def main():
    # Command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV containing tweets and binary labels")
    ap.add_argument("--text_col", default="tweet", help="Text column name")
    ap.add_argument("--label_col", default="label_binary", help="Binary label column")
    ap.add_argument("--top_k", type=int, default=80, help="Number of candidate terms to export")
    ap.add_argument("--min_df", type=int, default=2, help="Minimum document frequency")
    ap.add_argument("--max_df", type=float, default=0.9, help="Maximum document frequency (fraction)")
    ap.add_argument("--ngram_max", type=int, default=2, help="Maximum n-gram size (default: unigrams + bigrams)")
    ap.add_argument("--output_csv", default="results/tfidf_contrast_candidates.csv")
    args = ap.parse_args()

    # Load dataset
    df = pd.read_csv(args.input)
    df = df.dropna(subset=[args.text_col, args.label_col]).copy()

    # Binary labels (0 = non-green, 1 = green)
    y = df[args.label_col].astype(int).values
    texts = df[args.text_col].map(clean_text).tolist()

    # TF-IDF vectorizer
    # Stopwords removed; min_df/max_df reduce noise and overly common terms
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.ngram_max),
        strip_accents="unicode",
    )

    # Document-term matrix
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    # Split indices by class
    idx_green = np.where(y == 1)[0]
    idx_nongreen = np.where(y == 0)[0]

    if len(idx_green) == 0 or len(idx_nongreen) == 0:
        raise ValueError("Both green and non-green examples are required for contrastive TF-IDF.")

    # Mean TF-IDF values per term in each class
    mean_green = np.asarray(X[idx_green].mean(axis=0)).ravel()
    mean_non = np.asarray(X[idx_nongreen].mean(axis=0)).ravel()

    # Contrast metrics
    diff = mean_green - mean_non
    ratio = (mean_green + 1e-12) / (mean_non + 1e-12)

    # Document frequency (for interpretability / filtering)
    df_term = np.asarray((X > 0).sum(axis=0)).ravel()

    # Assemble candidate table
    out = pd.DataFrame({
        "term": vocab,
        "mean_tfidf_green": mean_green,
        "mean_tfidf_non_green": mean_non,
        "diff_green_minus_non": diff,
        "ratio_green_over_non": ratio,
        "doc_freq": df_term
    })

    # Basic candidate filtering
    # - remove placeholders
    # - remove very short terms
    out = out[~out["term"].isin(["<url>", "<user>"])]
    out = out[out["term"].str.len() >= 3]

    # Rank candidates:
    # prioritize terms strongly associated with green claims
    out = out.sort_values(
        by=["diff_green_minus_non", "mean_tfidf_green", "ratio_green_over_non"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # Save full ranked list
    out.to_csv(args.output_csv, index=False)
    print(f"Saved ranked candidate terms to: {args.output_csv}")

    # Export top-K candidates for anchor evaluation
    top_terms = out["term"].head(args.top_k).tolist()
    with open("topK_candidates.txt", "w", encoding="utf-8") as f:
        for t in top_terms:
            f.write(t + "\n")

    print(f"Saved top-{args.top_k} term list to: topK_candidates.txt")
    print("\nTop 20 candidates:")
    print(out[[
        "term",
        "diff_green_minus_non",
        "mean_tfidf_green",
        "mean_tfidf_non_green",
        "doc_freq"
    ]].head(20))

if __name__ == "__main__":
    main()
