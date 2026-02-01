import argparse
import re
import json
import math
import time
import numpy as np
import pandas as pd
import torch
import spacy
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# CONFIG
# -------------------------
# Trained classifier checkpoint
MODEL_PATH = "models/distilbert-green-claim-binary-best"

# Train/test splits produced during training
TEST_FILE = "data/test_split.csv"
TRAIN_FILE = "data/train_split.csv"

# Anchor validation threshold (minimum estimated precision)
TAU = 0.95

# Monte Carlo perturbation settings
N_PERT = 200          # number of perturbed samples per anchor evaluation
REPLACE_PROB = 0.35   # probability of replacing a non-anchor token
MIN_POOL_PER_POS = 50 # minimum pool size to allow replacements for a POS tag

# Inference settings
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32

# Beam-search settings
BEAM_WIDTH = 3          # number of partial anchors retained at each depth
DEPTH_MAX = 2           # maximum conjunction size (1..DEPTH_MAX)
MIN_TOKEN_LEN = 3
MAX_FEATS_PER_TWEET = 20 # cap per-tweet feature set to keep search bounded

# -------------------------
# spaCy pipeline
# -------------------------
# Used for tokenization, stopword filtering, and POS tagging for perturbations
nlp = spacy.load("en_core_web_md")

# -------------------------
# Load classifier
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

@torch.no_grad()
def predict_label(texts):
    """
    Predict class labels for a list of texts using the fine-tuned DistilBERT model.
    Runs in batches for efficiency.
    """
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        # DistilBERT does not use token_type_ids
        enc.pop("token_type_ids", None)

        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return np.array(preds)

# -------------------------
# Cleaned text view (only for matching features)
# -------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
HTML_AMP_RE = re.compile(r"&amp;")

def clean_for_match(s: str) -> str:
    """
    Normalize text for *matching* anchor phrases/features.
    This does NOT change the raw text used for perturbations.
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
    Whole-word phrase matching in the cleaned text.
    Used to decide whether a candidate feature applies to a tweet.
    """
    return re.search(r"\b" + re.escape(cand) + r"\b", clean_text) is not None

def anchor_applicable(clean_text: str, anchor_phrases) -> bool:
    """
    Conjunction rule: an anchor (possibly multiple tokens) applies iff
    ALL parts appear in the cleaned tweet.
    """
    return all(phrase_in_clean(clean_text, p) for p in anchor_phrases)

# -------------------------
# POS replacement pools (built from TRAIN data)
# -------------------------
def build_pos_pools(train_texts, max_per_pos=3000):
    """
    Build token replacement pools per POS tag from training data.
    During perturbation, a token is replaced with another token of the same POS.
    """
    pools = defaultdict(list)
    for doc in nlp.pipe(train_texts, batch_size=64):
        for t in doc:
            if not t.is_alpha:
                continue
            if t.is_stop:
                continue
            pos = t.pos_
            if len(pools[pos]) < max_per_pos:
                pools[pos].append(t.text)

    # Deduplicate pools (preserve order)
    for pos in list(pools.keys()):
        pools[pos] = list(dict.fromkeys(pools[pos]))
    return pools

# -------------------------
# Anchor span finding (to "protect" anchor words during perturbation)
# -------------------------
def find_anchor_spans(doc, anchor_phrase: str):
    """
    Locate occurrences of an anchor phrase in the spaCy-tokenized raw tweet.
    Allows joiner punctuation between words (e.g. cruelty-free).
    Returns spans of token indices (inclusive).
    """
    phrase_tokens = [w for w in anchor_phrase.lower().split() if w]
    if not phrase_tokens:
        return []

    JOINERS = {"-", "–", "—", "/", "\\", "·", "•", "'"}
    def is_joiner(tok):
        return tok.is_punct and tok.text in JOINERS

    spans = []
    n = len(doc)
    L = len(phrase_tokens)

    for i in range(n):
        if doc[i].text.lower() != phrase_tokens[0]:
            continue

        k = i
        j = 0
        end = None

        while k < n and j < L:
            tl = doc[k].text.lower()

            if tl == phrase_tokens[j]:
                end = k
                j += 1
                k += 1
                continue

            if j > 0 and is_joiner(doc[k]):
                end = k
                k += 1
                continue

            break

        if j == L:
            spans.append((i, end))

    return spans

def token_in_any_span(i, spans):
    """Helper: check if token index i lies inside any protected anchor span."""
    for s, e in spans:
        if s <= i <= e:
            return True
    return False

def union_spans_for_phrases(doc, anchor_phrases):
    """
    Find protected spans for each phrase in a conjunction anchor.
    If any phrase cannot be located in raw text, return None (not enforceable).
    """
    all_spans = []
    for ph in anchor_phrases:
        spans = find_anchor_spans(doc, ph)
        if len(spans) == 0:
            return None
        all_spans.extend(spans)
    return all_spans

# -------------------------
# Perturbation + precision estimation
# -------------------------
def perturb_raw_keep_anchors(raw_text: str, anchor_phrases, pos_pools, rng):
    """
    Generate one perturbed version of raw_text while keeping all anchor phrases fixed.
    Non-anchor tokens may be POS-replaced with probability REPLACE_PROB.
    """
    doc = nlp(raw_text)
    spans = union_spans_for_phrases(doc, anchor_phrases)
    if spans is None:
        return None

    out = []
    for i, t in enumerate(doc):
        tok = t.text
        ws = t.whitespace_

        # Keep anchor tokens unchanged
        if token_in_any_span(i, spans):
            out.append(tok + ws)
            continue

        # Do not perturb punctuation / stopwords / non-alphabetic tokens
        if (not t.is_alpha) or t.is_stop:
            out.append(tok + ws)
            continue

        # Replace token with same-POS token sampled from training pools
        if rng.random() < REPLACE_PROB:
            pool = pos_pools.get(t.pos_, [])
            if len(pool) >= MIN_POOL_PER_POS:
                out.append(rng.choice(pool) + ws)
            else:
                out.append(tok + ws)
        else:
            out.append(tok + ws)

    return "".join(out).strip()

def estimate_precision_B_multi(raw_text: str, anchor_phrases, fx: int, pos_pools, n=N_PERT, seed=0):
    """
    Monte Carlo estimate of anchor precision:
      P(f(z) = f(x) | anchor holds)
    where each z is produced by perturb_raw_keep_anchors().
    """
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n):
        z = perturb_raw_keep_anchors(raw_text, anchor_phrases, pos_pools, rng)
        if z is None:
            return None
        zs.append(z)

    preds = predict_label(zs)
    return float(np.mean(preds == fx))

# -------------------------
# Beam-search feature extraction
# -------------------------
def extract_unigram_features(clean_text: str):
    """
    Extract candidate unigram features from a cleaned tweet:
    - minimum length
    - remove stopwords
    - de-duplicate
    - cap at MAX_FEATS_PER_TWEET
    """
    toks = [t for t in clean_text.split() if len(t) >= MIN_TOKEN_LEN]
    toks = [t for t in toks if not nlp.vocab[t].is_stop]

    seen = set()
    feats = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            feats.append(t)

    return feats[:MAX_FEATS_PER_TWEET]

# -------------------------
# Coverage helpers
# -------------------------
def compute_coverage_map(df_test_clean, candidates):
    """
    Compute unigram coverage over test set:
      cov(term) = fraction of test tweets where term occurs (cleaned matching).
    """
    cov = {}
    for cand in candidates:
        cov[cand] = float(np.mean(df_test_clean.apply(lambda t: phrase_in_clean(t, cand))))
    return cov

def conjunctive_coverage_from_unigram_cov(anchor_phrases, cov_map):
    """
    Approximate coverage of conjunction using independence assumption:
      cov(A AND B) ≈ cov(A) * cov(B)
    Used only for tie-breaking among anchors.
    """
    prod = 1.0
    for p in anchor_phrases:
        prod *= float(cov_map.get(p, 0.0))
    return prod

# -------------------------
# MAIN
# -------------------------
def main():
    # Optional debugging: run only on first N TP-green tweets
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    # Load split data
    df_test = pd.read_csv(TEST_FILE).dropna(subset=["tweet", "label_binary"]).copy()
    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=["tweet", "label_binary"]).copy()

    df_test["tweet"] = df_test["tweet"].astype(str)
    df_train["tweet"] = df_train["tweet"].astype(str)

    # Predict labels on test
    print("Predicting on test (RAW)...")
    df_test["pred"] = predict_label(df_test["tweet"].tolist())

    # Anchor explanations are computed only for TRUE POSITIVE green claims
    tp_green = df_test[(df_test["label_binary"] == 1) & (df_test["pred"] == 1)].copy()
    print(f"TP green: {len(tp_green)} / {len(df_test)}")

    if args.max_samples is not None:
        tp_green = tp_green.head(args.max_samples)

    # Build perturbation replacement pools from training data
    print("Building POS replacement pools from train...")
    pos_pools = build_pos_pools(df_train["tweet"].tolist())
    print(f"Built pools for {len(pos_pools)} POS tags")

    # Cleaned views (only for feature extraction and matching checks)
    tp_green["tweet_clean"] = tp_green["tweet"].map(clean_for_match)
    df_test["tweet_clean"] = df_test["tweet"].map(clean_for_match)

    # ---- Precompute unigram coverage over test set (for tie-breaking) ----
    unigram_global = set()
    for _, r in tp_green.iterrows():
        unigram_global.update(extract_unigram_features(r["tweet_clean"]))
    unigram_cov_map = compute_coverage_map(df_test["tweet_clean"], unigram_global)

    # ---- Beam search over anchors ----
    t0 = time.perf_counter()

    num_applicable_pairs = 0
    num_perturb_calls = 0
    num_precB_none = 0

    rows = []
    processed_tp = 0

    for idx, row in tp_green.iterrows():
        processed_tp += 1
        print(f"[PROGRESS] {processed_tp}/{len(tp_green)} TP tweets")

        x_raw = row["tweet"]
        fx = int(row["pred"])
        x_clean = row["tweet_clean"]

        feats = extract_unigram_features(x_clean)
        if not feats:
            continue

        # Beam stores partial anchors: (anchor_tuple, precision, approx_cov)
        beam = [(tuple(), 0.0, 0.0)]
        found_best = None

        for depth in range(1, DEPTH_MAX + 1):
            candidates_next = []

            for (anch, _, _) in beam:
                for f in feats:
                    if f in anch:
                        continue

                    new_anchor = tuple(list(anch) + [f])

                    # Ensure anchor holds in this tweet (conjunction)
                    if not anchor_applicable(x_clean, new_anchor):
                        continue

                    num_applicable_pairs += 1
                    num_perturb_calls += 1

                    precB = estimate_precision_B_multi(
                        x_raw, new_anchor, fx, pos_pools,
                        n=N_PERT,
                        seed=idx % 1000
                    )

                    if precB is None:
                        num_precB_none += 1
                        continue

                    approx_cov = conjunctive_coverage_from_unigram_cov(new_anchor, unigram_cov_map)
                    candidates_next.append((new_anchor, precB, approx_cov))

                    # Stop once we find an anchor reaching TAU, prefer higher coverage
                    if precB >= TAU:
                        if found_best is None or approx_cov > found_best[2]:
                            found_best = (new_anchor, precB, approx_cov)

            if not candidates_next:
                break

            # Keep the best BEAM_WIDTH anchors to expand further
            candidates_next.sort(key=lambda x: (x[1], x[2]), reverse=True)
            beam = candidates_next[:BEAM_WIDTH]

            # Early stop: once we already found a valid anchor at this depth
            if found_best is not None:
                break

        # Save best anchor found for this tweet
        if found_best is not None:
            anchor_tuple, prec, approx_cov = found_best
            rows.append({
                "test_index": idx,
                "candidate": " AND ".join(anchor_tuple),
                "pred_label": fx,
                "precision_B": prec,
                "coverage_empirical": approx_cov,  # approximate for conjunctions
                "anchor_size": len(anchor_tuple),
                "beam_width": BEAM_WIDTH,
                "depth_max": DEPTH_MAX
            })

    # Save results
    res = pd.DataFrame(rows)
    res.to_csv("results/beam_precisionB_spacy.csv", index=False)
    best = res.sort_values(["test_index", "coverage_empirical"], ascending=[True, False]).drop_duplicates("test_index")
    best.to_csv("results/beam_best_anchor_precisionB.csv", index=False)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"BEAM ANCHORS part time: {elapsed:.2f} seconds")

    # Save run-level metrics
    num_perturb_texts = int(num_perturb_calls * N_PERT)
    approx_num_batches = int(math.ceil(num_perturb_texts / BATCH_SIZE)) if num_perturb_texts > 0 else 0
    anchor_rate_over_tp_green = float(best["test_index"].nunique() / len(tp_green)) if len(tp_green) > 0 else 0.0

    metrics = {
        "method": "beam_search_unigram_conjunction",
        "anchors_runtime_seconds": float(elapsed),
        "num_tp_green": int(len(tp_green)),
        "beam_width": int(BEAM_WIDTH),
        "depth_max": int(DEPTH_MAX),
        "max_feats_per_tweet": int(MAX_FEATS_PER_TWEET),
        "num_applicable_pairs": int(num_applicable_pairs),
        "num_perturb_calls": int(num_perturb_calls),
        "num_perturb_texts": int(num_perturb_texts),
        "num_precB_none": int(num_precB_none),
        "N_PERT": int(N_PERT),
        "BATCH_SIZE": int(BATCH_SIZE),
        "approx_num_batches": int(approx_num_batches),
        "TAU": float(TAU),
        "num_best_anchors": int(len(best)),
        "anchor_rate_over_tp_green": float(anchor_rate_over_tp_green),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "coverage_note": "coverage is approximated for conjunctions via independence product of unigram coverages"
    }

    with open("results/anchors_metrics_beam.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open("results/anchors_runtime_beam.txt", "w", encoding="utf-8") as f:
        f.write(f"anchors_runtime_seconds: {elapsed:.6f}\n")
        f.write(f"num_applicable_pairs: {num_applicable_pairs}\n")
        f.write(f"num_perturb_calls: {num_perturb_calls}\n")
        f.write(f"num_perturb_texts: {num_perturb_texts}\n")
        f.write(f"num_precB_none: {num_precB_none}\n")
        f.write(f"beam_width: {BEAM_WIDTH}\n")
        f.write(f"depth_max: {DEPTH_MAX}\n")

if __name__ == "__main__":
    main()
