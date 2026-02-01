import re
import numpy as np
import pandas as pd
import torch
import spacy
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import math
import time

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "models/distilbert-green-claim-binary-best"
TEST_FILE = "data/test_split.csv"
TRAIN_FILE = "data/train_split.csv"          # used to build replacement pools
CANDIDATE_FILE = "topK_candidates.txt"

TAU = 0.95
N_PERT = 200            # number of perturbations per (x, candidate)
REPLACE_PROB = 0.35     # probability to replace a non-anchor token
MIN_POOL_PER_POS = 50   # ensure pools are not tiny

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32

# -------------------------
# spaCy model
# -------------------------
nlp = spacy.load("en_core_web_md")  # or en_core_web_sm (md preferred for similarity)

# -------------------------
# Model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

@torch.no_grad()
def predict_label(texts):
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc.pop("token_type_ids", None)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return np.array(preds)


# -------------------------
# Predicate matching view (clean only for matching)
# -------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
HTML_AMP_RE = re.compile(r"&amp;")

def clean_for_match(s: str) -> str:
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
    # word boundary phrase match on cleaned text
    return re.search(r"\b" + re.escape(cand) + r"\b", clean_text) is not None


# -------------------------
# Build POS replacement pools from TRAIN RAW text
# -------------------------
def build_pos_pools(train_texts, max_per_pos=3000):
    pools = defaultdict(list)
    for doc in nlp.pipe(train_texts, batch_size=64):
        for t in doc:
            if not t.is_alpha:
                continue
            if t.is_stop:
                continue
            w = t.text
            pos = t.pos_
            # keep moderately sized pools
            if len(pools[pos]) < max_per_pos:
                pools[pos].append(w)
    # deduplicate
    for pos in list(pools.keys()):
        pools[pos] = list(dict.fromkeys(pools[pos]))
    return pools


# -------------------------
# Find anchor spans in RAW text using spaCy
# -------------------------
def find_anchor_spans(doc, anchor_phrase: str):
    """
    Word-level anchor matching on spaCy tokens, but robust to joiner punctuation
    between words (e.g. cruelty-free, cruelty / free).

    Returns list of (start_token_i, end_token_i) inclusive span token indices.
    The returned span covers the raw tokens, including joiners, so the raw
    substring is preserved during perturbation.
    """
    # anchor words (space-separated)
    phrase_tokens = [w for w in anchor_phrase.lower().split() if w]
    if not phrase_tokens:
        return []

    # Joiners we allow BETWEEN anchor words (keep conservative)
    JOINERS = {"-", "–", "—", "/", "\\", "·", "•", "'"}

    def is_joiner(tok):
        return tok.is_punct and tok.text in JOINERS

    spans = []
    n = len(doc)
    L = len(phrase_tokens)

    for i in range(n):
        # must start at first word
        if doc[i].text.lower() != phrase_tokens[0]:
            continue

        k = i      # pointer in doc
        j = 0      # pointer in phrase_tokens
        end = None

        while k < n and j < L:
            tl = doc[k].text.lower()

            if tl == phrase_tokens[j]:
                end = k
                j += 1
                k += 1
                continue

            # allow joiner punctuation only after at least one word matched
            # and only until the phrase completes
            if j > 0 and is_joiner(doc[k]):
                end = k  # include joiner in anchor span
                k += 1
                continue

            break

        if j == L:
            spans.append((i, end))

    return spans


def token_in_any_span(i, spans):
    for s, e in spans:
        if s <= i <= e:
            return True
    return False


# -------------------------
# spaCy perturbation: perturb RAW text, keep anchor phrase fixed
# -------------------------
def perturb_raw_keep_anchor(raw_text: str, anchor_phrase: str, pos_pools, rng):
    doc = nlp(raw_text)
    spans = find_anchor_spans(doc, anchor_phrase)

    # if anchor phrase not found in raw text exactly, return None (cannot enforce A)
    if len(spans) == 0:
        return None

    out = []
    for i, t in enumerate(doc):
        # keep spacing by using token.whitespace_
        tok = t.text
        ws = t.whitespace_

        # don't touch anchor tokens
        if token_in_any_span(i, spans):
            out.append(tok + ws)
            continue

        # don't replace punctuation, numbers, URLs etc.
        if (not t.is_alpha) or t.is_stop:
            out.append(tok + ws)
            continue

        if rng.random() < REPLACE_PROB:
            pool = pos_pools.get(t.pos_, [])
            if len(pool) >= MIN_POOL_PER_POS:
                repl = rng.choice(pool)
                out.append(repl + ws)
            else:
                out.append(tok + ws)
        else:
            out.append(tok + ws)

    return "".join(out).strip()


def estimate_precision_B(raw_text: str, anchor_phrase: str, fx: int, pos_pools, n=N_PERT, seed=0):
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n):
        z = perturb_raw_keep_anchor(raw_text, anchor_phrase, pos_pools, rng)
        if z is None:
            return None  # anchor not enforceable in raw text
        zs.append(z)

    preds = predict_label(zs)
    return float(np.mean(preds == fx))


# -------------------------
# MAIN
# -------------------------
def main():
    df_test = pd.read_csv(TEST_FILE).dropna(subset=["tweet", "label_binary"]).copy()
    df_train = pd.read_csv(TRAIN_FILE).dropna(subset=["tweet", "label_binary"]).copy()

    # model predictions on RAW tweets
    df_test["tweet"] = df_test["tweet"].astype(str)
    df_train["tweet"] = df_train["tweet"].astype(str)

    print("Predicting on test (RAW)...")
    df_test["pred"] = predict_label(df_test["tweet"].tolist())

    # TP green only
    tp_green = df_test[(df_test["label_binary"] == 1) & (df_test["pred"] == 1)].copy()
    print(f"TP green: {len(tp_green)} / {len(df_test)}")

    # build POS replacement pools from TRAIN raw tweets
    print("Building POS replacement pools from train...")
    pos_pools = build_pos_pools(df_train["tweet"].tolist())
    print(f"Built pools for {len(pos_pools)} POS tags")

    # candidates (already clean phrases from TF-IDF)
    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        candidates = [l.strip() for l in f if l.strip()]

    # cleaned view for matching only
    tp_green["tweet_clean"] = tp_green["tweet"].map(clean_for_match)

    # -------------------------
    # ANCHORS PART (timed + metrics)
    # -------------------------
    t0 = time.perf_counter()

    num_applicable_pairs = 0   # (x, cand) where phrase_in_clean is True
    num_perturb_calls = 0      # times we attempted precision_B (i.e., called estimate_precision_B)
    num_precB_none = 0         # precision_B returned None (anchor not enforceable)
    rows = []

    for idx, row in tp_green.iterrows():
        x_raw = row["tweet"]
        fx = int(row["pred"])
        x_clean = row["tweet_clean"]

        for cand in candidates:
            if not phrase_in_clean(x_clean, cand):
                continue

            num_applicable_pairs += 1
            num_perturb_calls += 1

            precB = estimate_precision_B(
                x_raw, cand, fx, pos_pools,
                n=N_PERT,
                seed=idx % 1000
            )
            if precB is None:
                num_precB_none += 1
                continue

            rows.append({
                "test_index": idx,
                "candidate": cand,
                "pred_label": fx,
                "precision_B": precB
            })

    res = pd.DataFrame(rows)

    # Coverage on TEST (cleaned matching) — add BEFORE saving candidate_precisionB_spacy.csv
    df_test["tweet_clean"] = df_test["tweet"].map(clean_for_match)

    cov_map = {
        cand: float(np.mean(df_test["tweet_clean"].apply(lambda t: phrase_in_clean(t, cand))))
        for cand in candidates
    }
    res["coverage_empirical"] = res["candidate"].map(cov_map)

    # Save with coverage included
    res.to_csv("results/candidate_precisionB_spacy.csv", index=False)
    print("Saved: candidate_precisionB_spacy.csv")

    # Select best anchor per test example (precision constraint, then max coverage)
    best = (
        res[res["precision_B"] >= TAU]
        .sort_values(["test_index", "coverage_empirical"], ascending=[True, False])
        .groupby("test_index")
        .head(1)
    )
    best.to_csv("results/candidate_best_anchor_precisionB.csv", index=False)
    print("Saved: candidate_best_anchor_precisionB.csv")
    print(best.head(10))

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"ANCHORS part time: {elapsed:.2f} seconds")

    # Derived compute driver
    num_perturb_texts = int(num_perturb_calls * N_PERT)
    approx_num_batches = int(math.ceil(num_perturb_texts / BATCH_SIZE)) if num_perturb_texts > 0 else 0

    # Optional overall coverage of "having an anchor" among TP green
    # (how many TP_green examples ended up with at least one candidate with precision>=TAU)
    anchor_rate_over_tp_green = float(best["test_index"].nunique() / len(tp_green)) if len(tp_green) > 0 else 0.0

    # Save metrics to JSON
    metrics = {
        "method": "candidate_first",
        "anchors_runtime_seconds": float(elapsed),
        "num_tp_green": int(len(tp_green)),
        "num_candidates": int(len(candidates)),
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
    }

    with open("results/anchors_metrics_candidate_first.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved: anchors_metrics_candidate_first.json")

    # (Optional) also save a tiny txt
    with open("results/anchors_runtime_candidate_first.txt", "w", encoding="utf-8") as f:
        f.write(f"anchors_runtime_seconds: {elapsed:.6f}\n")
        f.write(f"num_applicable_pairs: {num_applicable_pairs}\n")
        f.write(f"num_perturb_calls: {num_perturb_calls}\n")
        f.write(f"num_perturb_texts: {num_perturb_texts}\n")
        f.write(f"num_precB_none: {num_precB_none}\n")

    print("Saved: anchors_runtime_candidate_first.txt")



if __name__ == "__main__":
    main()
