import sqlite3
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
df = load = None
from ml.load import load_data

df = load_data()
use = df.dropna(subset=["vendor_id", "invoice_date", "total", "invoice_no"]).copy()
use["date"] = pd.to_datetime(use.invoice_date, errors="coerce")
use = use.dropna(subset=["date"])
print(f"{len(use)} of {len(df)} documents have the fields needed\n")

# Strip the generator's "-COPY" tag so the detector can't cheat
use["inv_clean"] = use.invoice_no.str.replace("-COPY", "", regex=False)
use["true_base"] = use.invoice_no_gt.str.replace("-COPY", "", regex=False)

# 1) Build candidate pairs. Only compare invoices from the SAME vendor
#    (this is "blocking": it avoids comparing every invoice with every other).
rows = []
for vid, g in use.groupby("vendor_id"):
    for a, b in combinations(g.itertuples(), 2):
        rows.append({
            "doc_a": a.doc_id, "doc_b": b.doc_id,
            "amount_diff": abs(a.total - b.total) / max(a.total, b.total),
            "date_gap": abs((a.date - b.date).days),
            "inv_sim": SequenceMatcher(None, a.inv_clean, b.inv_clean).ratio(),
            "is_dup": a.true_base == b.true_base,    # answer key, for grading only
        })
pairs = pd.DataFrame(rows)
y = pairs.is_dup.astype(int)
print(f"Candidate pairs: {len(pairs)}   true duplicate pairs: {y.sum()}\n")


def report(name, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    print(f"{name:22s} precision {precision_score(y, pred, zero_division=0):.3f} | "
          f"recall {recall_score(y, pred):.3f} | F1 {f1_score(y, pred):.3f} | "
          f"caught {tp}, false alarms {fp}, missed {fn}")


# 2) Baseline: exact match on amount and date (what a database constraint does)
exact = ((pairs.amount_diff == 0) & (pairs.date_gap == 0)).astype(int)
report("Exact-match rule", exact)

# 3) Model: logistic regression on similarity features, cross-validated,
#    so every pair is scored by a model that never saw it in training.
FEATURES = ["amount_diff", "date_gap", "inv_sim"]
model = make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=1000))
folds = StratifiedKFold(n_splits=min(5, int(y.sum())), shuffle=True, random_state=SEED)
prob = cross_val_predict(model, pairs[FEATURES], y, cv=folds, method="predict_proba")[:, 1]
report("Logistic regression", (prob >= 0.5).astype(int))

# 4) Save the suspected pairs for the review queue
pairs["dup_prob"] = prob
flagged = pairs[pairs.dup_prob >= 0.5][["doc_a", "doc_b", "dup_prob"]]
conn = sqlite3.connect("data/docusense.db")
flagged.to_sql("duplicate_pairs", conn, if_exists="replace", index=False)
conn.close()
print(f"\nSaved {len(flagged)} suspected duplicate pairs to table duplicate_pairs")