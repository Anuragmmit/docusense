import sqlite3

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.load import load_data

SEED = 42
REVIEW_BUDGET = 0.05          # reviewers can look at the top 5% most suspicious

df = load_data()
n_before = len(df)
df = df[(df.is_duplicate == 0) & df.total.notna() & (df.total > 0)
        & df.vendor_id.notna()].reset_index(drop=True)
print(f"Scoring {len(df)} invoices ({n_before - len(df)} excluded: duplicates or missing vendor/total)")
y = df.is_anomaly

# Features. Money is skewed, so we use logs; then ask "how unusual is this
# invoice FOR THIS VENDOR?" using median and MAD (robust to the outliers themselves).
df["log_total"] = np.log1p(df.total)
med = df.groupby("vendor_id").log_total.transform("median")
mad = df.groupby("vendor_id").log_total.transform(lambda s: (s - s.median()).abs().median())
df["vendor_z"] = (df.log_total - med) / (mad.clip(lower=0.1) * 1.4826)
df["global_z"] = (df.total - df.total.mean()) / df.total.std()

FEATURES = ["log_total", "vendor_z"]
iso = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED)
iso.fit(df[FEATURES])
df["iso_score"] = -iso.score_samples(df[FEATURES])      # higher = weirder

k = int(y.sum())                                        # k = number of true anomalies
budget_n = max(1, int(REVIEW_BUDGET * len(df)))


def evaluate(col):
    ranked = df.sort_values(col, ascending=False)
    top_k = ranked.head(k).is_anomaly
    top_b = ranked.head(budget_n).is_anomaly
    return {
        "roc_auc": roc_auc_score(y, df[col]),
        "avg_precision": average_precision_score(y, df[col]),
        f"precision@{k}": top_k.mean(),
        f"recall@budget({budget_n})": top_b.sum() / k,
    }


report = pd.DataFrame({
    "global z-score (baseline)": evaluate("global_z"),
    "vendor z-score": evaluate("vendor_z"),
    "isolation forest": evaluate("iso_score"),
}).T
print(f"Invoices: {len(df)}   true anomalies: {k}\n")
print(report.round(3), "\n")

# Save scores for dashboards / review queue
df["flagged"] = (df.iso_score.rank(ascending=False) <= budget_n).astype(int)
conn = sqlite3.connect("data/docusense.db")
df[["doc_id", "iso_score", "vendor_z", "flagged"]].to_sql(
    "anomaly_scores", conn, if_exists="replace", index=False)
conn.close()

# Experiment tracking (open it with:  mlflow ui)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("anomaly_detection")
with mlflow.start_run(run_name="isolation_forest"):
    mlflow.log_params({"n_estimators": 200, "features": ",".join(FEATURES),
                       "seed": SEED, "review_budget": REVIEW_BUDGET})
    mlflow.log_metrics({k_.replace("@", "_at_").replace("(", "_").replace(")", ""): float(v)
                        for k_, v in report.loc["isolation forest"].items()})
print("Saved table anomaly_scores and logged the run to MLflow")