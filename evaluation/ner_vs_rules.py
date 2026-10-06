import re
import sqlite3
from difflib import SequenceMatcher

import pandas as pd
import spacy

conn = sqlite3.connect("data/docusense.db")
df = pd.read_sql("""
    SELECT d.doc_id, d.file_name, o.text,
           e.invoice_no r_inv, e.invoice_date r_date, e.gstin r_gstin,
           e.vendor_id r_vendor, e.subtotal r_sub, e.gst r_gst, e.total r_total
    FROM documents d
    JOIN ocr_results o USING(doc_id)
    JOIN extracted_fields e USING(doc_id)""", conn)
split = pd.read_csv("data/split.csv")
gt = pd.read_csv("data/ground_truth.csv")
df = df.merge(split, on="doc_id").merge(gt, left_on="file_name", right_on="file")
df = df[df.split == "test"].reset_index(drop=True)
master = gt[["vendor_id", "vendor_name"]].drop_duplicates().values.tolist()
print(f"Test documents (never seen in training): {len(df)}\n")

nlp = spacy.load("models/ner")


def money(s):
    if s is None:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", s))
    except ValueError:
        return None


def vendor_id_from(name):
    if not name:
        return None
    best = max(master, key=lambda m: SequenceMatcher(None, m[1].lower(), name.lower()).ratio())
    ok = SequenceMatcher(None, best[1].lower(), name.lower()).ratio() >= 0.8
    return best[0] if ok else None


def predict(text):
    by = {}
    for ent in nlp(text).ents:
        by.setdefault(ent.label_, []).append(ent.text)
    first = lambda l: by[l][0] if l in by else None
    last = lambda l: by[l][-1] if l in by else None
    inv = first("INV_NO")
    return {
        "invoice_no": re.sub(r"\s+", "", inv).upper() if inv else None,
        "invoice_date": first("DATE"),
        "gstin": first("GSTIN"),
        "vendor_id": vendor_id_from(first("VENDOR")),
        "subtotal": money(first("SUBTOTAL")),
        "gst": money(first("GST")),
        "total": money(last("TOTAL")),
    }


ner = pd.DataFrame([predict(t) for t in df.text])
rules = pd.DataFrame({
    "invoice_no": df.r_inv, "invoice_date": df.r_date, "gstin": df.r_gstin,
    "vendor_id": df.r_vendor, "subtotal": df.r_sub, "gst": df.r_gst, "total": df.r_total})
hybrid = rules.where(rules.notna(), ner)    # use rules; fall back to NER if rules found nothing

truth = pd.DataFrame({
    "invoice_no": df.invoice_no, "invoice_date": df.invoice_date, "gstin": df.gstin,
    "vendor_id": df.vendor_id, "subtotal": df.subtotal, "gst": df.gst, "total": df.total})
NUM = ["subtotal", "gst", "total"]


def correct(pred):
    out = pd.DataFrame(index=pred.index)
    for c in truth.columns:
        if c in NUM:
            out[c] = (pd.to_numeric(pred[c]) - truth[c]).abs() < 0.01
        else:
            out[c] = pred[c] == truth[c]
    return out


report = {}
for name, pred in [("rules", rules), ("ner", ner), ("hybrid", hybrid)]:
    c = correct(pred)
    report[name] = {**c.mean().to_dict(), "all_fields_correct": c.all(axis=1).mean(),
                    "avg_field_accuracy": c.mean().mean()}
report = pd.DataFrame(report).T
print("Accuracy on unseen test documents (1.0 = 100%)")
print(report.round(3).T, "\n")

b = report.loc["rules", "avg_field_accuracy"]
for name in ["ner", "hybrid"]:
    a = report.loc[name, "avg_field_accuracy"]
    rel = (a - b) / b * 100 if b else float("nan")
    print(f"{name:6s} vs rules -> absolute {(a-b)*100:+.1f} pts | relative {rel:+.1f}%")