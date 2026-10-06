import sqlite3

import pandas as pd

conn = sqlite3.connect("data/docusense.db")
ext = pd.read_sql("""SELECT e.*, d.file_name FROM extracted_fields e
                     JOIN documents d USING(doc_id)""", conn)
gt = pd.read_csv("data/ground_truth.csv")
df = ext.merge(gt, left_on="file_name", right_on="file", suffixes=("", "_gt"))

correct = pd.DataFrame({
    "invoice_no": df.invoice_no == df.invoice_no_gt,
    "invoice_date": df.invoice_date == df.invoice_date_gt,
    "gstin": df.gstin == df.gstin_gt,
    "vendor_id": df.vendor_id == df.vendor_id_gt,
    "subtotal": (df.subtotal - df.subtotal_gt).abs() < 0.01,
    "gst": (df.gst - df.gst_gt).abs() < 0.01,
    "total": (df.total - df.total_gt).abs() < 0.01,
})

print(f"Documents evaluated: {len(df)}\n")
print("Per-field accuracy:")
print(correct.mean().round(3))

all_ok = correct.all(axis=1)
auto = df.needs_review == 0
print(f"\nFully correct documents:        {all_ok.mean():.3f}")
print(f"Straight-through rate (no human needed): {auto.mean():.3f}")
if auto.sum():
    print(f"Accuracy of auto-approved docs: {all_ok[auto].mean():.3f}")
print(f"Sent to human review:           {(~auto).sum()} docs")
print(f"  ...of which were actually wrong: {(~all_ok[~auto]).sum()}")
print(f"Wrong but NOT flagged (dangerous): {(~all_ok[auto]).sum()}")