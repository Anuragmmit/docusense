import sqlite3
import pandas as pd

conn = sqlite3.connect("data/docusense.db")
ext = pd.read_sql("""SELECT e.*, d.file_name FROM extracted_fields e
                     JOIN documents d USING(doc_id)""", conn)
gt = pd.read_csv("data/ground_truth.csv")
df = ext.merge(gt, left_on="file_name", right_on="file", suffixes=("", "_gt"))

print("Most common flags:")
print(df["flags"].str.split(",").explode().value_counts().head(8), "\n")

for col in ["invoice_no", "invoice_date", "gstin", "vendor_id"]:
    bad = df[df[col] != df[col + "_gt"]]
    print(f"--- {col}: {len(bad)} wrong. Examples (found vs expected):")
    print(bad[[col, col + "_gt"]].head(5).to_string(index=False), "\n")

for col in ["subtotal", "gst", "total"]:
    bad = df[(df[col] - df[col + "_gt"]).abs().fillna(1) > 0.01]
    print(f"--- {col}: {len(bad)} wrong. Examples (found vs expected):")
    print(bad[[col, col + "_gt"]].head(5).to_string(index=False), "\n")