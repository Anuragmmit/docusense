import sqlite3
import pandas as pd

conn = sqlite3.connect("data/docusense.db")
q = lambda s: conn.execute(s).fetchone()[0]
print("INGESTED docs :", q("SELECT COUNT(*) FROM documents WHERE status='INGESTED'"))
print("cleaned       :", q("SELECT COUNT(*) FROM image_quality"))
print("OCR'd         :", q("SELECT COUNT(*) FROM ocr_results"))

df = pd.read_sql("SELECT * FROM extracted_fields", conn)
cols = ["invoice_no", "invoice_date", "gstin", "vendor_id", "subtotal", "gst", "total"]
print("\nShare of documents where each field is MISSING:")
print(df[cols].isna().mean().round(3))
print("\nMost common flags:")
print(df.flags.str.split(",").explode().value_counts().head(8))
print("\nOCR confidence:")
print(df.mean_conf.describe().round(1))

# Show the OCR text of one failing document
bad = df[df.needs_review == 1].iloc[0]
print("\nFlags for this doc:", bad.flags)
print("----- OCR TEXT -----")
print(conn.execute("SELECT text FROM ocr_results WHERE doc_id=?", (bad.doc_id,)).fetchone()[0])