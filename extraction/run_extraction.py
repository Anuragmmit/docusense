import sqlite3

import pandas as pd

from extraction.fields import extract, validate

conn = sqlite3.connect("data/docusense.db")

gt = pd.read_csv("data/ground_truth.csv")
master = gt[["vendor_id", "vendor_name"]].drop_duplicates().values.tolist()

# Rules change often and this step is fast, so we rebuild the table each run.
conn.execute("DROP TABLE IF EXISTS extracted_fields")
conn.execute("""CREATE TABLE extracted_fields (
    doc_id TEXT PRIMARY KEY, invoice_no TEXT, invoice_date TEXT, gstin TEXT,
    vendor_id TEXT, vendor_name TEXT, vendor_match REAL,
    subtotal REAL, gst REAL, total REAL,
    mean_conf REAL, flags TEXT, needs_review INTEGER)""")

for doc_id, text, conf in conn.execute(
        "SELECT doc_id, text, mean_conf FROM ocr_results").fetchall():
    f = extract(text, master)
    flags = validate(f, conf)
    conn.execute("INSERT INTO extracted_fields VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (doc_id, f["invoice_no"], f["invoice_date"], f["gstin"],
                  f["vendor_id"], f["vendor_name"], f["vendor_match"],
                  f["subtotal"], f["gst"], f["total"],
                  conf, ",".join(flags), int(len(flags) > 0)))
conn.commit()
print("Extraction finished")
conn.close()