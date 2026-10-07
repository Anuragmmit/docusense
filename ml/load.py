import sqlite3

import pandas as pd


def load_data():
    conn = sqlite3.connect("data/docusense.db")
    ext = pd.read_sql("""SELECT e.*, d.file_name FROM extracted_fields e
                         JOIN documents d USING(doc_id)""", conn)
    conn.close()
    gt = pd.read_csv("data/ground_truth.csv")
    return ext.merge(gt[["file", "invoice_no", "is_duplicate", "is_anomaly"]],
                     left_on="file_name", right_on="file",
                     suffixes=("", "_gt"))