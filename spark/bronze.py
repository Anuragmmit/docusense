import sqlite3

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import functions as F

from spark.session import get_spark

spark = get_spark("bronze")
conn = sqlite3.connect("data/docusense.db")
B = "lake/bronze"


def load(sql, schema):
    pdf = pd.read_sql(sql, conn)
    pdf = pdf.astype(object).where(pdf.notna(), None)   # NaN -> None
    return spark.createDataFrame(pdf, schema=schema)


def upsert(df, path, key, update=False):
    """First run: create the table. Later runs: MERGE on the key (no duplicates)."""
    if not DeltaTable.isDeltaTable(spark, path):
        df.write.format("delta").save(path)
        return
    t = DeltaTable.forPath(spark, path)
    m = t.alias("t").merge(df.alias("s"), f"t.{key} = s.{key}")
    if update:
        m = m.whenMatchedUpdateAll()
    m.whenNotMatchedInsertAll().execute()


# 1) Document registry, partitioned by ingest date
docs = load("SELECT * FROM documents", """
    doc_id string, file_name string, source_path string, bronze_path string,
    sha256 string, size_bytes long, source string, ingested_at string,
    status string, duplicate_of string, error_message string""")
docs = docs.withColumn("ingest_date", F.to_date(F.substring("ingested_at", 1, 10)))
path = f"{B}/documents"
if not DeltaTable.isDeltaTable(spark, path):
    docs.write.format("delta").partitionBy("ingest_date").save(path)
else:
    upsert(docs, path, "doc_id")

# 2) OCR text, image quality, extracted fields
upsert(load("SELECT doc_id, text, mean_conf, n_words FROM ocr_results",
            "doc_id string, text string, mean_conf double, n_words long"),
       f"{B}/ocr", "doc_id")
upsert(load("SELECT doc_id, skew_angle, blur_score, processed_path FROM image_quality",
            "doc_id string, skew_angle double, blur_score double, processed_path string"),
       f"{B}/image_quality", "doc_id")
# Extraction rules change, so here existing rows get UPDATED too
upsert(load("SELECT * FROM extracted_fields", """
    doc_id string, invoice_no string, invoice_date string, gstin string,
    vendor_id string, vendor_name string, vendor_match double,
    subtotal double, gst double, total double,
    mean_conf double, flags string, needs_review long"""),
       f"{B}/extracted", "doc_id", update=True)

for name in ["documents", "ocr", "image_quality", "extracted"]:
    print(name, spark.read.format("delta").load(f"{B}/{name}").count())