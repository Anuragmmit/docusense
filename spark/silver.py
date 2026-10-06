from delta.tables import DeltaTable
from pyspark.sql import functions as F

from spark.session import get_spark

spark = get_spark("silver")
B, S = "lake/bronze", "lake/silver"
rd = lambda p: spark.read.format("delta").load(p)

docs = rd(f"{B}/documents").filter("status = 'INGESTED'")
ext = rd(f"{B}/extracted")
qual = rd(f"{B}/image_quality")

df = (ext.join(docs.select("doc_id", "file_name", "sha256", "ingest_date"), "doc_id")
         .join(qual.select("doc_id", "skew_angle", "blur_score"), "doc_id", "left")
         .withColumn("invoice_date", F.to_date("invoice_date", "yyyy-MM-dd"))
         .withColumn("processed_at", F.current_timestamp()))

# Data quality rules: every failed rule adds a reason
reasons = F.concat_ws(",",
    F.when(F.col("invoice_no").isNull(), F.lit("missing_invoice_no")),
    F.when(F.col("vendor_id").isNull(), F.lit("missing_vendor")),
    F.when(F.col("invoice_date").isNull(), F.lit("bad_date")),
    F.when(F.col("invoice_date") > F.current_date(), F.lit("future_date")),
    F.when(F.col("total").isNull() | (F.col("total") <= 0), F.lit("bad_total")),
)
df = df.withColumn("dq_reasons", reasons)
good = df.filter(F.col("dq_reasons") == "").drop("dq_reasons")
bad = df.filter(F.col("dq_reasons") != "")


def merge(frame, path, key="doc_id"):
    if not DeltaTable.isDeltaTable(spark, path):
        frame.write.format("delta").save(path)
    else:
        (DeltaTable.forPath(spark, path).alias("t")
            .merge(frame.alias("s"), f"t.{key} = s.{key}")
            .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())


merge(good, f"{S}/invoices")
merge(bad, f"{S}/quarantine")

# If a row was quarantined earlier but is valid now, remove it from quarantine
if DeltaTable.isDeltaTable(spark, f"{S}/quarantine"):
    (DeltaTable.forPath(spark, f"{S}/quarantine").alias("q")
        .merge(good.select("doc_id").alias("g"), "q.doc_id = g.doc_id")
        .whenMatchedDelete().execute())

print("silver invoices :", rd(f"{S}/invoices").count())
print("quarantined     :", rd(f"{S}/quarantine").count())
rd(f"{S}/quarantine").select("file_name", "dq_reasons").show(10, truncate=False)