from pyspark.sql import Window
from pyspark.sql import functions as F

from spark.session import get_spark

spark = get_spark("gold")
S, G = "lake/silver", "lake/gold"
inv = spark.read.format("delta").load(f"{S}/invoices")


def save(df, name):
    # Gold is rebuilt from Silver every time, so overwrite is safe.
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .save(f"{G}/{name}")
    print(name, df.count())


# FACT: one row per invoice, with numbers and keys only
fact = inv.select(
    "doc_id", "invoice_no", "vendor_id",
    F.date_format("invoice_date", "yyyyMMdd").cast("int").alias("date_key"),
    "subtotal", "gst", "total", "needs_review", "mean_conf", "skew_angle", "blur_score")
save(fact, "fact_invoices")

# DIM VENDOR: most common GSTIN per vendor (OCR may misread a few)
w = Window.partitionBy("vendor_id").orderBy(F.desc("n"))
dim_vendor = (inv.groupBy("vendor_id", "vendor_name", "gstin").agg(F.count("*").alias("n"))
                 .withColumn("rk", F.row_number().over(w)).filter("rk = 1")
                 .select("vendor_id", "vendor_name", "gstin"))
save(dim_vendor, "dim_vendor")

# DIM DATE: a calendar table
dim_date = (spark.sql("SELECT explode(sequence(to_date('2026-01-01'), "
                      "to_date('2026-12-31'), interval 1 day)) AS d")
    .select(F.date_format("d", "yyyyMMdd").cast("int").alias("date_key"),
            F.col("d").alias("date"), F.year("d").alias("year"),
            F.month("d").alias("month"), F.date_format("d", "MMMM").alias("month_name"),
            F.quarter("d").alias("quarter"), F.date_format("d", "E").alias("weekday")))
save(dim_date, "dim_date")

# AGGREGATE: what Power BI will love
monthly = (fact.join(dim_date, "date_key")
    .groupBy("vendor_id", "year", "month")
    .agg(F.count("*").alias("invoice_count"),
         F.round(F.sum("total"), 2).alias("total_spend"),
         F.round(F.avg("total"), 2).alias("avg_invoice")))
save(monthly, "gold_vendor_monthly")