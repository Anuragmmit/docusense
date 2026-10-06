from delta.tables import DeltaTable

from spark.session import get_spark

spark = get_spark("verify")

print("\nTop 5 vendors by spend:")
spark.sql("""
  SELECT v.vendor_name, COUNT(*) invoices, ROUND(SUM(f.total),2) spend
  FROM delta.`lake/gold/fact_invoices` f
  JOIN delta.`lake/gold/dim_vendor` v USING (vendor_id)
  GROUP BY v.vendor_name ORDER BY spend DESC LIMIT 5""").show(truncate=False)

print("History of the silver table (every change is recorded):")
DeltaTable.forPath(spark, "lake/silver/invoices").history() \
    .select("version", "timestamp", "operation").show(truncate=False)

print("Time travel: row count at version 0 vs now")
v0 = spark.read.format("delta").option("versionAsOf", 0).load("lake/silver/invoices").count()
now = spark.read.format("delta").load("lake/silver/invoices").count()
print(v0, now)