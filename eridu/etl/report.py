import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

spark: SparkSession = SparkSession.builder.appName("Eridu ETL Report").getOrCreate()

pairs_df: DataFrame = spark.read.parquet("data/pairs-all.parquet")


# Do we have positive and negative pairs?
pairs_df.groupBy("match").count().orderBy("count", ascending=False).show()

# What are the categories? Note that they aren't always the same.
pairs_df.groupBy("left_category", "right_category").count().orderBy("count", ascending=False).show()

# What about language pairs?
pairs_df.groupBy("left_lang", "right_lang").count().orderBy("count", ascending=False).show()

# Let's look at a sample of matching names...
pairs_df.select("left_name", "right_name", "match").limit(10).show(20, truncate=False)

# How many single word names are there vs. multi-word names?
pairs_df.filter(
    (F.size(F.split(pairs_df.left_name, " ")) == 1) & (pairs_df.left_lang == pairs_df.right_lang)
).select("left_name", "right_name", "match").show(20, truncate=36)
