from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, col, avg
from pyspark.sql.types import FloatType, IntegerType

# Initialize SparkSession
spark = SparkSession.builder.appName("EngagementRateCalculator").getOrCreate()

# Load influencers data
influencer_df = spark.read.option("header", "true").csv("Instagram/influencers.txt")
influencer_df = influencer_df.withColumnRenamed("_c2", "followers")  
influencer_df = influencer_df.withColumn("followers", influencer_df["followers"].cast(IntegerType()))

# Define a UDF to extract likes from metadata
@udf(FloatType())
def extract_likes(metadata_content):
    import json
    try:
        metadata = json.loads(metadata_content)
        return float(metadata['like_count'])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0

# Process each influencer
results = []
for influencer in influencer_df.collect():
    username = influencer.username
    followers = influencer.followers
    path = f"Instagram/METADATA/{username}*"
    
    df = spark.read.text(path)
    df = df.withColumn("likes", extract_likes(df.value))
    
    if followers > 0:  
        df = df.withColumn("engagement_rate", (col("likes") / followers) * 100)
        average_engagement_rate = df.agg(avg("engagement_rate")).first()[0]
    else:
        average_engagement_rate = 0
    
    results.append((username, average_engagement_rate))


results_df = spark.createDataFrame(results, ["username", "average_engagement_rate"])
results_df.write.csv("Instagram/engagement_rates.csv", mode="overwrite", header=True)


spark.stop()
