from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("MergeDatasets").getOrCreate()

# Load datasets
dataset1 = spark.read.csv("Instagram/engagement_rates.csv", header=True, inferSchema=True)
dataset2 = spark.read.csv("Instagram/genz_alignment.csv", header=True, inferSchema=True)
dataset3 = spark.read.csv("Instagram/hashtag_search.csv", header=True, inferSchema=True)

# Join datasets
merged_inner = dataset1.join(dataset2, dataset1.username == dataset2.username, "inner").join(dataset3, dataset1.username == dataset3.username, "inner")
merged_inner.show()
merged_inner.write.csv("instagram_data.csv", mode="overwrite", header=True)

# Stop Spark session
spark.stop()
