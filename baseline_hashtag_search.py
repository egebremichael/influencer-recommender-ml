from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, explode, array, sum as spark_sum, collect_list
from pyspark.sql.types import IntegerType, StringType, FloatType
import re
from textblob import TextBlob

# Initialize SparkSession
spark = SparkSession.builder.appName("GenZAlignment").getOrCreate()



@udf(StringType())
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

preprocess_text_udf = udf(preprocess_text, StringType())


@udf(IntegerType())
def search_through_hashtag(text):
    gen_z_hashtags = [
        '#asos', '#l4l', '#instagood', '#viral', '#tiktok', '#meme', '#nofilter', '#selfcare', '#ootd', 
        '#onfleek', '#blessed', '#snatched', '#yolo', '#swag', '#bae', '#squadgoals', '#fomo', 
        '#adulting', '#tbh', '#tbt', '#fbf', '#lit', '#woke', '#savage', '#mood', '#vibes', 
        '#thirsty', '#stan', '#flex', '#clout', '#ship', '#glowup', '#cancelculture', '#spillthetea', 
        '#maincharacter', '#nft', '#crypto', '#streamerlife', '#selflove', '#hustle', '#grind', 
        '#goalsaf', '#relationshipgoals', '#fitnessgoals', '#aesthetic', '#sustainable', '#ecoconscious', 
        '#vegan', '#plantbased', '#foodie', '#travelgram', '#explore', '#adventure', '#digitalnomad', 
        '#influencer', '#styleinspo', '#fashionista', '#beautyblogger', '#gaming', '#esports', '#fitfam', 
        '#wellness', '#mindfulness', '#mentalhealthawareness', '#bodypositivity', '#activism', '#pride', 
        '#loveislove', '#equality', '#blm', '#stopasianhate', '#climatechange', '#savetheplanet', '#zerowaste', 
        '#minimalism', '#tinyhouse', '#vanlife', '#diy', '#art', '#handmade', '#kpop', '#anime', '#manga', 
        '#netflixandchill', '#marvel', '#dccomics', '#cosplay', '#fandom', '#throwback', '#flashback', 
        '#trendy', '#trending', '#hot', '#cool', '#edgy', '#hip', '#fresh', '#unique', '#rare', '#exclusive', 
        '#hype', '#hypebeast', '#tech', '#geek', '#nerd', '#science', '#innovation', '#startup', '#entrepreneur'
    ]
    text = preprocess_text(text)
    blob = TextBlob(text)
    tokens = blob.words

    characteristics_found += sum(1 for hashtag in gen_z_hashtags if hashtag in text)
    
    return characteristics_found


def analyze_captions(all_captions):
    total_characteristics = search_through_hashtag(all_captions)

    for data in aggregated_data:
        total_captions = len(data.all_captions)
    positive_sentiments = 0
    for caption in data.all_captions:
        if TextBlob(caption).sentiment.polarity > 0:
            positive_sentiments += 1

    positive_sentiment_ratio = positive_sentiments / max(total_captions, 1)
    alignment_score = (total_characteristics + positive_sentiment_ratio * 100) / (total_captions) 
    
    return alignment_score


influencer_df = spark.read.option("header", "true").csv("Instagram/influencers.txt")
influencers = influencer_df.select("username")

for influencer in influencer_df.rdd.collect():
    username = influencer.username
    path = f"Instagram/METADATA/{username}*"
    df = spark.read.text(path)
    df = df.withColumn("parsed_data", parse_udf(df.value))
    df = df.select(col("parsed_data.caption").alias("caption"))

    aggregated_data = df.agg(
        collect_list("caption").alias("all_captions"),
    ).collect()[0]

    hashtag_score = analyze_captions(aggregated_data.all_captions, [comment for sublist in aggregated_data.all_comments for comment in sublist])

    result = spark.createDataFrame([(username, hashtag_score)], ["username", "hashtag_score"])
    result.write.csv("Instagram/hashtag_search.csv", mode="append", header=True)


spark.stop()


