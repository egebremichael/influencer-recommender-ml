import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, explode, array, sum as spark_sum, collect_list
from pyspark.sql.types import IntegerType, StringType, FloatType
import re
from textblob import TextBlob

# Initialize SparkSession
spark = SparkSession.builder.appName("GenZAlignment").getOrCreate()



# Process each influencer
influencer_df = spark.read.option("header", "true").csv("Instagram/influencers.txt")
influencers = influencer_df.select("username")

for influencer in influencer_df.rdd.collect():
    username = influencer.username
    path = f"Instagram/METADATA/{username}*"
    df = spark.read.text(path)
    df = df.withColumn("parsed_data", parse_udf(df.value))
    df = df.select(col("parsed_data.caption").alias("caption"), col("parsed_data.comments").alias("comments"))

    aggregated_data = df.agg(
        collect_list("caption").alias("all_captions"),
        collect_list("comments").alias("all_comments")
    ).collect()[0]

    alignment_score = analyze_comments_and_caption(aggregated_data.all_captions, [comment for sublist in aggregated_data.all_comments for comment in sublist])

    # Check if alignment score is above a threshold
    if alignment_score > 30:
        alignment_score = 1

    # Save results
    result = spark.createDataFrame([(username, alignment_score)], ["username", "alignment_score"])
    result.write.csv("Instagram/genz_alignment.csv", mode="append", header=True)


# Gen Z keywords, emojis, hashtags bag of words
gen_z_keywords = [
    'genz', 'mood', 'vibe', 'aesthetic', 'sustainable', 'tiktok', 'viral', 
    'meme', 'cancelled', 'woke', 'squad', 'lit', 'fam', 'goals', 'af', 'stan', 
    'tea', 'shade', 'savage', 'lowkey', 'highkey', 'flex', 'clout', 'yeet', 
    'slay', 'finsta', 'snack', 'ghost', 'simp', 'bet', 'fire', 'salty', 
    'no cap', 'goat', 'sksksk', 'and i oop', 'fomo', 'jomo', 'sus', 'clapback', 
    'spill the tea', 'bop', 'cringe', 'hypebeast', 'drip', 'big yikes', 
    'oomf', 'hits different', 'send it', 'receipts', 'shook', 'glow up', 
    'smh', 'tbh', 'rn', 'tysm', 'irl', 'nft', 'crypto', 'bitcoin', 'blockchain', 
    'influencer', 'stream', 'gig', 'flex', 'chill', 'grind', 'hustle', 
    'iconic', 'on point', 'poppin', 'snatched', 'read', 'gag', 'sis', 'dub', 
    'L', 'W', 'ratioed', 'on fleek', 'deadass', 'boujee', 'thirsty', 'glo up', 
    'bruh', 'finna', 'gucci', 'extra', 'basic', 'doxx', 'swat', 'ghosted', 
    'slaps', 'banger', 'cop', 'drop', 'merch', 'stan twitter', 'main character', 
    'vibe check', 'weird flex', 'ok boomer', 'bet', 'big mood', 'thicc', 
    'skrrt', 'roasted', 'dragged', 'cancel culture', 'bae', 'fleek', 'turnt', 
    'ship', 'OTP', 'doe', 'ratchet', 'turn up', 'squad goals', 'byeee', 
    'yas', 'spicy', 'receipts', 'pressed', 'orbiting', 'deep dive', 'fire', 
    'karen', 'snowflake', 'troll', 'woke', 'cray', 'chad', 'simp', 'emote', 'kek', 
    'pog', 'yeet', 'feels', 'based',  'triggered', 'canceled', 'blue check', 'ratio'
]
gen_z_emojis = [
    '😂', '👌', '🔥', '💯', '🤔', '🥺', '✨', '🙌', '🤡', '💅', '👀', '🙈', '💖', '🌈', '😤', '😳', 
    '🥴', '🤯', '😍', '🤤', '👯‍♂️', '🚀', '🎉', '🕺', '👻', '🤖', '😭', '😎', '😏', '😜', '🤩', '🫶🏽',
    '🙌', '🤌', '🫰', '💪🏾', '🙈', '🙊', '🙉', '🍃', '🩷', '💚', '🩵', '💛', '🤍', '🖤', '☺️'
    '😇', '😌', '🧐', '👽', '🤠', '👑', '🧚‍♀️', '🧜‍♀️', '🐍', '🍑', '🔮', '🎃', '👟', '🧢', '🔒', 
    '🎶', '👁️', '🌟', '💌', '📌', '🧃', '🍵', '🥑', '🌮', '🌯', '🍿', '🍔', '🍕', '🍟', '🧁', '🍩', 
    '🍬', '🍭', '🍮', '🍪', '🥂', '🍾', '🍷', '🍺', '🍸', '🏳️‍🌈', '🏳️‍⚧️', '🛹', '🚴‍♀️', '🧘‍♂️', 
    '🏄‍♀️', '🤿', '🚣', '🎮', '👾', '🕹️', '🎤', '🎧', '📸', '💻', '📱', '🔊', '📢', '🥰', '😷', 
    '🤑', '🤫', '🧠', '🤙', '👊', '✊', '🤘', '🤟', '👁️‍🗨️', '💥', '💢', '💫', '🍓', '🍉', '🥭', 
    '🥦', '🧄', '🥔', '🥩', '🍳', '🥘', '🍲', '🎵', '👗', '🧵', '🛍️', '🎁', '💎', '🧸', '🚨', '🎡', 
    '🏖️', '🏝️', '🌍', '🌞', '🌜', '🌛', '🌚', '🌝', '🌕', '🌖', '🌗', '🌘', '🌑', '🌒', '🌓', '🌔', 
    '🍀', '🌿', '🌱', '🌵', '🌴', '🌲', '🌳', '🌺', '🌸', '🌼', '🌻', '🐾', '🦋', '🌊', '🔑', '💣'
]
gen_z_hashtags = [
    '#l4l', '#instagood', '#viral', '#tiktok', '#meme', '#nofilter', '#selfcare', '#ootd', 
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

# Sentiment Analysis
sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
sentiment_caption = [TextBlob(comment).sentiment.polarity for caption in captions]
sentiments.append(sentiment_caption)

# Visualizing Sentiment Distribution
plt.figure(figsize=(10, 6))
plt.hist(sentiments, bins=5, color='skyblue')
plt.title('Distribution of Sentiments in Instagram Captions and Comments')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.axvline(x=sum(sentiments)/len(sentiments), color='red', linestyle='--', label='Average Sentiment')
plt.legend()
plt.show()

# Keyword Frequency Analysis
all_texts = ' '.join(comments) + ' ' + caption
all_texts = all_texts.lower()

keyword_frequency = {keyword: all_texts.count(keyword) for keyword in gen_z_keywords + gen_z_emojis + gen_z_hashtags if all_texts.count(keyword) > 0}

# Visualizing Keyword Frequency
plt.figure(figsize=(10, 8))
keys = list(keyword_frequency.keys())
values = list(keyword_frequency.values())
plt.barh(keys, values, color='lightgreen')
plt.title('Frequency of Gen Z Keywords, Hashtags, and Emojis')
plt.xlabel('Frequency')
plt.ylabel('Keywords/Hashtags/Emojis')
plt.tight_layout()
plt.show()


spark.stop()