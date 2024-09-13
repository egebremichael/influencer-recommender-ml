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
def analyze_text_for_genz_traits(text):
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
    
    characteristics_found = sum(1 for word in tokens if word in gen_z_keywords)
    characteristics_found += sum(1 for char in text if char in gen_z_emojis)
    characteristics_found += sum(1 for hashtag in gen_z_hashtags if hashtag in text)
    
    return characteristics_found


def analyze_comments_and_caption(all_captions, all_comments):
    """Combine analysis of caption and comments for Gen Z alignment."""
    total_characteristics = analyze_text_for_genz_traits(all_captions)
    total_characteristics += analyze_text_for_genz_traits(all_comments)

    for data in aggregated_data:
        total_captions = len(data.all_captions)
        total_comments = sum(len(comment_list) for comment_list in data.all_comments)
    positive_sentiments = 0
    for caption in data.all_captions:
        if TextBlob(caption).sentiment.polarity > 0:
            positive_sentiments += 1
    for comment in data.all_comments:
        if TextBlob(comment).sentiment.polarity > 0:
            positive_sentiments += 1

    positive_sentiment_ratio = positive_sentiments / max(total_comments + total_captions, 1)
    alignment_score = (total_characteristics + positive_sentiment_ratio * 100) / (total_comments + total_captions) 
    
    return alignment_score


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
    result = spark.createDataFrame([(username, alignment_score)], ["username", "alignment_genZ"])
    result.write.csv("Instagram/genz_alignment.csv", mode="append", header=True)


spark.stop()
