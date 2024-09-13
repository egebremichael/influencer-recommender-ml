from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']


encoder = OneHotEncoder(categories=[CATEGORIES])
encoder.fit(np.array(CATEGORIES).reshape(-1, 1))

@app.route('/get_similarities', methods=['GET'])
def get_similarities():
    category = request.args.get('category', 'fitness')
    if not category or category.lower() not in [cat.lower() for cat in CATEGORIES]:
        return jsonify({'error': 'Invalid or missing category'}), 400

    print("Category requested:", category)

    # Load the dataset
    df = pd.read_csv('instagram_data.csv')
    df['category'] = df['category'].str.lower().str.strip()  # Normalize categories

    df_filtered = df[df['category'] == category.lower()]
    if df_filtered.empty:
        return jsonify({'error': 'No entries found for the specified category'}), 404

    print("Filtered DataFrame shape:", df_filtered.shape)
    df_filtered['normalized_engagement_rate'] = df_filtered['engagement_rate'] / df_filtered['engagement_rate'].max()

    # brand vector
    brand = {'category': category, 'intended_engagement_rate': 0.5, 'target_audience_genZ': 1}
    normalized_brand_engagement_rate = brand['intended_engagement_rate'] / max(df_filtered['engagement_rate'].max(), 0.5)
    encoded_brand_category = encoder.transform([[brand['category'].lower()]]).toarray()
    brand_vector = np.concatenate(([normalized_brand_engagement_rate], [brand['target_audience_genZ']], encoded_brand_category[0]))

    # influencer vectors
    influencer_vectors = np.hstack((df_filtered[['normalized_engagement_rate', 'alignment_genZ']].values, encoder.transform(df_filtered[['category']]).toarray()))

    # Calculate cosine similarity
    similarity_scores = cosine_similarity([brand_vector], influencer_vectors)
    df_filtered['similarity_score'] = similarity_scores.T

    # Sort by similarity score and return top influencers
    recommended_influencers = df_filtered.sort_values(by='similarity_score', ascending=False)
    print("Top influencers:", recommended_influencers[['username', 'similarity_score']].head(15))

    return jsonify(recommended_influencers[['username', 'similarity_score']].head(15).to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
 