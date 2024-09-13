from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import warnings

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Categories for OneHotEncoder
CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']

# Initialize OneHotEncoder
encoder = OneHotEncoder(categories=[CATEGORIES], sparse_output=False)  # Changed to sparse_output
encoder.fit(np.array(CATEGORIES).reshape(-1, 1))

# Suppress specific warning if necessary (optional)
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/target')
def serve_target():
    category = request.args.get('category')
    if category not in CATEGORIES:
        return jsonify({'error': 'Invalid category'}), 400

    # Redirect to target-page.html and let client-side JavaScript handle the data fetching
    return send_from_directory('static', 'target-page.html')

@app.route('/get_similarities', methods=['GET'])
def get_similarities():
    category = request.args.get('category', 'fitness')
    if not category or category.lower() not in [cat.lower() for cat in CATEGORIES]:
        return jsonify({'error': 'Invalid or missing category'}), 400

    # Load the dataset
    df = pd.read_csv('instagram_data.csv')
    df['category'] = df['category'].str.lower().str.strip()  # Normalize categories

    # Filter and copy the DataFrame
    df_filtered = df[df['category'] == category.lower()].copy()
    if df_filtered.empty:
        return jsonify({'error': 'No entries found for the specified category'}), 404

    # Calculate normalized engagement rate
    df_filtered.loc[:, 'normalized_engagement_rate'] = df_filtered['engagement_rate'] / df_filtered['engagement_rate'].max()

    # Brand vector
    brand = {'category': category, 'intended_engagement_rate': 0.5, 'target_audience_genZ': 1}
    normalized_brand_engagement_rate = brand['intended_engagement_rate'] / max(df_filtered['engagement_rate'].max(), 0.5)
    encoded_brand_category = encoder.transform([[brand['category'].lower()]])
    brand_vector = np.concatenate(([normalized_brand_engagement_rate], [brand['target_audience_genZ']], encoded_brand_category[0]))

    # Influencer vectors
    influencer_vectors = np.hstack((
        df_filtered[['normalized_engagement_rate', 'alignment_genZ']].values,
        encoder.transform(df_filtered[['category']])
    ))

    # Calculate cosine similarity
    similarity_scores = cosine_similarity([brand_vector], influencer_vectors)
    df_filtered.loc[:, 'similarity_score'] = similarity_scores.T

    # Sort by similarity score and return top influencers
    recommended_influencers = df_filtered.sort_values(by='similarity_score', ascending=False)

    return jsonify(recommended_influencers[['username', 'similarity_score']].head(15).to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
