import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


categories_url = "https://asos2.p.rapidapi.com/categories/list"
categories_params = {
  "country": "US"
}
headers = {
    'X-RapidAPI-Key': 'f8f91464a8mshfabc82f63b3ad7dp1c2959jsn67f3f6088bcf',
    'X-RapidAPI-Host': 'asos2.p.rapidapi.com'
}


categories_response = requests.get(categories_url, headers=headers, params=categories_params)
categories_data = categories_response.json()


def collect_category_titles(categories):
    titles = []
    
    for category in categories:
        title = category.get('content', {}).get('title')
        if title:
            titles.append(title)
        
        # Handle nested categories
        children = category.get('children', [])
        titles.extend(collect_category_titles(children))
    
    return titles

navigation_data = categories_data.get('navigation', [])
categories_df = collect_category_titles(navigation_data)

# Filter out irrelevant categories
irrelevant_keywords = [
    'Men', 'Home', 'App and Mobile Top Level - Carousel', 'List', 'UP TO 50% OFF!', 'Categories', 'View all',
    'App & Mobile Promo', 'Unlimited Next Day Delivery', 'NEW PRODUCTS', 'New in: Today', 'Download the app!',
    'Shop By Trend', 'Shop By Occasion', 'SHOP BY PRODUCT', 'SHOP BY BRAND', 'SHOP BY TRAINER STYLE', 
    'SHOP BY BAGS', 'SHOP BY JEWELRY', 'Browse by', 'Shop by activity', 'Shop by body fit', 'Shop by Size', 
    'Shop by Dresses', 'A-Z Brands Link', 'A-Z of brands', 'Top Brands', 'ASOS Brands', 'Discover Brands', 
    'View all brands', 'CTAs', 'CTAs New', 'CTA1', 'SS24 ACCESSORIES', 'TRAINER CTAs', 'SALE: UP TO 70% OFF', 
    'Final sale', 'Last chance to buy', 'SALE Selling fast', 'SALE View all', 'Discover More', 'More ASOS', 
    'Unlimited express delivery', '10% Student Discount', 'Gift Vouchers', 'Size XS', 'Size S', 'Size M', 
    'Size L', 'Size XL', 'Size XXL', 'Wide Fit', 'Plus Size', 'Tall', 'Petite', 'Maternity', 'Nike', 'ASOS Design', 
    'Columbia', 'Topman', 'Ape By A Bathing Ape', 'Adidas', 'Converse', 'Dr Martens', 'Vans', 'UGG', 'New Balance', 
    'Crocs', 'Puma', 'Reebok', 'River Island', 'Jack & Jones', 'Bershka', 'Pull&Bear', 'The North Face', 
    'Tommy Hilfiger', 'Stradivarius', 'Miss Selfridge', '& Other Stories', 'Monki', 'Mango', 'COLLUSION', 'Topshop'
]

# Initialize TfidfVectorizer
custom_stop_words = list(ENGLISH_STOP_WORDS.union(irrelevant_keywords))
vectorizer = TfidfVectorizer(stop_words=custom_stop_words, ngram_range=(1, 2), max_df=0.5, min_df=0.01)

X = vectorizer.fit_transform(categories_df)
X_reduced = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random').fit_transform(X.toarray())

# Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_reduced)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='black', s=100)
plt.title('Category Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


for i in range(3):  
    print(f"Cluster {i}:")
    cluster_categories = categories_df[clusters == i]
    print(cluster_categories.tolist())
    print("\n")

# Silhouette Analysis

# Calculate silhouette scores
silhouette_avg = silhouette_score(X_reduced, clusters)
sample_silhouette_values = silhouette_samples(X_reduced, clusters)
n_clusters = len(np.unique(clusters))

# silhouette plot
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(18, 7)
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(X_reduced) + (n_clusters + 1) * 10])

y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = \
        sample_silhouette_values[clusters == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10  

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([]) 
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')

plt.show()


