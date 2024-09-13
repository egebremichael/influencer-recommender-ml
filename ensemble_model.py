import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# load the datasetß
df = pd.read_csv('instagram_data.csv') 

# Example brand details
brand = {'category': 'fashion', 'intended_engagement_rate': 0.4, 'target_audience_genZ': 1}
#brand = {'category': 'fitness', 'intended_engagement_rate': 0.4, 'target_audience_genZ': 1}
#brand = {'category': 'beauty', 'intended_engagement_rate': 0.4, 'target_audience_genZ': 1}

# One-hot encode categories
encoder = OneHotEncoder()
encoded_categories = encoder.fit_transform(df[['category']]).toarray()
encoded_brand_category = encoder.transform([[brand['category']]]).toarray()

# Normalize engagement rates
df['normalized_engagement_rate'] = df['engagement_rate'] / df['engagement_rate'].max()
max_engagement_rate = max(df['engagement_rate'].max(), brand['intended_engagement_rate'])
normalized_brand_engagement_rate = brand['intended_engagement_rate'] / max_engagement_rate

# brand vector
brand_vector = np.concatenate(([normalized_brand_engagement_rate], [brand['target_audience_genZ']], encoded_brand_category[0]))

# influencer vectors
df_encoded = pd.concat([df.drop('category', axis=1), pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out())], axis=1)
influencer_vectors = df_encoded[['normalized_engagement_rate', 'alignment_genZ']].values
influencer_vectors = np.hstack((influencer_vectors, encoded_categories))

# Calculate cosine similarity
similarity_scores = cosine_similarity([brand_vector], influencer_vectors)
df['similarity_score'] = similarity_scores.T

recommended_influencers = df.sort_values(by='similarity_score', ascending=False)
threshold = recommended_influencers['similarity_score'].quantile(0.9)

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP = recommended_influencers[recommended_influencers['similarity_score'] >= threshold].shape[0]
total_recommended = len(recommended_influencers) * 0.1  
FP = total_recommended - TP
FN = len(recommended_influencers) - TP - FP

# Calculating Precision, Recall, and F1 Score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")


category_mapping = {'Fashion': 1, 'Beauty': 2, 'Tech': 3}
df['category_simple'] = df['category'].map(category_mapping)
brand_category_simple = category_mapping[brand['category']]

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot influencers
ax.scatter(df['normalized_engagement_rate'], df['alignment_genZ'], df['category_simple'], c='blue', marker='o', s=100, label='Influencers')

# Plot brand
ax.scatter([normalized_brand_engagement_rate], [brand['target_audience_genZ']], [brand_category_simple], c='red', marker='^', s=200, label='Brand')

ax.set_xlabel('Normalized Engagement Rate')
ax.set_ylabel('Alignment with GenZ')
ax.set_zlabel('Category')
ax.set_title('3D Visualization of Brand and Influencers')
ax.legend()

true_labels = (df['similarity_score'] >= threshold).astype(int)
# Calculate the ROC curve
fpr, tpr, _ = roc_curve(true_labels, df['similarity_score'])
roc_auc = auc(fpr, tpr)

# Calculate the Precision-Recall curve
precision, recall, _ = precision_recall_curve(true_labels, df['similarity_score'])
pr_auc = average_precision_score(true_labels, df['similarity_score'])

# Plot the ROC curve
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Plot the Precision-Recall curve
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()