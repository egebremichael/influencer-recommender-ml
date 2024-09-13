import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('titles_with_categories.csv')

# Splitting the dataset
X = df['title']
y = df['category']

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)  
X_features = vectorizer.fit_transform(X)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']} 
model = SVC()  


grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)


model = grid_search.best_estimator_
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(cm)
print(report)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


new_title = "Running Shoes"
new_title_features = vectorizer.transform([new_title])
predicted_category = model.predict(new_title_features)
print(f'Predicted category for "{new_title}": {predicted_category[0]}')
