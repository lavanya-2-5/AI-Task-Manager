import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle

# Step 1: Load the labeled dataset
df = pd.read_csv('labeled_tasks.csv')

# Step 2: Drop NaN values from 'cleaned_description' and 'priority'
df = df.dropna(subset=['cleaned_description', 'Priority'])

# Step 3: Features and labels
X = df['cleaned_description']
y = df['Priority']

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Train Random Forest with GridSearchCV
model = RandomForestClassifier()
params = {'n_estimators': [100, 150], 'max_depth': [None, 10, 20]}
grid = GridSearchCV(model, param_grid=params, cv=3)
grid.fit(X_vectorized, y)

# Step 6: Save best model and vectorizer
with open('priority_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model Trained and Saved Successfully!")