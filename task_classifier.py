import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
ps = PorterStemmer()

# Step 1: Load Dataset
df = pd.read_csv("preprocessed_tasks.csv")

# Step 2: Drop rows with missing description or priority
df = df.dropna(subset=["Description", "Priority"])

# Step 3: Text preprocessing
def preprocess(text):
    if pd.isnull(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Step 4: Apply preprocessing
df["cleaned_description"] = df["Description"].apply(preprocess)

# Step 5: Remove blank rows
df = df[df["cleaned_description"].str.strip() != ""]

# Step 6: Map Priority to numeric values
priority_map = {"low": 0, "medium": 1, "high": 2}
df["Priority"] = df["Priority"].str.lower().map(priority_map)

# Step 7: Drop if Priority still NaN
df = df.dropna(subset=["Priority"])

# Step 8: Define features and target
X = df["cleaned_description"]
y = df["Priority"]

# Step 9: Vectorize text
tfidf = TfidfVectorizer(max_features=5000)
try:
    X_vectorized = tfidf.fit_transform(X).toarray()
except ValueError as e:
    print("❌ Error in vectorization:", e)
    print("⚠️ Empty vocabulary — clean your data.")
    exit()

# Step 10: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 11: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 12: Predict and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Step 13: Predict for full data
df["PredictedPriority"] = model.predict(X_vectorized)

# Step 14: Save file
df.to_csv("model_predictions.csv", index=False)
print("✅ model_predictions.csv created successfully.")