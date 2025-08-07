import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load cleaned task data (from preprocessed file)
df = pd.read_csv("preprocessed_tasks.csv")

# ✅ Check and drop rows with missing cleaned descriptions
df = df.dropna(subset=["cleaned_description"])

# ✅ Load trained model and vectorizer
model = joblib.load("priority_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ✅ Transform cleaned descriptions using loaded vectorizer
X = vectorizer.transform(df["cleaned_description"])

# ✅ Predict priority
predictions = model.predict(X)

# ✅ Add predictions to the dataframe
df["predicted_priority"] = predictions

# ✅ Save the updated file
df.to_csv("model_predictions.csv", index=False)

print("✅ Task priority predictions done and saved to 'model_predictions.csv'")