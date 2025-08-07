import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Ensure NLTK resources are downloaded
nltk.download('stopwords')

# Load Trello task data
df = pd.read_csv('trello_tasks.csv')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
df['Cleaned_Description'] = df['Description'].apply(clean_text)

# Save to CSV
df.to_csv('preprocessed_tasks.csv', index=False)
print("✅ Preprocessed data saved to preprocessed_tasks.csv")