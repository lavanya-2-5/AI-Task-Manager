import pandas as pd
import random

# Step 1: Read preprocessed file
df = pd.read_csv("preprocessed_tasks.csv")

# Step 2: Create synthetic priority labels
def assign_priority(desc):
    keywords = {
        'high': ['urgent', 'immediate', 'asap', 'critical'],
        'medium': ['soon', 'moderate', 'normal'],
        'low': ['whenever', 'later', 'low']
    }

    for word in keywords['high']:
        if word in desc:
            return 'High'
    for word in keywords['medium']:
        if word in desc:
            return 'Medium'
    for word in keywords['low']:
        if word in desc:
            return 'Low'
    
    # Random fallback if no keyword matched
    return random.choice(['High', 'Medium', 'Low'])

# Apply the function
df['Priority'] = df['cleaned_description'].apply(assign_priority)

# Step 3: Save to new CSV
df.to_csv("labeled_tasks.csv", index=False)

print("✅ labeled_tasks.csv generated successfully!")