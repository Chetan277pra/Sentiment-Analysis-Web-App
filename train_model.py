# train_model.py (Final Version - Tailored for News Headline Data)
# This script is now specifically built to read your unique dataset.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

print("--- Step 1: Loading the news headline dataset... ---")

try:
    # Read the CSV file with the correct encoding
    df = pd.read_csv('dataset.csv', encoding='latin1')
    print("Success! 'dataset.csv' was found and loaded.")
except Exception as e:
    print(f"!!! An unexpected error occurred while reading the file: {e} !!!")
    exit()

# --- NEW: Process the News Headline Data ---
# This is the most important part. We combine all 'Top...' columns into one.
print("\n--- Step 2: Combining news headlines into a single text block... ---")

# Replace any non-string values (like missing headlines) with an empty space
for col in df.columns:
    if 'Top' in col:
        df[col] = df[col].fillna('').astype(str)

# Combine all headline columns ('Top1' through 'Top25') into one column called 'text'
df['text'] = df.apply(lambda row: ' '.join(row[f'Top{i}'] for i in range(1, 26)), axis=1)
print("Successfully combined headlines.")

# Use the 'Label' column for sentiment
# We will rename it to 'sentiment' for consistency.
if 'Label' not in df.columns:
    print("\n!!! ERROR: The required 'Label' column was not found. !!!")
    exit()

df['sentiment'] = df['Label'].apply(lambda x: 'positive' if x == 1 else 'negative')
print("Successfully processed the 'Label' column for sentiment.")
# --- End of new data processing section ---

# We only need the 'text' and 'sentiment' columns now
df_final = df[['text', 'sentiment']]

# Clean up any rows that might have ended up with empty text
df_final.dropna(subset=['text', 'sentiment'], inplace=True)

X = df_final['text']
y = df_final['sentiment']

print(f"\n--- Step 3: Training the model on {len(df_final)} days of news... ---")

# This part converts your text into numbers so the model can understand it
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# This is the actual machine learning model. We train it here.
classifier = MultinomialNB()
classifier.fit(X_vectorized, y)
print("Success! The model has been trained.")

print("\n--- Step 4: Saving the 'brain' files... ---")

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Saved 'vectorizer.pkl'")

# Save the classifier
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
print("Saved 'sentiment_model.pkl'")

print("\nAll done! Your project is now ready to run.")
