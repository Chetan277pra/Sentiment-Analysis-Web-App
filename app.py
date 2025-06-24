# app.py (Final News Analyzer Version)
# This version trains a model on your specific news dataset and runs the app.

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# --- Define filenames for our model "brain" ---
VECTORIZER_FILE = 'news_vectorizer.pkl'
MODEL_FILE = 'news_model.pkl'

# --- Function to Train and Save the Model ---
def train_and_save_model():
    """
    This function reads the news dataset, trains the model, and saves it.
    It will only run if the model files don't already exist.
    """
    print("--- Model not found. Starting training process... ---")
    
    try:
        df = pd.read_csv('dataset.csv', encoding='latin1')
        print("Success! 'dataset.csv' was found and loaded.")
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not read dataset.csv: {e} !!!")
        return False

    print("--- Combining news headlines... ---")
    for col in df.columns:
        if 'Top' in col:
            df[col] = df[col].fillna('').astype(str)
    
    df['text'] = df.apply(lambda row: ' '.join(row[f'Top{i}'] for i in range(1, 26)), axis=1)
    df['sentiment'] = df['Label'].apply(lambda x: 'positive' if x == 1 else 'negative')
    
    df_final = df[['text', 'sentiment']]
    df_final.dropna(subset=['text', 'sentiment'], inplace=True)

    X = df_final['text']
    y = df_final['sentiment']

    print(f"--- Training model on {len(df_final)} days of news... ---")
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    
    classifier = MultinomialNB()
    classifier.fit(X_vectorized, y)
    print("--- Training complete. ---")

    print("--- Saving model files... ---")
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to {MODEL_FILE} and {VECTORIZER_FILE}")
    
    return True


# --- Initialize the App ---
app = Flask(__name__)

# --- Load the Model (or Train if it doesn't exist) ---
if not (os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE)):
    if not train_and_save_model():
        # If training fails (e.g., dataset not found), exit.
        exit()

print("--- Loading trained model from files... ---")
with open(VECTORIZER_FILE, 'rb') as f:
    vectorizer = pickle.load(f)
with open(MODEL_FILE, 'rb') as f:
    classifier = pickle.load(f)
print("--- Model loaded successfully! ---")


# --- This function uses the trained news model to make a prediction ---
def analyze_sentiment(text):
    vectorized_text = vectorizer.transform([text])
    prediction = classifier.predict(vectorized_text)[0]

    if str(prediction).lower() == 'positive':
        return {'label': 'Positive (Market Up Forecast)', 'emoji': '📈'}
    else:
        return {'label': 'Negative (Market Down Forecast)', 'emoji': '📉'}


# --- This is the code for the website's main page ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form.get('text_input')
        if input_text:
            prediction_result = analyze_sentiment(input_text)
            
    return render_template('index.html', result=prediction_result, submitted_text=input_text)


# --- This is the deployment-ready code to start the server ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
