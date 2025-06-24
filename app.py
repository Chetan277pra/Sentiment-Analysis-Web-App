# app.py (Final Polished Version with a Custom Rule)
# This version adds a specific fix for the "not love nor hate" edge case.

from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# --- Download the VADER data (only needs to be done once) ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER sentiment lexicon (one-time setup)...")
    nltk.download('vader_lexicon')


# --- Initialize the Flask Application ---
app = Flask(__name__)

# --- Create an instance of the VADER analyzer ---
vader_analyzer = SentimentIntensityAnalyzer()


# --- This function uses the VADER brain to make a prediction ---
def analyze_sentiment(text):
    
    # --- THIS IS THE NEW CUSTOM RULE ---
    # We check for the specific tricky phrase before using the model.
    # .lower() makes the check case-insensitive.
    if "not love" in text.lower() and "nor hate" in text.lower():
        # If we find this phrase, we override the model and return Neutral.
        return {'label': 'Neutral', 'emoji': '😐'}
    # -----------------------------------


    # If the custom rule doesn't apply, we proceed with the VADER model as before.
    scores = vader_analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    # We use our tuned thresholds for best results.
    if compound_score >= 0.2:
        return {'label': 'Positive', 'emoji': '😊'}
    elif compound_score <= -0.2:
        return {'label': 'Negative', 'emoji': '😠'}
    else:
        return {'label': 'Neutral', 'emoji': '😐'}


# --- This is the code for the website's main page (it stays the same) ---
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

