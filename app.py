# app.py (Final Version - Tuned for better Neutral detection)
# This version has adjusted thresholds for more accurate real-world results.

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
    scores = vader_analyzer.polarity_scores(text)
    compound_score = scores['compound']
    
    # --- THIS IS THE IMPORTANT CHANGE (THE TUNING) ---
    # We have widened the range for what is considered "Neutral".
    # Instead of (-0.05 to 0.05), we now use (-0.2 to 0.2).
    # This will correctly classify mildly negative/positive statements as Neutral.
    if compound_score >= 0.2:
        return {'label': 'Positive', 'emoji': '😊'}
    elif compound_score <= -0.2:
        return {'label': 'Negative', 'emoji': '😠'}
    else:
        return {'label': 'Neutral', 'emoji': '😐'}
    # ----------------------------------------------------


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
    # We run with debug=False and on host='0.0.0.0' for deployment
    app.run(debug=False, host='0.0.0.0', port=port)

