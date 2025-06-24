# app.py (Final Version using the VADER library)
# This is the most accurate version for social media and mixed sentiment.
import os
from flask import Flask, render_template, request
# We import the new, more powerful VADER "brain" from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- Download the VADER data (only needs to be done once) ---
try:
    # This checks if you already have the data, if not, it downloads it
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
    # 1. Get the polarity scores from VADER.
    # It returns a dictionary with positive, negative, neutral, and a compound score.
    scores = vader_analyzer.polarity_scores(text)
    
    # 2. The 'compound' score is the most useful one.
    # It's a single number from -1 (most negative) to +1 (most positive).
    compound_score = scores['compound']
    
    # 3. Determine the result based on the compound score.
    # These are the standard, recommended thresholds for VADER.
    if compound_score >= 0.05:
        return {'label': 'Positive', 'emoji': '😊'}
    elif compound_score <= -0.05:
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


# --- Start the web server ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


