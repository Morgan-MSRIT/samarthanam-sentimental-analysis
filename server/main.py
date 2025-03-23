from flask import Flask, request, jsonify
import os
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

app = Flask(__name__)
app.debug = True

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.strip().lower()

def analyze_sentiment(text):
    cleaned_text = clean_text(text)

    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(cleaned_text)['compound']

    textblob_score = TextBlob(cleaned_text).sentiment.polarity

    combined_score = (vader_score + textblob_score) / 2

    sentiment_score = int((combined_score + 1) * 50)
    return sentiment_score

def classify_sentiment(score):
    if score <= 40:
        return "Negative"
    elif 41 <= score < 60:
        return "Neutral"
    elif score >= 60:
        return "Positive"


@app.route("/sentiment-analysis", methods=['POST'])
def analyze_feedback():
    data = request.json.get('data', [])

    # print("Data: ", data)
    
    if not data:
        return jsonify({"error": "No feedback data provided"}), 400

    analyzed_feedback = []
    for feedback in data:
        text = feedback.get('additionalInfo', '')
        sentiment_score = analyze_sentiment(text)
        sentiment = classify_sentiment(sentiment_score)
        
        feedback_review = {
            "sentiment_score": sentiment_score,
            "sentiment": sentiment
        }
        analyzed_feedback.append(feedback_review)

    return jsonify(analyzed_feedback)


if __name__ == '__main__':
    app.run(port=5000)