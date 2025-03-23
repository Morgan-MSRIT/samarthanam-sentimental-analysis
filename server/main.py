from flask import Flask, request, jsonify
import os
import sys
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from google import genai

app = Flask(__name__)
app.debug = True
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())




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

positive_data = []
neutral_data = []
negative_data = []

def classify_sentiment(score, text):
    if score <= 40:
        negative_data.append(text)
        return "Negative"

    elif 41 <= score < 60:
        neutral_data.append(text)
        return "Neutral"
    elif score >= 60:
        positive_data.append(text)
        return "Positive"



def sample_feedback(positive_data, neutral_data, negative_data):
    positive_ratio = 5
    neutral_ratio = 3
    negative_ratio = 2

    num_positive = int(positive_ratio * 10)
    num_neutral = int(neutral_ratio * 10)
    num_negative = int(negative_ratio * 10)

    sampled_positive = random.sample(positive_data, min(num_positive, len(positive_data)))
    sampled_neutral = random.sample(neutral_data, min(num_neutral, len(neutral_data)))
    sampled_negative = random.sample(negative_data, min(num_negative, len(negative_data)))

    sampled_feedback = sampled_positive + sampled_neutral + sampled_negative

    return sampled_feedback, sampled_positive, sampled_neutral, sampled_negative




def get_summary_from_gemini(sampled_feedback, sampled_positive, sampled_neutral, sampled_negative):

    prompt = "Summarize the following event feedback for 'Samarthanam Foundation Events' into a brief overall assessment. Highlight key insights without categorizing them. \n"

    prompt += "Postive Feedback: \n"
    for fd in sampled_positive:
        prompt += fd + "\n"

    prompt += "Neutral Feedback: \n"
    for fd in sampled_neutral:
        prompt += fd + "\n"
    
    prompt += "Negative Feedback: \n"
    for fd in sampled_negative:
        prompt += fd + "\n"

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    print("\n\nResponse: ", response)
    return response

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
        sentiment = classify_sentiment(sentiment_score, text)
        
        feedback_review = {
            "sentiment_score": sentiment_score,
            "sentiment": sentiment
        }
        analyzed_feedback.append(feedback_review)
    # print("\n\nPositive Data: ", positive_data)
    # print("\n\nNeutral Data: ", neutral_data)
    # print("\n\nNegative Data: ", negative_data)
    sampled_feedback, sampled_positive, sampled_neutral, sampled_negative =  sample_feedback(positive_data, neutral_data, negative_data)
    feedback_summary = get_summary_from_gemini(sampled_feedback, sampled_positive, sampled_neutral, sampled_negative)
    # print("\n\nFeedback Summary: ", feedback_summary)
    return jsonify(analyzed_feedback)


if __name__ == '__main__':
    app.run(port=5000)