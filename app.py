from flask import Flask, render_template, request, jsonify
import joblib
import tensorflow as tf
from keras.models import load_model
import numpy as np
from utils import preprocess_for_fake
from utils import preprocess_for_sentiments

app = Flask(__name__)

# Load the trained models
fake_review_model = load_model('models/fake_review_model.h5')
sentiment_model = joblib.load('models/sentiment_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/review-input')
def review_input():
    return render_template('review_input.html')

@app.route('/analyze-review', methods=['POST'])
def analyze():
    user_review = request.form['review-input']

    # Preprocess the review for fake detection and sentiment analysis
    review_input_fake = preprocess_for_fake(user_review)  # TF-IDF vectorized
    review_input_sentiment = preprocess_for_sentiments(user_review)  # Tokenized and padded

    # Predictions
    is_fake = fake_review_model.predict(review_input_fake)[0]
    sentiment = sentiment_model.predict(review_input_sentiment)[0]
    fake_review_probability = fake_review_model.predict(review_input_fake)[0][0]  # Assuming a single input
    if fake_review_probability >= 0.5:
        review_authenticity = "Real"
        is_fake = False
    else:
        review_authenticity = "Fake"
        is_fake = True


    return render_template(
        'result.html',
        review=user_review,
        authenticity=review_authenticity,
        is_fake='Fake' if is_fake else 'Not Fake',
        sentiment=sentiment
    )
@app.route('/ping', methods=['GET'])
def ping():
    return {"status": "success", "message": "Flask is responsive!"}, 200


if __name__ == '__main__':
    app.run(debug=True)
