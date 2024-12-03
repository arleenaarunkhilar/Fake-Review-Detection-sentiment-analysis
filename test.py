import joblib
import tensorflow as tf
from keras.models import load_model
from utils import preprocess_for_fake
from utils import preprocess_for_sentiments

fake_review_model = load_model('models/fake_review_model.h5')
sentiment_model = joblib.load('models/sentiment_model.pkl')
# Test fake review model
test_review = "This product is useless"
review_input_fake = preprocess_for_fake(test_review)
print(fake_review_model.predict(review_input_fake))

# Test sentiment analysis model
review_input_sentiment = preprocess_for_sentiments(test_review)
print(sentiment_model.predict(review_input_sentiment))
