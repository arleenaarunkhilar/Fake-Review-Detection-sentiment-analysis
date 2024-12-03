import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib

tokenizer = joblib.load('models/tokenizer.pkl')

tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def preprocess_for_fake(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=100)  # Adjust MAX_SEQUENCE_LENGTH accordingly
    return padded_sequence

def preprocess_for_sentiments(text):
     return tfidf_vectorizer.transform([text]) 
