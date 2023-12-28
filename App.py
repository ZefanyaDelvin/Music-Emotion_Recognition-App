import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import re
import string
import streamlit as st

# Load data
path = "SingleLabel.csv"
df = pd.read_csv(path)

# Preprocess text
str_punc = string.punctuation.replace(',', '').replace("'", '').replace('!', '').replace('?', '')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    return text

df['lyrics'] = df['lyrics'].apply(clean_text)
df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True, inplace=True)

# Data Balancing
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df['lyrics'].values.reshape(-1, 1), df['label'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.ravel())
X_test_tfidf = tfidf_vectorizer.transform(X_test.ravel())

# Initialize and train Support Vector Machine (SVM)
svm_classifier = SVC()  
svm_classifier.fit(X_train_tfidf, y_train)  

# Define the function for predicting emotion
def predict_emotion(lyrics):
    cleaned_lyrics = clean_text(lyrics)
    cleaned_lyrics = tfidf_vectorizer.transform([cleaned_lyrics])
    prediction = svm_classifier.predict(cleaned_lyrics)[0]
    return prediction

# Streamlit app
st.title('Emotion Prediction from Lyrics')

# Example input string for prediction
input_string = st.text_area("Input text")
st.write(input_string)

input_vector = tfidf_vectorizer.transform([input_string])

# Predict the class
# emotion_mapping = {0: 'Sadness', 1: 'Tension', 2: 'Tenderness'}
prediction = svm_classifier.predict(input_vector)[0]

st.title(prediction)
