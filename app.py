from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import streamlit as st

max_features=10000
max_len=500

#loading the imdb word index
@st.cache_data
def get_word_index():
    return imdb.get_word_index()

word_index = get_word_index()   
reversed_word_index = {value : key for key, value in word_index.items()}

#loading the trained model weights
model = load_model("simple_rnn.h5")

#function to decode the review
def decoded_review(encoded_review):
    return " ".join([reversed_word_index.get(i-3, "?") for i in encoded_review]) # Indices 0, 1, and 2 are reserved, so subtract 3 to map to correct words


#function to preprocess the user input
def preprocessed_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    padded_review = padded_review.clip(0, max_features - 1)
    return padded_review


#prediction function
def predict_sentiment(review):
    if not review.strip():
        return "Invalid input. Please enter a non-empty review.", None
    
    preprocessed_review = preprocessed_text(review)
    
    prediction = model.predict(preprocessed_review)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]



st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Write a movie review below:")

user_input = st.text_area("Movie review")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a review to classify.")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.success(f"The review is **{sentiment}** with confidence **{score:.2f}**")
        st.progress(int(score*100))