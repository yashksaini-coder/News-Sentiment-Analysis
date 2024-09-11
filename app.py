import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained vectorizer and label encoder
@st.cache_resource
def load_vectorizer():
    return pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@st.cache_resource
def load_label_encoder():
    return pickle.load(open('label_encoder.pkl', 'rb'))

@st.cache_resource
def load_model():
    return pickle.load(open('sentiment_analysis_model.h5', 'rb'))

# Define Streamlit app
def main():
    st.title("News Sentiment Analysis")
    
    st.write("Enter news text below to predict its sentiment:")
    text_input = st.text_area("News Text", height=150)
    
    # Load model, vectorizer, and label encoder
    model = load_model()
    vectorizer = load_vectorizer()
    label_encoder = load_label_encoder()

    if st.button("Predict Sentiment"):
        if text_input:
            # Transform the input text
            text_input_tfidf = vectorizer.transform([text_input])
            
            # Predict sentiment
            prediction = model.predict(text_input_tfidf)
            predicted_sentiment = label_encoder.inverse_transform(prediction)[0]
            
            # Display result for different sentiment cases
            if predicted_sentiment == "positive":
                st.success(f"Predicted sentiment: {predicted_sentiment}. The news has an optimistic and favorable tone.")
            elif predicted_sentiment == "neutral":
                st.error(f"Predicted sentiment: {predicted_sentiment}. The news contains negative or unfavorable information.")
        else:
            st.warning("Please enter some news text to analyze.")

# Run the app
if __name__ == "__main__":
    main()
