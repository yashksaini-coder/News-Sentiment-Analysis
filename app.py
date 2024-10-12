import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the GROQ client with the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is not set. Please check your .env file.")
else:
    client = Groq(api_key=api_key)

# Function to perform sentiment analysis with a template prompt

# Function to perform sentiment analysis with a template prompt
def analyze_sentiment(text):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the sentiment of the following text and return the sentiment result in simple words. Indicate whether the sentiment is positive, neutral, or negative."
                },
                {
                    "role": "user",
                    "content": text,
                },
            ]
        )
                
        # Check if the response is successful
        if response and response.choices:
            return response.choices[0].message.content  # Return the sentiment analysis result
        else:
            st.error("No response from the API.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit application layout
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if user_input:
        result = analyze_sentiment(user_input)
        if result:
            # Determine sentiment and set color
            if "positive" in result.lower():
                st.markdown(f"<h3 style='color: green;'>Sentiment: {result}</h3>", unsafe_allow_html=True)
            elif "negative" in result.lower():
                st.markdown(f"<h3 style='color: red;'>Sentiment: {result}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: orange;'>Sentiment: {result}</h3>", unsafe_allow_html=True)  # Neutral case
    else:
        st.warning("Please enter some text to analyze.")
