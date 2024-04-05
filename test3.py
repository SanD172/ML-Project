import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

plt.style.use('ggplot')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Define aspect extraction function (simulated)
def extract_aspects(text):
    # Simulated aspect extraction, you may replace this with an actual aspect extraction model
    aspects = ['taste', 'service', 'price']  # Simulated aspects
    return aspects

# VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to perform aspect-based sentiment analysis
def aspect_based_sentiment_analysis(text):
    aspects = extract_aspects(text)
    aspect_sentiments = {}
    for aspect in aspects:
        aspect_text = text  # You may replace this with actual aspect-specific text
        sentiment_scores = sia.polarity_scores(aspect_text)
        aspect_sentiments[aspect] = sentiment_scores
    return aspect_sentiments

# Streamlit app
def main():
    st.title('Aspect-Based Sentiment Analysis')

    # Read in data
    df = pd.read_csv('Reviews.csv')
    st.subheader('Sample Data:')
    st.write(df.head())

    # Quick EDA
    st.subheader('Count of Reviews by Stars:')
    fig, ax = plt.subplots()
    df['Score'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel('Review Stars')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Run aspect-based sentiment analysis for each review
    res = {}
    for i, row in df.iterrows():
        text = row['Text']
        myid = row['Id']
        aspect_sentiments = aspect_based_sentiment_analysis(text)
        res[myid] = aspect_sentiments

    # Convert results to DataFrame
    aspect_sentiments_df = pd.DataFrame(res).T
    aspect_sentiments_df = aspect_sentiments_df.reset_index().rename(columns={'index': 'Id'})
    aspect_sentiments_df = aspect_sentiments_df.merge(df, how='left')

    # Display aspect-based sentiment analysis results
    st.subheader('Aspect-Based Sentiment Analysis Results:')
    st.write(aspect_sentiments_df.head())

if __name__ == '__main__':
    main()
