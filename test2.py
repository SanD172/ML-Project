import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Read in data
df = pd.read_csv('Reviews.csv')  # Replace 'path_to_your_data.csv' with the actual path
df = df.sample(frac=0.5, random_state=42)  # Use only half of the data
sia = SentimentIntensityAnalyzer()
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Set style
plt.style.use('ggplot')

# Main function
def main():
    st.title('Sentiment Analysis')
    quick_eda()
    user_input()

# Quick EDA
def quick_eda():
    ax = df['Score'].value_counts().sort_index() \
        .plot(kind='bar',
              title='Count of Reviews by Stars',
              figsize=(10, 5))
    ax.set_xlabel('Review Stars')
    st.pyplot()

# User Input
def user_input():
    sentence = st.text_input('Enter a sentence for sentiment analysis:')
    if sentence:
        vader_result = sia.polarity_scores(sentence)
        st.write("VADER Sentiment Analysis:")
        st.write(vader_result)

        roberta_result = polarity_scores_roberta(sentence)
        st.write("Roberta Model Sentiment Analysis:")
        st.write(roberta_result)

        # Comparison between VADER and Roberta
        vader_compound = vader_result['compound']
        roberta_pos = roberta_result['roberta_pos']
        roberta_neg = roberta_result['roberta_neg']

        if vader_compound >= 0:
            vader_sentiment = "Positive"
        else:
            vader_sentiment = "Negative"

        if roberta_pos > roberta_neg:
            roberta_sentiment = "Positive"
        else:
            roberta_sentiment = "Negative"

        st.write("Comparison:")
        st.write(f"VADER Sentiment: {vader_sentiment}")
        st.write(f"Roberta Sentiment: {roberta_sentiment}")

# Roberta Pretrained Model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

if __name__ == "__main__":
    main()
