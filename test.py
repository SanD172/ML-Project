import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Read in data
df = pd.read_csv('Reviews.csv')  # Replace 'path_to_your_data.csv' with the actual path
sia = SentimentIntensityAnalyzer()
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Set style
plt.style.use('ggplot')

# Main function
def main():
    st.title('Sentiment Analysis')
    quick_eda()
    basic_nltk()
    vader_sentiment()
    roberta_model()
    user_input()

# Quick EDA
def quick_eda():
    ax = df['Score'].value_counts().sort_index() \
        .plot(kind='bar',
              title='Count of Reviews by Stars',
              figsize=(10, 5))
    ax.set_xlabel('Review Stars')
    st.pyplot()

# Basic NLTK
def basic_nltk():
    example = df['Text'][50]
    st.write(example)
    tokens = nltk.word_tokenize(example)
    st.write(tokens[:10])
    tagged = nltk.pos_tag(tokens)
    st.write(tagged[:10])

# VADER Sentiment Scoring
def vader_sentiment():
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text']
        myid = row['Id']
        res[myid] = sia.polarity_scores(text)
    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'Id'})
    vaders = vaders.merge(df, how='left')
    
    # Plot VADER results
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
    sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
    sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
    axs[0].set_title('Positive')
    axs[1].set_title('Neutral')
    axs[2].set_title('Negative')
    plt.tight_layout()
    st.pyplot()

# Roberta Pretrained Model
def roberta_model():
    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg' : scores[0],
            'roberta_neu' : scores[1],
            'roberta_pos' : scores[2]
        }
        return scores_dict
    
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['Text']
            myid = row['Id']
            vader_result = sia.polarity_scores(text)
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
            roberta_result = polarity_scores_roberta(text)
            both = {**vader_result_rename, **roberta_result}
            res[myid] = both
        except RuntimeError:
            print(f'Broke for id {myid}')
    
    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={'index': 'Id'})
    results_df = results_df.merge(df, how='left')
    
    # Compare Scores between models
    sns.pairplot(data=results_df,
                 vars=['vader_neg', 'vader_neu', 'vader_pos',
                      'roberta_neg', 'roberta_neu', 'roberta_pos'],
                hue='Score',
                palette='tab10')
    plt.tight_layout()
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

if __name__ == "__main__":
    main()