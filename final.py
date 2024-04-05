import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Read in data
df = pd.read_csv('Reviews.csv')
df = df.head(500)

# Quick EDA
st.subheader('Count of Reviews by Stars')
ax = df['Score'].value_counts().sort_index().plot(kind='bar', figsize=(10, 5))
ax.set_xlabel('Review Stars')
st.pyplot()

# Basic NLTK
example = df['Text'][50]
st.write("Example Text:", example)
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
st.write("POS Tagging and Chunking Results:")
st.write(entities)

# Step 1. VADER Sentiment Scoring
st.subheader('VADER Sentiment Analysis')
sia = SentimentIntensityAnalyzer()
vader_results = []
for i, row in df.iterrows():
    text = row['Text']
    myid = row['Id']
    vader_result = sia.polarity_scores(text)
    vader_results.append({**{'Id': myid}, **vader_result})

vaders = pd.DataFrame(vader_results)
vaders = vaders.merge(df, how='left')

# Plot VADER results
st.subheader('VADER Compound Score by Amazon Star Review')
fig, ax = plt.subplots()
sns.barplot(data=vaders, x='Score', y='compound', ax=ax)
ax.set_title('Compund Score by Amazon Star Review')
st.pyplot(fig)

st.subheader('VADER Scores by Review Score')
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
st.pyplot(fig)

# Step 3. Roberta Pretrained Model
st.subheader('Roberta Pretrained Model')
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

roberta_results = []
max_length = 512  # Maximum sequence length
for i, row in df.iterrows():
    text = row['Text']
    myid = row['Id']
    # Tokenize and pad the input sequence
    encoded_text = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits.softmax(dim=1)
    scores_dict = {
        'roberta_neg': scores[0, 0].item(),
        'roberta_neu': scores[0, 1].item(),
        'roberta_pos': scores[0, 2].item()
    }
    roberta_results.append({**{'Id': myid}, **scores_dict})

roberta_df = pd.DataFrame(roberta_results)
roberta_df = roberta_df.merge(df, how='left')

# Compare Scores between models
st.subheader('Comparison of VADER and Roberta Scores')
fig, ax = plt.subplots(figsize=(10, 8))
sns.pairplot(data=pd.concat([vaders, roberta_df]),
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='tab10',
             diag_kind='kde',
             plot_kws={'alpha': 0.5})
st.pyplot(fig)

# Step 4: Review Examples
st.subheader('Review Examples')
st.write("Example of a low-score review with high Roberta positive sentiment:")
st.write(roberta_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0])
st.write("Example of a low-score review with high VADER positive sentiment:")
st.write(vaders.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0])
st.write("Example of a high-score review with high Roberta negative sentiment:")
st.write(roberta_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0])
st.write("Example of a high-score review with high VADER negative sentiment:")
st.write(vaders.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0])

# Extra: The Transformers Pipeline
st.subheader('The Transformers Pipeline')
sent_pipeline = pipeline("sentiment-analysis")
st.write(sent_pipeline('I love sentiment analysis!'))
st.write(sent_pipeline('booo'))
