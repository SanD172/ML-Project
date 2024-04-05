import streamlit as st
import pandas as pd
from transformers import pipeline

# Read in data
@st.cache
def load_data():
    df = pd.read_csv('Reviews.csv')  # Replace 'path_to_your_data.csv' with the actual path
    df = df.sample(frac=0.5, random_state=42)  # Use only half of the data
    return df

# Main function
def main():
    st.title('Aspect-based Sentiment Analysis')
    df = load_data()

    target_aspect = st.text_input('Enter the target aspect (e.g., "service", "food", "ambience"):')
    if target_aspect:
        aspect_sentiment_analysis(df, target_aspect)

# Aspect-based Sentiment Analysis
@st.cache(allow_output_mutation=True)
def get_nlp_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def aspect_sentiment_analysis(df, target_aspect):
    st.subheader(f'Sentiment Analysis for the Aspect: {target_aspect}')

    # Filter reviews mentioning the target aspect
    target_reviews = df[df['Text'].str.contains(target_aspect, case=False)]

    if len(target_reviews) == 0:
        st.write(f"No reviews found mentioning the aspect: {target_aspect}")
        return

    st.write(f"Total Reviews mentioning '{target_aspect}': {len(target_reviews)}")

    nlp = get_nlp_pipeline()

    # Perform sentiment analysis on the filtered reviews
    for idx, review in target_reviews.iterrows():
        st.write(f"Review {idx + 1}:")
        st.write(review['Text'])

        # Sentiment analysis using Roberta Model
        roberta_result = nlp(review['Text'])[0]
        scores_dict = {
            'roberta_neg': roberta_result['score'] if roberta_result['label'] == 'LABEL_0' else 0.0,
            'roberta_neu': roberta_result['score'] if roberta_result['label'] == 'LABEL_1' else 0.0,
            'roberta_pos': roberta_result['score'] if roberta_result['label'] == 'LABEL_2' else 0.0
        }
        st.write("Roberta Model Sentiment Analysis:")
        st.write(scores_dict)

# Run the app
if __name__ == "__main__":
    main()
