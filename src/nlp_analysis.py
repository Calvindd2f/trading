import logging
from textblob import TextBlob
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text

def analyze_sentiment(text):
    """
    Perform sentiment analysis using TextBlob.
    Returns polarity and subjectivity.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def extract_keywords(text, n=10):
    """
    Extract keywords from text using spaCy.
    Returns the top n keywords.
    """
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    counter = Counter(keywords)
    return counter.most_common(n)

def process_text_data(df):
    """
    Process a DataFrame of text data for sentiment analysis and keyword extraction.
    """
    df['clean_text'] = df['text'].apply(clean_text)
    df['polarity'], df['subjectivity'] = zip(*df['clean_text'].apply(analyze_sentiment))
    df['keywords'] = df['clean_text'].apply(lambda x: extract_keywords(x))
    return df

def load_text_data(filepath):
    """Load text data from a CSV file."""
    return pd.read_csv(filepath)

def main():
    # Example usage
    filepath = 'data/text_data.csv'  # Update with your file path
    df = load_text_data(filepath)
    
    # Ensure there is a 'text' column
    if 'text' not in df.columns:
        logging.error("DataFrame must contain a 'text' column.")
        return
    
    processed_df = process_text_data(df)
    logging.info("Processed text data:")
    logging.info(processed_df.head())

    # Save the processed data to a new CSV file
    processed_df.to_csv('data/processed_text_data.csv', index=False)
    logging.info("Processed data saved to 'data/processed_text_data.csv'.")

if __name__ == "__main__":
    main()
