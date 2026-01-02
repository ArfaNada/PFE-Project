#!/usr/bin/env python
"""
Sentiment Analysis Script

This script performs sentiment analysis on preprocessed tweet data using
TextBlob, VADER, and NRCLex sentiment analysis tools.
"""

import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex

# Download VADER lexicon
nltk.download('vader_lexicon')


def get_sentiment_textblob(text):
    """
    Get sentiment using TextBlob.
    
    Args:
        text: Input text
        
    Returns:
        'Positive', 'Negative', or 'Neutral'
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


def get_sentiment_vader(text):
    """
    Get sentiment using VADER.
    
    Args:
        text: Input text
        
    Returns:
        'Positive', 'Negative', or 'Neutral'
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    
    if sentiment_scores['compound'] > 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def encode_sentiment(sentiment):
    """
    Encode sentiment labels as numeric values.
    
    Args:
        sentiment: 'Positive', 'Negative', or 'Neutral'
        
    Returns:
        1 for Positive, -1 for Negative, 0 for Neutral
    """
    mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    return mapping.get(sentiment, 0)


def analyze_emotions_nrclex(text):
    """
    Analyze emotions using NRCLex.
    
    Args:
        text: Input text
        
    Returns:
        DataFrame with emotion scores
    """
    text_object = NRCLex(text)
    sentiment_scores = pd.DataFrame(
        list(text_object.raw_emotion_scores.items()),
        columns=['Sentiment', 'Count']
    )
    return sentiment_scores


def get_emotion_words(text):
    """
    Get words associated with each emotion.
    
    Args:
        text: Input text
        
    Returns:
        DataFrame mapping words to emotions
    """
    text_object = NRCLex(text)
    
    # Get sentiment scores
    sentiment_scores = pd.DataFrame(
        list(text_object.raw_emotion_scores.items()),
        columns=['Sentiment', 'Count']
    )
    
    # Create DataFrame of words and their sentiments
    sentiment_words = pd.DataFrame(
        list(text_object.affect_dict.items()),
        columns=['words', 'sentiments']
    )
    
    # Create columns for each sentiment
    sentiments = sentiment_scores['Sentiment'].tolist()
    for sentiment in sentiments:
        sentiment_words[sentiment] = 0
    
    # Mark which sentiments apply to each word
    for idx, word_sentiments in enumerate(sentiment_words['sentiments']):
        for sentiment in sentiments:
            sentiment_words.loc[idx, sentiment] = int(sentiment in word_sentiments)
    
    return sentiment_words


def perform_sentiment_analysis(input_path, output_path):
    """
    Complete sentiment analysis pipeline.
    
    Args:
        input_path: Path to preprocessed CSV file
        output_path: Path to save labeled CSV file
    """
    # Load preprocessed data
    data = pd.read_csv(input_path)
    
    print(f"Data shape: {data.shape}")
    
    # Ensure text columns are strings
    data["Author Bio"] = data["Author Bio"].astype(str)
    data["Tweet"] = data["Tweet"].astype(str)
    
    # Apply TextBlob sentiment analysis
    print("Applying TextBlob sentiment analysis...")
    data["Bio_Sentiment"] = data["Author Bio"].apply(get_sentiment_textblob)
    data["Tweet_Sentiment"] = data["Tweet"].apply(get_sentiment_textblob)
    
    # Encode sentiments as numeric values
    data['Bio_Sentiment'] = data['Bio_Sentiment'].apply(encode_sentiment)
    data['Tweet_Sentiment'] = data['Tweet_Sentiment'].apply(encode_sentiment)
    
    # Analyze emotions for tweets
    print("Analyzing emotions with NRCLex...")
    tweet_text = " ".join(data['Tweet'])
    tweet_emotions = analyze_emotions_nrclex(tweet_text)
    print("\nTweet Emotion Scores:")
    print(tweet_emotions)
    
    # Analyze emotions for author bios
    bio_text = " ".join(data['Author Bio'])
    bio_emotions = analyze_emotions_nrclex(bio_text)
    print("\nBio Emotion Scores:")
    print(bio_emotions)
    
    # Get emotion words
    combined_text = " ".join(data['Tweet'].fillna('') + ' ' + data['Author Bio'].fillna(''))
    emotion_words = get_emotion_words(combined_text)
    
    # Display top words for each emotion
    print("\nTop words for each emotion:")
    for sentiment in tweet_emotions['Sentiment']:
        word_list = emotion_words.loc[emotion_words[sentiment] == 1, 'words'].head(10)
        print(f"\n{sentiment}: {word_list.values}")
    
    # Save labeled data
    data.to_csv(output_path, index=False)
    print(f"\nLabeled data saved to {output_path}")
    print(f"Final data shape: {data.shape}")
    
    return data


if __name__ == "__main__":
    # Example usage - update paths as needed
    input_path = "preprocessed_data_with_stemming.csv"
    output_path = "new_data_labeled_stemmed.csv"
    
    # Uncomment to run
    # perform_sentiment_analysis(input_path, output_path)




