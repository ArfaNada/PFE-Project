#!/usr/bin/env python
"""
Data Preprocessing Script

This script loads multiple CSV files, combines them, and performs text preprocessing
including URL removal, emoji transformation, punctuation removal, contraction expansion,
stemming, and stopword removal.
"""

import os
import re
import string
import pandas as pd
import numpy as np
import nltk
import demoji
import emoji
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Download required NLTK resources
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

# Initialize NLP tools
stopwords = set(nltk.corpus.stopwords.words('english'))
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()


def load_multiple_datasets(dataset_dir):
    """
    Load and concatenate multiple CSV files from a directory.
    
    Args:
        dataset_dir: Directory containing the CSV files
        
    Returns:
        Combined DataFrame
    """
    file_patterns = [
        "overwhelmed", "positivity", "selfharm", "sleeping too much",
        "stress", "suicide", "trauma", "worthless",
        "#conversationstarters", "#dailythoughts", "#mentalhealthrecovery",
        "#mentalhealthsupport", "#thoughtoftheday", "#tweetchat",
        "anger", "antidepressants", "anxious", "depressed", "depression",
        "dysthymia", "emotion", "feel", "feeling down", "feeling hopeless",
        "frustration", "grief", "heart break", "helpless", "isolated",
        "lackofenergy", "lonely", "loss of interest", "mental health",
        "mental illness", "mood swing", "negativity", "opinion"
    ]
    
    dataframes = []
    for pattern in file_patterns:
        filepath = os.path.join(dataset_dir, f"queries_for_{pattern}.csv")
        if os.path.exists(filepath):
            dataframes.append(pd.read_csv(filepath))
    
    return pd.concat(dataframes, ignore_index=True)


def url_limited(text):
    """Remove URLs from text."""
    return re.sub(r'http\S+', '', text)


def punct_word(text):
    """Remove punctuation and convert to lowercase."""
    return "".join([char.lower() for char in text if char not in string.punctuation])


def expand_contractions(text, contractions_dict=None):
    """
    Expand English contractions to their full form.
    
    Args:
        text: Input text with contractions
        contractions_dict: Dictionary of contractions and their expansions
        
    Returns:
        Text with expanded contractions
    """
    if contractions_dict is None:
        contractions_dict = {
            "ain't": "are not", "'s": " is", "aren't": "are not",
            "can't": "cannot", "can't've": "cannot have",
            "'cause": "because", "could've": "could have", "couldn't": "could not",
            "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would",
            "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
            "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
            "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
            "i'll've": "i will have", "i'm": "i am", "I've": "i have", "isn't": "is not",
            "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
            "it'll've": "it will have", "let's": "let us", "ma'am": "madam",
            "mayn't": "may not", "might've": "might have", "mightn't": "might not",
            "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
            "mustn't've": "must not have", "needn't": "need not",
            "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
            "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
            "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
            "she'll": "she will", "she'll've": "she will have", "should've": "should have",
            "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
            "that'd": "that would", "that'd've": "that would have", "there'd": "there would",
            "there'd've": "there would have", "they'd": "they would",
            "they'd've": "they would have", "they'll": "they will",
            "they'll've": "they will have", "they're": "they are", "they've": "they have",
            "to've": "to have", "wasn't": "was not", "we'd": "we would",
            "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
            "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
            "what'll've": "what will have", "what're": "what are", "what've": "what have",
            "when've": "when have", "where'd": "where did", "where've": "where have",
            "who'll": "who will", "who'll've": "who will have", "who've": "who have",
            "why've": "why have", "will've": "will have", "won't": "will not",
            "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
            "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
            "y'all'd've": "you all would have", "y'all're": "you all are",
            "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you'll've": "you will have", "you're": "you are",
            "you've": "you have"
        }
    
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    
    def replace(match):
        return contractions_dict[match.group(0)]
    
    return contractions_re.sub(replace, text)


def stem_sentence(sentence):
    """
    Tokenize and stem a sentence.
    
    Args:
        sentence: Input sentence
        
    Returns:
        Stemmed sentence
    """
    tokens = tokenizer.tokenize(sentence.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def remove_stopwords(sentence):
    """
    Remove stopwords from a sentence.
    
    Args:
        sentence: Input sentence
        
    Returns:
        Sentence without stopwords
    """
    tokens = tokenizer.tokenize(sentence)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return " ".join(filtered_tokens)


def preprocess_data(input_path, output_path):
    """
    Complete preprocessing pipeline for tweet data.
    
    Args:
        input_path: Path to input CSV or directory containing CSV files
        output_path: Path to save preprocessed CSV
    """
    # Load data
    if os.path.isdir(input_path):
        data = load_multiple_datasets(input_path)
    else:
        data = pd.read_csv(input_path)
    
    print(f"Initial data shape: {data.shape}")
    
    # Remove missing values
    data.dropna(inplace=True)
    print(f"After removing NaN: {data.shape}")
    
    # Keep only English tweets
    data = data[data['Language'] == 'en']
    print(f"After filtering English only: {data.shape}")
    
    # Remove unnecessary columns
    data = data.drop(['Tweet ID', 'Author ID', 'Created At', 'Language'], axis=1)
    
    # Process Tweet and Author Bio columns
    for column in ['Tweet', 'Author Bio']:
        print(f"Processing {column}...")
        
        # Remove URLs
        data[column] = data[column].apply(url_limited)
        
        # Transform emojis
        data[column] = data[column].astype(str).apply(
            lambda x: emoji.demojize(x, delimiters=(' ', ' '))
        )
        
        # Remove punctuation
        data[column] = data[column].apply(punct_word)
        
        # Expand contractions
        data[column] = data[column].apply(expand_contractions)
        
        # Apply stemming
        data[column] = data[column].apply(stem_sentence)
        
        # Remove stopwords
        data[column] = data[column].apply(remove_stopwords)
    
    # Save preprocessed data
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Final data shape: {data.shape}")
    
    return data


if __name__ == "__main__":
    # Example usage - update paths as needed
    input_path = "path/to/dataset/directory"  # or path to single CSV
    output_path = "preprocessed_data_with_stemming.csv"
    
    # Uncomment to run
    # preprocess_data(input_path, output_path)
if __name__ == "__main__":
    # Example usage - update paths as needed
    input_path = "path/to/dataset/directory"  # or path to single CSV
    output_path = "preprocessed_data_with_stemming.csv"
    
    # Uncomment to run
    # preprocess_data(input_path, output_path)




