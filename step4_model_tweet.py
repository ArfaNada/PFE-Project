#!/usr/bin/env python
"""
Tweet Sentiment Classification Model

This script trains a Decision Tree classifier on tweet text to predict sentiment
and saves the model for later use.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_tweet_model(data_path, output_dir="."):
    """
    Train a sentiment classification model on tweet text.
    
    Args:
        data_path: Path to labeled dataset CSV
        output_dir: Directory to save the trained model and vectorizer
        
    Returns:
        Trained classifier and vectorizer
    """
    # Load data
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape}")
    
    # Create TF-IDF features from tweets
    tfidf_v = TfidfVectorizer()
    X = tfidf_v.fit_transform(data['Tweet'].values.astype('U'))
    y = data['Tweet_Sentiment']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=40
    )
    
    # Train Decision Tree classifier
    print("Training Decision Tree classifier...")
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dtc.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy:.4f}")
    
    # Save the vectorizer
    vectorizer_path = os.path.join(output_dir, 'tfidf_v_final_tweet.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_v, f)
    print(f"Vectorizer saved to {vectorizer_path}")
    
    # Save the model
    model_path = os.path.join(output_dir, 'decision_tree_model_tweet.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(dtc, f)
    print(f"Model saved to {model_path}")
    
    return dtc, tfidf_v


def load_and_predict(text, vectorizer_path='tfidf_v_final_tweet.pkl', 
                     model_path='decision_tree_model_tweet.pkl'):
    """
    Load trained model and make predictions on new text.
    
    Args:
        text: Text to classify
        vectorizer_path: Path to saved vectorizer
        model_path: Path to saved model
        
    Returns:
        Predicted sentiment
    """
    # Load the vectorizer
    with open(vectorizer_path, 'rb') as f:
        tfidf_v = pickle.load(f)
    
    # Load the model
    with open(model_path, 'rb') as f:
        dtc = pickle.load(f)
    
    # Create DataFrame with text
    data_new = pd.DataFrame([[text]], columns=['text'])
    
    # Transform and predict
    tfidf_pred = tfidf_v.transform(data_new["text"])
    y_pred = dtc.predict(tfidf_pred)
    
    # Map prediction to label
    sentiment_map = {1: 'Positive', -1: 'Negative', 0: 'Neutral'}
    return sentiment_map.get(y_pred[0], 'Unknown')


if __name__ == "__main__":
    # Example usage - update paths as needed
    data_path = "new_data_labeled_stemmed.csv"
    output_dir = "."
    
    # Train model (uncomment to run)
    # dtc, tfidf_v = train_tweet_model(data_path, output_dir)
    
    # Example prediction
    example_text = "i have demons in my head"
    # prediction = load_and_predict(example_text)
    # print(f"Text: {example_text}")
    # print(f"Predicted sentiment: {prediction}")




