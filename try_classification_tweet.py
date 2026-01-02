#!/usr/bin/env python
"""
Tweet Sentiment Classification - Algorithm Comparison

This script compares multiple classification algorithms on tweet sentiment
analysis including Random Forest, Decision Tree, Logistic Regression, SVM, KNN,
Naive Bayes, and Passive Aggressive Classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def prepare_data(data_path):
    """
    Load and prepare data for classification.
    
    Args:
        data_path: Path to labeled dataset CSV
        
    Returns:
        X_train, X_test, y_train, y_test, tfidf_v
    """
    # Load data
    data = pd.read_csv(data_path)
    print(f"Initial data shape: {data.shape}")
    
    # Drop unnecessary columns
    data = data.drop(["Favorites", "Retweets", "Followers"], axis=1, errors='ignore')
    data = data.drop(columns=["Author Bio", "Bio_Sentiment"], axis=1, errors='ignore')
    
    # Create TF-IDF features
    tfidf_v = TfidfVectorizer()
    X = tfidf_v.fit_transform(data['Tweet'].values.astype('U'))
    y = data['Tweet_Sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, tfidf_v


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, cmap='Blues', fmt='g',
        xticklabels=['Neutral', 'Positive', 'Negative'],
        yticklabels=['Neutral', 'Positive', 'Negative']
    )
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return cm


def compare_classifiers(data_path):
    """
    Compare multiple classification algorithms.
    
    Args:
        data_path: Path to labeled dataset CSV
    """
    # Prepare data
    X_train, X_test, y_train, y_test, tfidf_v = prepare_data(data_path)
    
    results = {}
    
    # 1. Random Forest
    print("\n1. Training Random Forest...")
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred_rf = rfc.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {results['Random Forest']:.4f}")
    
    # 2. Decision Tree
    print("\n2. Training Decision Tree...")
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)
    y_pred_dt = dtc.predict(X_test)
    results['Decision Tree'] = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Accuracy: {results['Decision Tree']:.4f}")
    
    # Plot confusion matrix for Decision Tree
    print("\nConfusion Matrix:")
    cm = plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree - Confusion Matrix")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    target_names = ['Neutral', 'Positive', 'Negative']
    print(classification_report(y_test, y_pred_dt, target_names=target_names))
    
    # 3. Logistic Regression
    print("\n3. Training Logistic Regression...")
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {results['Logistic Regression']:.4f}")
    
    # 4. SVM
    print("\n4. Training SVM...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    results['SVM'] = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {results['SVM']:.4f}")
    
    # 5. K-Nearest Neighbors
    print("\n5. Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results['KNN'] = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Accuracy: {results['KNN']:.4f}")
    print(f"KNN Train Score: {knn.score(X_train, y_train):.4f}")
    print(f"KNN Test Score: {knn.score(X_test, y_test):.4f}")
    
    # 6. Naive Bayes
    print("\n6. Training Naive Bayes...")
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    nb = GaussianNB()
    nb.fit(X_train_dense, y_train)
    y_pred_nb = nb.predict(X_test_dense)
    results['Naive Bayes'] = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Accuracy: {results['Naive Bayes']:.4f}")
    
    # 7. Passive Aggressive Classifier
    print("\n7. Training Passive Aggressive Classifier...")
    pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    pac.fit(X_train, y_train)
    y_pred_pac = pac.predict(X_test)
    results['Passive Aggressive'] = accuracy_score(y_test, y_pred_pac)
    print(f"Passive Aggressive Accuracy: {results['Passive Aggressive']:.4f}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    for model, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:25s}: {accuracy:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage - update path as needed
    data_path = "new_data_labeled_stemmed.csv"
    
    # Uncomment to run
    # results = compare_classifiers(data_path)
if __name__ == "__main__":
    # Example usage - update path as needed
    data_path = "new_data_labeled_stemmed.csv"
    
    # Uncomment to run
    # results = compare_classifiers(data_path)




