# DepressDetect — Early Depression Detection from Social Media

A machine learning pipeline for detecting early signs of depression from social media data (Twitter/X). The project uses sentiment analysis and multiple classification algorithms to analyze tweets and author bios for mental health indicators.

## Features

- **Data Collection**: Automated tweet collection using Twitter API
- **Text Preprocessing**: Comprehensive text cleaning (URL removal, emoji handling, stemming, stopword removal)
- **Sentiment Analysis**: Multi-method sentiment analysis using TextBlob, VADER, and NRCLex
- **Classification Models**: Comparison of 7+ ML algorithms (Decision Tree, Random Forest, SVM, KNN, etc.)
- **Dual Analysis**: Separate models for tweet content and author bios

## Repository Structure

- `step1_collecting_data.py`: Collect tweets and metadata from Twitter API
- `step2_preprocessing.py`: Data cleaning and preprocessing (tokenization, normalization, stemming)
- `step3_sentiment_analysis.py`: Sentiment and emotion analysis using multiple libraries
- `step4_model_tweet.py`: Train and save Decision Tree model on tweet text
- `step5_model_bio.py`: Train and save Decision Tree model on author bio
- `try_classification_bio.py`: Compare multiple classification algorithms on bio features
- `try_classification_tweet.py`: Compare multiple classification algorithms on tweet features
- `requirements.txt`: Python package dependencies

## Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('words'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('omw-1.4')"
```

### 4. Configure Twitter API Credentials

Create a `.env` file in the project root:

```bash
TWITTER_CONSUMER_KEY=your_consumer_key
TWITTER_CONSUMER_SECRET=your_consumer_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
TWITTER_BEARER_TOKEN=your_bearer_token
```

## Usage

Run the pipeline steps in order:

### Step 1: Data Collection

```bash
python step1_collecting_data.py
```

Update the script to collect tweets for your desired keywords.

### Step 2: Data Preprocessing

```bash
python step2_preprocessing.py
```

Update input/output paths in the script before running.

### Step 3: Sentiment Analysis

```bash
python step3_sentiment_analysis.py
```

Performs sentiment labeling using TextBlob and emotion analysis with NRCLex.

### Step 4 & 5: Train Models

```bash
python step4_model_tweet.py
python step5_model_bio.py
```

Trains and saves Decision Tree models for predictions.

### Experiment with Classifiers

```bash
python try_classification_tweet.py
python try_classification_bio.py
```

Compare performance of multiple algorithms (Random Forest, SVM, Logistic Regression, KNN, Naive Bayes, etc.)

## Mental Health Keywords

The project focuses on mental health-related keywords including:

- Depression indicators: depressed, depression, dysthymia, feeling hopeless
- Anxiety indicators: anxious, stress, overwhelmed
- Emotional states: grief, frustration, anger, isolation
- Support topics: #mentalhealthsupport, #mentalhealthrecovery

## Models & Algorithms

### Tested Algorithms

1. Decision Tree Classifier
2. Random Forest Classifier
3. Logistic Regression
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Passive Aggressive Classifier

### Sentiment Analysis Tools

- **TextBlob**: Polarity-based sentiment classification
- **VADER**: Social media-optimized sentiment analysis
- **NRCLex**: Emotion lexicon analysis (fear, anger, joy, sadness, etc.)

## Output Files

- `queries_for_*.csv`: Raw collected tweets
- `preprocessed_data_with_stemming.csv`: Cleaned and processed data
- `new_data_labeled_stemmed.csv`: Sentiment-labeled dataset
- `decision_tree_model_tweet.pkl`: Trained tweet classifier
- `decision_tree_model_bio.pkl`: Trained bio classifier
- `tfidf_v_final_tweet.pkl`: TF-IDF vectorizer for tweets
- `tfidf_v_final_bio.pkl`: TF-IDF vectorizer for bios

## Notes

- All scripts use configurable paths - update input/output paths before running
- Twitter API credentials are required for data collection
- The preprocessing pipeline includes: URL removal, emoji conversion, punctuation removal, contraction expansion, stemming, and stopword removal
- Models are saved using pickle for later use in production

## License

See [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) file for details.
