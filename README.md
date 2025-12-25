# DepressDetect — Early Depression Detection from Social Media

A machine-learning pipeline for detecting early signs of depression from social media data (Twitter/X), with an integration point for Telegram.

## Repository Structure

- `STEP 1-COLLECTING DATA-.py`: collect tweets and metadata used for the project.
- `STEP 2 -PREPROCESSING-.py`: data cleaning and preprocessing (tokenization, normalization, etc.).
- `STEP 3-SENTIMENTAL ANALYSIS-.py`: sentiment and lexical analysis steps used as features.
- `STEP 4-MODEL TWEET.py`: train/evaluate model(s) using tweet text.
- `STEP 5-MODEL BIO-.py`: train/evaluate model(s) using author bio information.
- `TRY CLASSIFICATION ALGORITHMS  ON  'Author Bio'.py`: experiments with different classifiers on bio features.
- `TRY CLASSIFICATION ALGORITHMS  ON  'TWEET'.py`: experiments with different classifiers on tweet features.

## Quick Start

1. Create a Python environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies. If a `requirements.txt` is provided, run:

```bash
pip install -r requirements.txt
```

Otherwise install the common packages used in the project, for example:

```bash
pip install pandas scikit-learn nltk spacy matplotlib
```

3. Run the pipeline steps in order, or open individual scripts for experimentation:

```bash
python "STEP 1-COLLECTING DATA-.py"
python "STEP 2 -PREPROCESSING-.py"
python "STEP 3-SENTIMENTAL ANALYSIS-.py"
python "STEP 4-MODEL TWEET.py"
python "STEP 5-MODEL BIO-.py"
```

## Notes

- The scripts are written as standalone steps; inspect each file for required input/output paths and any API keys or credentials for data collection.
- Tests and a `requirements.txt` may not be present — review and pin dependencies before production use.
- The repo mentions Telegram integration; the integration code or instructions are located inside the relevant script(s).
