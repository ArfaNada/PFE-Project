#!/usr/bin/env python
"""
Twitter Data Collection Script

This script uses the Tweepy library to collect tweets based on search queries
and save them to CSV files for analysis.
"""

import os
import tweepy as tw
import pandas as pd

# Twitter API credentials from environment variables
consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
consumer_sec = os.getenv('TWITTER_CONSUMER_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_sec = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

# Validate credentials
if not all([consumer_key, consumer_sec, access_token, access_token_sec, bearer_token]):
    raise ValueError(
        "Missing Twitter API credentials. Please set the following environment variables:\n"
        "TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, "
        "TWITTER_ACCESS_TOKEN_SECRET, TWITTER_BEARER_TOKEN"
    )

# Authenticate the client
auth = tw.OAuth1UserHandler(consumer_key, consumer_sec)
auth.set_access_token(access_token, access_token_sec)

# Create the Tweepy client
client = tw.Client(
    bearer_token=bearer_token,
    consumer_key=auth.consumer_key,
    consumer_secret=auth.consumer_secret,
    access_token=auth.access_token,
    access_token_secret=auth.access_token_secret,
    wait_on_rate_limit=True
)

# Create an API object
api = tw.API(auth)


def all_to_csv(query: str, count: int, output_dir: str = "."):
    """
    Collect tweets based on a search query and save to CSV.
    
    Args:
        query: Search query string
        count: Number of tweets to collect
        output_dir: Directory to save the CSV file
    """
    # Collect tweets using cursor
    tweets = tw.Cursor(
        method=api.search_tweets,
        q=query,
        tweet_mode="extended"
    ).items(count)

    # Helper function to extract data from tweet
    def get_data(tweet):
        return pd.DataFrame({
            "Tweet": tweet["full_text"],
            "Author Bio": tweet["user"]["description"],
            "Followers": tweet["user"]["followers_count"],
            "Retweets": tweet["retweet_count"],
            "Favorites": tweet["favorite_count"],
            "Tweet ID": tweet["id_str"],
            "Author ID": tweet["user"]["id"],
            "Created At": tweet["created_at"],
            "Language": tweet["lang"]
        }, index=[0])
    
    # Create empty DataFrame
    df = pd.DataFrame()
    
    # Collect and append data
    for tweet in tweets:
        df = pd.concat([df, get_data(tweet._json)], ignore_index=True)
    
    # Clean the DataFrame
    df.dropna(inplace=True)
    df = df.astype({
        "Retweets": "int64",
        "Followers": "int64",
        "Favorites": "int64",
        "Tweet ID": "str",
        "Author ID": "str",
        "Tweet": "str"
    })
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"queries_for_{query}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} tweets to {output_path}")


# Example usage
if __name__ == "__main__":
    # Mental health related keywords for data collection
    keywords = [
        "#mentalhealthsupport", "#mentalhealthrecovery", "#dailythoughts",
        "#conversationstarters", "worthless", "trauma", "suicide", "stress",
        "sleeping too much", "self-harm", "positivity", "overwhelmed",
        "opinion", "negativity", "mood swing", "mental illness", "mental health",
        "loss of interest", "for_lonely", "lack of energy", "isolated",
        "helpless", "heartbreak", "grief", "frustration", "feeling hopeless",
        "feeling down", "feel emotion", "dysthymia", "depression", "depressed",
        "anxious", "antidepressants", "anger", "#tweetchat", "#thoughtoftheday"
    ]
    
    # Example: collect tweets for a single query
    all_to_csv("dailychat", 20)

