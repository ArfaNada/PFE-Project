#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This is a Python script that uses the Tweepy library to collect tweets from Twitter and save them to a CSV file.

import tweepy as tw,pandas as pd,os


consumer_key = 'l3OhDWBtgxgoVh4w9dSzf3Pyh'
consumer_sec = '3BrH61g5QPTW4lKEN4MVY43bQ7hE6BkMkKzXwBz0e7LewuAi6a'
access_token = '1561392108933713921-vKMmoSfWQFghl8m9piEXa16krA4FEB'
access_token_sec ='KwUkTPMbUOqNYUVpqnzo2fDKrP6YxkRbEMaqOBcwJc95Q'
bearer_token = " AAAAAAAAAAAAAAAAAAAAAGMbmAEAAAAAU%2FMX9z9LrWuJbON4nOATqAY6RZM%3DOfwlqAHdDaKCE8YEGG6dkf1PJQ2M7x2xiI9158dYoMeIxLlDH3AAAAAAAAAAAAAAAAAAAAAGMbmAEAAAAAU%2FMX9z9LrWuJbON4nOATqAY6RZM%3DOfwlqAHdDaKCE8YEGG6dkf1PJQ2M7x2xiI9158dYoMeIxLlDH3"

#Authenticating the Client

auth = tw.OAuth1UserHandler(consumer_key,consumer_sec)
auth.set_access_token(access_token,access_token_sec)

#Creating the Tweepy Client

client = tw.Client(
    bearer_token= bearer_token,
    consumer_key = auth.consumer_key,
    consumer_secret = auth.consumer_secret,
    access_token = auth.access_token,
    access_token_secret = auth.access_token_secret,
    wait_on_rate_limit = True
)

#Creating an API object
api = tw.API(auth)

#Defining the all_to_csv Function
def all_to_csv(query:str,count:int):
    #Collecting Tweets and Creating DataFrame
    Reps = tw.Cursor(                   
        method = api.search_tweets,
        q = query,
        tweet_mode = "extended"
    ).items(count)

    #Defining a Helper Function
    get_data = lambda T : pd.DataFrame(
            data = {
                    "Tweet" : T["full_text"],
                    "Author Bio" : T["user"]["description"],
                    "Followers" : T["user"]["followers_count"],
                    "Retweets" : T["retweet_count"],
                    "Favorites" : T["favorite_count"],
                    "Tweet ID" : T["id_str"],
                    "Author ID" : T["user"]["id"],
                    "Created At" : T["created_at"],
                    "Language" : T["lang"]
            },
            index = [0]
    )
    
    #Creating an Empty DataFrame
    df = pd.DataFrame(
        index = [0]
    )
    #Collecting and Appending Data to DataFrame
    for x in Reps:
        df = pd.concat(
            [df,get_data(x._json)],
            ignore_index=False
        )
    #Cleaning and Saving the DataFrame
    df.reset_index(drop=True,inplace=True) #reset the index
    df.dropna(inplace=True) #drop any rows with missing values
    df = df.astype(dtype = {  # convert specific columns to their appropriate data types
        "Retweets" : "int64",
        "Followers" : "int64",
        "Favorites" : "int64",
        "Tweet ID" : "str",
        "Author ID" : "str",
        "Tweet" : "str"
    })
    #saves the DataFrame to a CSV file
    df.to_csv(f"queries_for_{query}.csv",index=False)
#Calling the Function
all_to_csv("dailychat",20)


# In[ ]:


'''
The key words:
#mentalhealthsupport, #mentalhealthrecovery, #dailythoughts, #conversationstarters, worthless, trauma, 
suicide, stress, sleeping too much, self-harm, positivity, overwhelmed, opinion, negativity, mood swing, 
mental illness, mental health, loss of interest, for_lonely, lack of energy, isolated, helpless, heartbreak, 
grief, frustration, feeling hopeless, feeling down, feel emotion, dysthymia, depression, depressed, anxious, 
antidepressants, anger, #tweetchat et #thoughtoftheday
'''


# In[ ]:


In essence, the code collects data from each tweet, converts it into a DataFrame using the get_data function, and appends 
the resulting DataFrame to an existing DataFrame df. This process is repeated for each tweet in the Reps
collection, effectively accumulating the data from all the tweets into the df DataFrame.

