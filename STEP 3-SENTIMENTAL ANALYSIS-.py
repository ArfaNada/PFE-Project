#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
data=pd.read_csv("C:/users/Nada/MyDataSet/preprocessed_data_with_stemming.csv")


# In[32]:


import numpy as np
import pandas as pd
from textblob import TextBlob


# In[33]:


#converting values to strings
data["Author Bio"] = data["Author Bio"].astype(str)


# In[34]:



def get_sentiment(text):
    vs = TextBlob(text).sentiment[0]
    if vs > 0:
        return 'Positive'
    elif vs < 0:
        return 'Negative'
    else:
        return 'Neutral'


# In[35]:


data["Bio_Sentiment"] = data["Author Bio"].apply(get_sentiment)
data["Tweet_Sentiment"] = data["Tweet"].apply(get_sentiment)


# In[6]:


data


# In[36]:


#sentiment_label_both
data["Sentiment_label_both"] = data["Tweet"].str.cat(data["Author Bio"], sep=' ').apply(get_sentiment)


# In[9]:


data.head()


# In[37]:


# transform labels to numeric values
data['Bio_Sentiment'] = data['Bio_Sentiment'].replace({'Positive': 1, 'Negative': -1, 'Neutral': 0})


# In[38]:


# transform labels to numeric values
data['Tweet_Sentiment'] = data['Tweet_Sentiment'].replace({'Positive': 1, 'Negative': -1, 'Neutral': 0})


# In[39]:


# transform labels to numeric values
data['Sentiment_label_both'] = data['Sentiment_label_both'].replace({'Positive': 1, 'Negative': -1, 'Neutral': 0})


# In[13]:


data.head()


# In[40]:


from nrclex import NRCLex
import nltk
text=" ".join(data['Tweet'])
text_object = NRCLex(text)

#extracting the raw emotion scores from the NRCLex object and creating a DataFrame from them. Each row represents a sentiment 
sentiment_scores = pd.DataFrame(list(text_object.raw_emotion_scores.items())) 
sentiment_scores = sentiment_scores.rename(columns={0: "Sentiment", 1: "Count"})
sentiment_scores


# In[41]:


text=" ".join(data['Author Bio'])
text_object = NRCLex(text)

sentiment_scores = pd.DataFrame(list(text_object.raw_emotion_scores.items()))
sentiment_scores = sentiment_scores.rename(columns={0: "Sentiment", 1: "Count"})
sentiment_scores


# In[43]:


# Concatenate Tweet and Author Bio columns and remove null values
data['text'] = data['Tweet'].fillna('') + ' ' + data['Author Bio'].fillna('')
text = " ".join(data['text'])

# Create NRCLex object
text_object = NRCLex(text)

# Get sentiment scores and create a DataFrame
sentiment_scores = pd.DataFrame(list(text_object.raw_emotion_scores.items()), columns=['Sentiment', 'Count'])


# In[19]:


print(sentiment_scores)


# In[27]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer #Importing the required modules
get_ipython().system(' pip install VADER')
nltk.download('vader_lexicon') #Downloading VADER resources

# Instantiate the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to apply the analyzer to each tweet in your dataset
def get_sentiment(tweet):
    sentiment_scores = analyzer.polarity_scores(tweet)
    if sentiment_scores['compound'] > 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to each tweet in your dataset
data['sentiment_vader'] = data['text'].apply(get_sentiment)

# View the distribution of sentiments in your dataset
data['sentiment_vader'].value_counts()


# In[44]:


#Corresponding words to their sentiments
sentiment_words = pd.DataFrame(list(text_object.affect_dict.items()),columns = ['words','sentiments'])
sentiment_words


# In[49]:


# Corresponding sentiments to their dictionary of words
sentiment = sentiment_scores['Sentiment'].to_list()
sentiment_words

for y in sentiment:
    sentiment_words[y] = 9
sentiment_words

# Updating the values in the sentiment_words DataFrame based on sentiment matches
for a, i in enumerate(sentiment_words['sentiments']):
    for y in sentiment:
        sentiment_words.loc[a, y] = int(y in i)
#.loc indexer is used to modify the DataFrame at the specified row and column.
sentiment_words.head(5)

for y in sentiment:
    word_list = sentiment_words.loc[sentiment_words[y] == 1, 'words'].head(10)
    print(f"Sentiment: {y}")
    print(f"{word_list.values}\n")


# In[23]:


#if the value for the 'positive' category is 5, that means that the text contains 5 words that are associated with
#positive sentiment

# Define a function to get the sentiment scores for a given text
def get_sentiment_scores(text):
    text_object = NRCLex(text)
    sentiment_scores = text_object.raw_emotion_scores
    return sentiment_scores

# Apply the function to both the "Tweet" and "Author Bio" columns
data["text_Sentiment_Scores"] = data["text"].apply(get_sentiment_scores)

# View the resulting dataframe with sentiment scores
print(data[["text_Sentiment_Scores"]])


# In[24]:


print(data.shape)


# In[25]:


data


# In[28]:


data= data.drop(['Sentiment_label_both','text','sentiment_vader','text_Sentiment_Scores'], axis=1)


# In[29]:


data


# In[28]:


# Save the labeled DataFrame as a new CSV file
data.to_csv("C:/Users/Nada/MyDataSet/new_data_labeled_stemmed.csv", index=False)


# In[ ]:




