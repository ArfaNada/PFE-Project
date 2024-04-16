#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
my_data = pd .concat([

    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_overwhelmed.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_positivity.csv"),
    pd.read_csv("C:/Users/Nada\MyDataSet/queries_for_selfharm.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_sleeping too much.csv"),
    pd.read_csv("C:/Users/Nada\MyDataSet/queries_for_stress.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_suicide.csv"),
    pd.read_csv("C:/Users/Nada\MyDataSet/queries_for_trauma.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_worthless.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_#conversationstarters.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_#dailythoughts.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_#mentalhealthrecovery.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_#mentalhealthsupport.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_#thoughtoftheday.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_#tweetchat.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_anger.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_antidepressants.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_anxious.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_depressed.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_depression.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_dysthymia.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_emotion.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_feel.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_feeling down.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_feeling hopeless.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_frustration.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_grief.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_heart break.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_helpless.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_isolated.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_lackofenergy.csv"),
    pd.read_csv("C:/Users/Nada\MyDataSet/queries_for_lonely.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_loss of interest.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_mental health.csv"),
    pd.read_csv("C:/Users/Nada\MyDataSet/queries_for_mental illness.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_mood swing.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_negativity.csv"),
    pd.read_csv("C:/Users/Nada/MyDataSet/queries_for_opinion.csv"),
])
# Save the DataFrame as a CSV file
my_data.to_csv("C:/Users/Nada/MyDataSet/my_data.csv", index=False)


# In[13]:


import pandas as pd 
data = pd.read_csv("C:/Users/Nada/MyDataSet/my_data.csv")


# In[14]:


length = len(my_data)
print("Length of my_data:", length)


# In[ ]:





# In[3]:


my_data.isnull().sum()


# In[4]:


#remove missing values from a DataFrame
my_data.dropna(inplace= True)


# In[5]:


my_data.isnull().sum()


# In[6]:


#prendre les tweets écrits en anglais seulement
data_new = my_data[my_data['Language'] == 'en']


# In[7]:


# remove the last 4 columns
data_new = data_new.drop(['Tweet ID','Author ID','Created At','Language'], axis=1)


# In[8]:


#package nltk to cleaning data

import nltk
nltk.download('omw-1.4') #Downloading the Open Multilingual WordNet

import string
from nltk.stem import WordNetLemmatizer

#Downloading NLTK Resources
nltk.download('stopwords') #collection of common stopwords
nltk.download('punkt') #dataset contains a pre-trained model for tokenization
nltk.download('words') #English words
nltk.download('wordnet') #words and their relationships, including synonyms, antonyms, and word senses

#Initializing Variables for Text Processing
stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer() #lemmatizer: It is initialized as an instance of the WordNetLemmatizer class


# In[9]:


# remove url
import re
def url_limited(text):
    return re.sub(r'http\S+', '', text)


# In[10]:


data_new["Tweet"]= data_new["Tweet"].apply(url_limited)
data_new["Author Bio"]= data_new["Author Bio"].apply(url_limited)


# In[11]:


# Transform emojis
import demoji
import emoji
emoji_result = data_new["Tweet"].apply(lambda x: demoji.findall(x))

for item in emoji_result.tolist():
    if  item:
        print(item)


# In[12]:


# Transform emoji
data_new["Tweet"]=data_new["Tweet"].astype(str).apply(lambda x: emoji.demojize(x, delimiters=(' ', ' ')))
data_new["Author Bio"]=data_new["Author Bio"].astype(str).apply(lambda x: emoji.demojize(x, delimiters=(' ', ' ')))


# In[13]:


def punct_word(text) :
    return "".join([i.lower() for i in text if i not in string.punctuation])


# In[14]:


data_new["Tweet"]= data_new["Tweet"].apply(punct_word)
data_new["Author Bio"]= data_new["Author Bio"].apply(punct_word)


# In[15]:


#contraction2
# Dictionary of English contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "i'd": "i would", "i'd've": "i would have","i'll": "i will",
                     "i'll've": "i will have","i'm": "i am","I've": "i have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not",
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}
##contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys())) #find and match contractions in text
# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)


# In[16]:


data_new["Tweet"]= data_new["Tweet"].apply(expand_contractions)
data_new["Author Bio"]= data_new["Author Bio"].apply(expand_contractions)


# In[15]:


#tokenizing and stemming
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# Create tokens out of alphanumeric characters
tokenizer = RegexpTokenizer(r'\w+')

# Create stemmer
stemmer = PorterStemmer()

# Define function to apply stemming to a sentence
def stem_sentence(sentence):
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence.lower())
    
    # Apply stemming to each token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Return the stemmed sentence
    return ' '.join(stemmed_tokens)

# Apply stemming to the "Tweet" column of the "data" dataframe
data_new["Tweet"] = data_new["Tweet"].apply(stem_sentence)
data_new["Author Bio"] = data_new["Author Bio"].apply(stem_sentence)


# In[16]:


data_new["Tweet"]= data_new["Tweet"].apply(stem_sentence)
data_new["Author Bio"]= data_new["Author Bio"].apply(stem_sentence)


# In[17]:


# tokenize data and check that it is not a stopword
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

stopwords = set(stopwords.words('english'))

# Create tokens out of alphanumeric characters
tokenizer = RegexpTokenizer(r'\w+')

def remove_stopwords(sentence):
    tokens = tokenizer.tokenize(sentence)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return " ".join(filtered_tokens)


# In[18]:


data_new["Tweet"]= data_new["Tweet"].apply(remove_stopwords)
data_new["Author Bio"]= data_new["Author Bio"].apply(remove_stopwords)


# In[20]:


#transform new_data to a csv file
data_new.to_csv("C:/Users/Nada/MyDataSet/preprocessed_data_with_stemming.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




