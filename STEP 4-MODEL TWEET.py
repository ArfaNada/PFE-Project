#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data_all= pd.read_csv("C:/Users/Nada/MyDataSet/new_data_labeled_stemmed.csv")


# In[2]:


print(data_all.shape)


# In[4]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


# In[5]:


X= data_all['Tweet']


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer()
X = tfidf_v.fit_transform(data_all['Tweet'].values.astype('U'))
y = data_all['Tweet_Sentiment']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)


# In[8]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# initialize the decision tree classifier
dtc = DecisionTreeClassifier(random_state=42)

# fit the model on the training data
dtc.fit(X_train, y_train)

# make predictions on the test data
y_pred = dtc.predict(X_test)

# calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy score:", accuracy)


# In[8]:


# Save the vectorizer object
import pickle
with open('tfidf_v_final_tweet.pkl', 'wb') as f:#to read text
    pickle.dump(tfidf_v, f)
# Save the model to a file
filename = 'decision_tree_model_tweet.pkl' 
with open(filename, 'wb') as file:
    pickle.dump(dtc, file)


# In[10]:


# Load the vectorizer object
import pickle
with open('tfidf_v_final_tweet.pkl', 'rb') as f:
    tfidf_v_loaded = pickle.load(f)

# Load the model object
filename = 'decision_tree_model_tweet.pkl'
with open(filename, 'rb') as file:
    dtc_loaded = pickle.load(file)

# Use the loaded objects to make predictions on new data
var='i have demons in my head'
print(var)
data_new = pd.DataFrame([[var]], columns=['text'])

tfidf_pred = tfidf_v_loaded.transform(data_new["text"])
y_pred = dtc_loaded.predict(tfidf_pred)
print(y_pred)


# In[ ]:




