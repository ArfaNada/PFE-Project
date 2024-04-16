#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data_all= pd.read_csv("C:/Users/Nada/MyDataSet/new_data_labeled_stemmed.csv")


# In[2]:


print(data_all.shape)


# In[3]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
#tfid modele nlp 


# In[4]:


X= data_all['Author Bio']


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer()
X = tfidf_v.fit_transform(data_all['Author Bio'].values.astype('U'))
y = data_all['Bio_Sentiment']


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# In[7]:


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


#This code saves the DecisionTreeClassifier object
#dtc to a file named 'decision_tree_model.pkl'
import pickle
with open('tfidf_v_final_bio.pkl', 'wb') as f:#bech yakra el texte
    pickle.dump(tfidf_v, f)
# Save the model to a file
filename = 'decision_tree_model_bio.pkl' #bech yekhdem elmodele 
with open(filename, 'wb') as file:
    pickle.dump(dtc, file)


# In[9]:


# Load the vectorizer object
import pickle
with open('tfidf_v_final_bio.pkl', 'rb') as f:
    tfidf_v_loaded = pickle.load(f)

# Load the model object
filename = 'decision_tree_model_bio.pkl'
with open(filename, 'rb') as file:
    dtc_loaded = pickle.load(file)

# Use the loaded objects to make predictions on new data
var = "I'm a curious explorer who enjoys discovering new things."
print(var)
data_new = pd.DataFrame([[var]], columns=['text'])
tfidf_pred = tfidf_v_loaded.transform(data_new["text"])
y_pred = dtc_loaded.predict(tfidf_pred)
print(y_pred)


# In[ ]:




