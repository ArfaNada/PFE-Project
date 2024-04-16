#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("C:/users/Nada/MyDataSet/new_data_labeled_stemmed.csv")


# In[2]:


len(data)


# In[3]:


data.drop(["Favorites", "Retweets","Followers"], axis=1, inplace=True)


# In[4]:


#data = data.drop(columns=["Author Bio","Bio_Sentiment"], axis=1)
data = data.drop(columns=["Tweet","Tweet_Sentiment"], axis=1)


# In[5]:


#X = data.drop(columns=["Tweet_Sentiment",'Author Bio','Bio_Sentiment'])
X = data.drop(columns=["Bio_Sentiment"])
#X = data["Tweet"]


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer()
X = tfidf_v.fit_transform(data['Author Bio'].values.astype('U'))
y = data['Bio_Sentiment']


# In[7]:


X


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Initialize the Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rfc.fit(X_train, y_train)

list_x = list(tfidf_v.get_feature_names())

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[10]:


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


# In[9]:


from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# print the confusion matrix
print("Confusion Matrix:\n", cm)


# In[12]:


# plot confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['0:Neutre', '1:Positif', '-1:Négatif'], yticklabels=['0:Neutre', '1:Positif', '-1:Négatif'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()


# In[13]:


from sklearn.metrics import classification_report

# Print the classification report
target_names = ['0:Neutre', '1:Positif', '-1:Négatif'] # Change these to match your class names
print(classification_report(y_test, y_pred, target_names=target_names))


# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the logistic regression classifier
logreg = LogisticRegression(random_state=42)

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[12]:


from sklearn.svm import SVC

#Initialize the SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)
#Fit the model on the training data:
svm.fit(X_train, y_train)
# Make predictions on the test data:
y_pred = svm.predict(X_test)
# Calculate the accuracy score:
accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[33]:


# aprés une modification au niveau de model 
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=3)  # n_neighbors=3
KNN.fit(X_train,y_train)   # la méthode fit pour entrainer un modèle
  
print("Score train ",KNN.score(X_train,y_train) ) # la méthode score pour calculer le test_score ou le train_score
print("Score test ",KNN.score(X_test,y_test) )


# In[13]:


# Convert X_train and X_test from sparse matrix to dense numpy array
X_train = X_train.toarray()
X_test = X_test.toarray()

# Fit the Naive Bayes classifier on the training data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_naive_bayes = classifier.predict(X_test)

# Calculate the accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_naive_bayes)
accuracy


# In[14]:


from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=1000)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


# In[20]:



'''
var = 'overwork,loneliness,difficult or traumatic events can promote depression i lost touch with reality,psychosis then effectes thoughts,emotions feelings and behaviours overworker loneliness'
print(var)
data_new = pd.DataFrame([[var]], columns=['text'])
data_new
tfidf_pred=tfidf_v.transform(data_new["text"])
tfidf_pred
dtc.predict(tfidf_pred)  #1  predict resultat apres avoir appliqué le modele
'''


# In[ ]:




