#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("C:/users/Nada/MyDataSet/new_data_labeled_stemmed.csv")


# In[2]:


len(data)


# In[3]:


data.head(2)


# In[4]:


data.drop(["Favorites", "Retweets","Followers"], axis=1, inplace=True)


# In[5]:


data = data.drop(columns=["Author Bio","Bio_Sentiment"], axis=1)


# In[6]:


X = data.drop(columns=["Tweet_Sentiment"])


# In[7]:


X 


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer()
X = tfidf_v.fit_transform(data['Tweet'].values.astype('U'))
y = data['Tweet_Sentiment']


# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Initialize the Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42) #number of decision trees to be created in the random forest:100

# Fit the model on the training data
rfc.fit(X_train, y_train) # fit is a method called ,trains the classifier on the training set

list_x = list(tfidf_v.get_feature_names()) #list of feature names that were used in the TF-IDF vectorization process

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[50]:


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


# In[59]:


from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# print the confusion matrix
print("Confusion Matrix:\n", cm)


# In[58]:


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


# In[52]:


from sklearn.metrics import classification_report

# Print the classification report
target_names = ['0', '1', '-1'] 
print(classification_report(y_test, y_pred, target_names=target_names))


# In[37]:


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


# In[38]:


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


# In[39]:


# aprés une modification au niveau de model 
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=3)  # 3 nearest neighbors for classification
KNN.fit(X_train,y_train)   # la méthode fit pour entrainer un modèle
  
print("Score train ",KNN.score(X_train,y_train) ) # accuracy of the model on the training data
print("Score test ",KNN.score(X_test,y_test) ) #accuracy of the model on the test data


# In[40]:


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


# In[41]:


from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=1000)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


# In[ ]:




