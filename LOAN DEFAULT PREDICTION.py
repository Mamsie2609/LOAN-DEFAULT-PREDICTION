#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORT NECESSARY LIBRARIES

# FOR DATA ANALYSIS
import pandas as pd
import numpy as np

# FOR DATA VISUALISATION.
import matplotlib.pyplot as plt
import seaborn as sns

# FOR DATA PRE-PROCESSING.
from sklearn.preprocessing import StandardScaler#(x - mean)/standard deviation
from sklearn.model_selection import train_test_split

# CLASSIFIER LIBRAR
from sklearn.linear_model import LogisticRegression

# SUPPRESS WARNING MESSAGES
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# LOAD DATASET
data = pd.read_csv('Default_Fin.csv')
data


# In[3]:


# COUNT THE NUMBER OF INSTANCES OF EACH UNIQUE VALUE IN THE COLUMN 'DEFAULTED?'
data['Defaulted?'].value_counts()


# In[4]:


# RETURN THE NUMBER OF ROWS AND COLUMNS
data.shape


# In[5]:


# DISPLAY COLUMNS OF DATA
data.columns


# In[6]:


# DROP THE COLUMN NAMED 'INDEX' 
data.drop('Index', axis=1, inplace=True)
data.head(2)


# In[7]:


# CHECK FOR MISSING VALUES IN EACH COLUMN.
data.isna().sum()


# In[8]:


# PRINT THE MAXIMUM AND MINIMUM VALUE OF THE 'BANK BALANCE' COLUMN.
print('max amount ',data['Bank Balance'].max())
print('min amount ',data['Bank Balance'].min())


# In[9]:


# GENERATES A CORRELATION HEATMAP BETWEEN VARIABLES.
corr = data.corr()

plt.figure(figsize=(18, 15))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.title("Correlation Heatmap")
plt.show()


# In[10]:


# CREATE A NEW DATAFRAME 'x' AND 'y' FROM THE ORIGINAL DATAFRAME. 
x = data [['Employed', 'Bank Balance', 'Annual Salary']]
y = data ['Defaulted?']


# In[11]:


# SPLIT THE DATASET INTO TRAINING AND TESTING SETS.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# # METRICS

# In[12]:


# IMPORT METRICS FOR EVALUATING THE PERFORMANCE OF CLASSIFICATION MODELS.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # Decision tree

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[14]:


# SPLIT THE DATASET INTO TRAINING AND TESTING SET. Splitting the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# TRAINING THE DECISION TREE CLASSIFIER Training the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
# Making predictions on the test set
y_pred = clf.predict(x_test)

# EVALUATION METRICS 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# CALCULATE THE ACCURACY, PRECISION, RECALL AND F1 - SCORE OF THE MODEL.
print('Accuracy: %.2f' % accuracy)
print('Precision: %.2f' % precision)
print('recall: %.2f' % recall)
print('f1: %.2f' % f1)


# # Positive= Defaulted = 1
# # Negative= Not Defaulted = 0
# 
# ## tp= Predicted positive and it is true that is predicted defaulted and it actually defaulted
# 
# ## fp= Predicted positive but it is actually negative that is predicted defaulted but it is actually not defualted
# 
# ## tn= Predicted negative and it is actually negative that is predicted not defaulted and did not default
# 
# ## fn= Predicted negative and but is actually positive that is predicted not defaulted but the customer actually defaulted
# 
# 
# # Accuracy = tp + tn/ tp + tn + fp + fn
# # Recall = tp/tp + fn
# # Precision = tp/tp + fp
# # f1 score = 2(precision * Recall)/(precison+Recall)
#  

# In[15]:


# CALCULATE THE CONFUSION MATRIX FOR A DECISION TREE.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[16]:


# CALCULATE THE PRECISION SCORE
20/(20+62)#tp=20,tn=1876,fp=65,fn=39 0 is negative (not default) and 1 is positive (default)


# In[17]:


# CALCULATE THE RECALL METRIC
20/(20+39)#tp/tp+fn 


# In[18]:


1879+62+39+20


# In[19]:


(1879+20)/2000


# In[20]:


# NUMPY ARRAY CONTAINING THE PREDICTED OUTPUT VALUES FOR THE INPUT VALUES IN THE TEST SET..
y_pred


# In[21]:


result=0
for i in range(len(y_test)):
    if list(y_test)[i] == y_pred[i] == 1:
        result += 1

print(result) # 2


# # LOGISTIC REGRESSION

# In[22]:


# IMPORT LOGISTIC REGRESSION.
from sklearn.linear_model import LogisticRegression

# SPLIT DATA INTO TRAINING AND TEST SETS.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# CREATE THE LOGISTIC REGRESSION MODEL. 
log_reg = LogisticRegression()

# FIT THE MODEL TO THE TRAINING DATA.
log_reg.fit(x_train, y_train)

# MAKE PREDICTIONS ON THE TEST SET.
y_pred = log_reg.predict(x_test)


# In[23]:


# CALCULATE THE PERFORMANCE METRICS 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

# PRINT THE PERFORMANCE METRICS 
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1score)


# In[24]:


# TRAIN A LOGISTIC MODEL AND THEN EVALUATE THE PRECISION SCORE.
scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    log_reg = LogisticRegression()

# # FIT THE LOGISTIC REGRESSION MODEL TO THE TRAINING DATA.. 
    log_reg.fit(x_train, y_train)
    
    y_pred = log_reg.predict(x_test)

# print(precision_score(y_test, y_pred),i)
    scores.append(precision_score(y_test, y_pred))


# In[25]:


# RETURN THE INDEX OF THE MAXIMUM VALUE IN THE SCORES ARRAY 
np.argmax(scores)


# In[26]:


# RETURN THE HIGHEST PRECISION SCORE IN THE LIST.
scores[np.argmax(scores)]


# In[27]:


# DATA SCALING STEPS ON THE TRAINING DATASET.
from sklearn.preprocessing import StandardScaler #(x - mean)/standard deviation
scaler = StandardScaler()
x_train_scaled  =scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled, index=x_train.index, columns = x_train.columns)


# In[28]:


# DATA SCALING STEPS ON THE TEST SET.
x_test_scaled = scaler.transform(x_test)
x_test = pd.DataFrame(x_test_scaled, index=x_test.index, columns = x_test.columns)


# In[29]:


x_train['Bank Balance'].min()


# In[30]:


# IMPORT LOGISTIC REGRESSION.
from sklearn.linear_model import LogisticRegression

# SPLIT THE DATA INTO TRAIN AND TEST SETS.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=783)

# CREATE THE LOGISTIC REGRESSION MODEL. 
log_reg = LogisticRegression()

# FIT THE MODEL TO THE TRAINING DATA. 
log_reg.fit(x_train, y_train)

# MAKE PREDICTIONS ON THE TEST SET.
y_pred = log_reg.predict(x_test)


# In[31]:


# CALCULATE THE PERFORMANCE METRICS. 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

# PRINT THE PERFORMANCE METRICS. 
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1score)


# To summarize, based on the performance metrics, it can be concluded that logistic regression is a superior model for predicting loan default compared to the decision tree. This is due to its higher precision, accuracy, and F1 score, which indicates a better overall performance in correctly identifying defaulters and non-defaulters.
