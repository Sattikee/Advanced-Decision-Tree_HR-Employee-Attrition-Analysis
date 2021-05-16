#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df=pd.read_csv(r"C:\Users\HP\Downloads\Python materials\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df


# In[9]:


sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[10]:


df.head()


# In[11]:


df.info()


# In[12]:


pd.set_option("display.float_format","{:.1f}".format)


# In[13]:


df.describe()


# In[15]:


df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis='columns',inplace=True)


# In[16]:


categorical_col=[]
for column in df.columns:
    if df[column].dtype==object and len(df[column].unique())<=50:
        categorical_col.append(column)
        print(f"{column}:{df[column].unique()}")
        print("========================")


# In[17]:


categorical_col


# In[18]:


df['Attrition1']=df['Attrition'].map({'Yes':1,'No':0})


# In[19]:


df.head()


# In[20]:


df.drop('Attrition',axis=1,inplace=True)


# In[22]:


#df['Attrition1']=df.Attrition1.astype("category").cat.codes #converttocategorical


# In[23]:


df.head(2)


# In[24]:


df.Attrition1.value_counts()


# In[28]:


#Visualizing the distribution of the data for every feature
df.hist(edgecolor='black', color='pink', linewidth=1.2, figsize=(20, 20))


# In[39]:


#Plotting how every feature correlate with the "target"
sns.set(font_scale=1.5)
plt.figure(figsize=(40,40))

sns.set(font_scale=1.2)
plt.figure(figsize=(40,40))

for i,column in enumerate(categorical_col, 1):
    plt.subplot(3,3,i)
    g=sns.barplot(x=f"{column}",y=df['Attrition1'],data=df)
    plt.ylabel('Attrition Count')
    plt.xlabel(f'{column}')


# In[36]:


df.head(2)


# In[ ]:


Business Travel : The workers who travel a lot are more likely to quit than other employees.
Department:The worker in Research and Development are more likely to stay than the workers on other department.
Educationfield: The workers with Human Resources and Technical Degree are more likely to quit than employees from other fields of educations.
Gender:The Male are more likely to quit.
JobRole:The workers in Laboratory Technician, Sales Representative, and Human Resources are more likely to quit the workers in other positions.
MaritalStatus: The workers who have single marital status are more likely to quit than married or divorced.
Overtime:The workers who work more hours are likely to quit than others.


# In[40]:


plt.figure(figsize=(30, 30))
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})


# In[ ]:


# Transform categorical data into dummies
# categorical_col.remove("Attrition")
# data = pd.get_dummies(df, columns=categorical_col)
# data.info()
#from sklearn.preprocessing import LabelEncoder
#label = LabelEncoder()
#for column in categorical_col:
    #df[column] = label.fit_transform(df[column])


# In[41]:


from sklearn.model_selection import train_test_split

X = df.drop('Attrition1', axis=1)
y = df.Attrition1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[42]:


y_train.shape


# In[43]:


y_test.shape


# In[44]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[45]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[ ]:


Decision Tree parameters:

criterion: The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.

splitter: The strategy used to choose the split at each node. Supported strategies are "best" to choose the best split and "random" to choose the best random split.

max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

min_samples_split: The minimum number of samples required to split an internal node.

min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

max_features: The number of features to consider when looking for the best split.

max_leaf_nodes: Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

min_impurity_split: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.


# In[48]:


df.head(2)


# In[46]:


from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)


# In[50]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[51]:


params = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
}


# In[52]:


tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=2, cv=5)
tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")


# In[ ]:


tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(X_train, y_train)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)


# In[ ]:


get_ipython().system('pip install six')


# In[ ]:


from six import StringIO


# In[ ]:


get_ipython().system('pip install --upgrade scikit-learn==0.20.3')


# In[ ]:


from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(df.columns)
features.remove("bank")


# In[ ]:




