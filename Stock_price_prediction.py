#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


orginaldataset = pd.read_excel("Stock_Price_data_set.xlsx")


# In[69]:


orginaldataset


# In[70]:


modifieddataset = orginaldataset # to access the orginal dataset when ever we want we are creating a copy and performing feature engineering

# feature Engineering 

date = modifieddataset.pop("Date") #removing the date column for evaluation


# In[71]:


modifieddataset


# In[72]:


df= pd.DataFrame(modifieddataset)


# In[73]:


X = df.drop(['Volume','Adj Close'],axis=1)
X # independent variables


# In[74]:


Y = X.pop("Close")


# In[75]:


Y #dependent variables


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3 , random_state =0)


# In[78]:


X_train


# In[79]:


X_test


# In[80]:


Y_train


# In[81]:


from sklearn.linear_model import LinearRegression 


# In[82]:


reg = LinearRegression()


# In[83]:


reg.fit(X_train,Y_train) 


# In[84]:


predictX = reg.predict(X_train)


# In[85]:


from sklearn import metrics
train_error_score = metrics.r2_score(Y_train, predictX) 
print("R squared Error - Training : ", train_error_score) # prediction on Training data 


# In[86]:


predictY = reg.predict(X_test)


# In[87]:


test_error_score = metrics.r2_score(Y_test, predictY) 
print("R squared Error - Test: ", test_error_score) 


# In[115]:


predictY 


# In[119]:


Y_test.values


# In[ ]:




