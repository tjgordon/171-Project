
# coding: utf-8

# In[ ]:


#This notebook cleans data from a real estate dataset creating a model 
#which can predict a house's valuation using its characteristics

#The final model has an accuracy of 97.8% (mean error percentage 2.2%) and a median absolute error of 1.67


# In[ ]:


#IMPORT LIBRARIES


# In[2]:


import os; import pandas as pd; import numpy as np


# In[ ]:


#LOAD DATA


# In[1]:


import CleaningNB


# In[ ]:


filename = 'zillow'


# In[4]:


#master = pd.read_csv(filename) #dataset with 80 variables
# The selection of columns was done for the NN model, so use the full dataset here
master = CleaningNB.properties
backup = master


# In[ ]:


#START RUNNING CODE FROM HERE

master = backup
master.shape


# In[ ]:


#remve unusable columns

master = backup.drop(columns='region_2')
master.shape


# In[ ]:


#remove INF and NaNs, replace with special value            #EDIT: just drop those rows since there's too many anyway

master = master.replace([np.inf, -np.inf], np.nan)
master = master.dropna()
master.shape


# In[ ]:


#reduce number of rows

master = master.sample(frac=0.35)
master.shape


# In[ ]:


#identify variables with highest nunique

master.nunique()


# In[ ]:


#IDENTIFY RELEVANT COLUMNS

data = master.columns.drop(['new_id', 'points'])
dataset = master[data]
master.shape


# In[ ]:


#reduce number of columns (after categorical -> indicator) by dropping variable with highest nunique

data = data.drop('designation')
dataset = master[data]
dataset.nunique()


# In[ ]:


#use dummy encoder to convert categorical variables to indicators

nonNumeric = data.drop(dataset[data].select_dtypes('number').columns)

dataNon = dataset[nonNumeric]

dataDummy = pd.get_dummies(dataNon)


# In[ ]:


#replace categorical variables with indicator variables 

dataset = dataset.drop(columns=nonNumeric)

dataset[dataDummy.columns] = dataDummy


# In[ ]:


#update data then proceed to model

data = dataset.columns


# In[ ]:


#SPLIT INTO TRAIN AND TEST

from sklearn.model_selection import train_test_split

trainData, testData, trainTarget, testTarget = train_test_split(dataset, master['points'], 
                                                                test_size = 0.4, random_state = 42)


# In[ ]:


#sanity check

trainData.shape


# In[ ]:


#   4-STEP MODELLING PROCESS: IMPORT WHICH MODEL, MAKE INSTANCE OF MODEL, TRAIN USING FIT, PREDICT LABELS OF TESTDATA


# In[1]:


from xgboost import XGBRegressor


# In[2]:


model = XGBRegressor(base_score=)


# In[ ]:


model.fit(trainData, trainTarget)


# In[ ]:


prediction = model.predict(testData)   #predict probabilities, for ROC and KS
prediction


# In[ ]:


prediction.mean()


# In[ ]:


#EVALUATE MODEL USING METRICS


# In[ ]:


model.score(trainData, trainTarget)


# In[ ]:


#error percentage

((prediction - testTarget)/testTarget * 100).abs().mean()


# In[ ]:


#side-by-side comparison

from sklearn.metrics import median_absolute_error as scr

scr(testTarget, prediction)


# In[ ]:


from sklearn.metrics import mean_squared_log_error as scor
scor(testTarget, prediction)            #multioutput not necessary


# In[ ]:


from sklearn.metrics import explained_variance_score as scorev
scorev(testTarget, prediction)            #multioutput not necessary

