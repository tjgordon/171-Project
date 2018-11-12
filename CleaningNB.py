
# coding: utf-8

# In[1]:


#!/usr/bin/env python3


import numpy as np
import pandas as pd
import seaborn as sns


#%%
# Data source: 
# https://www.kaggle.com/c/zillow-prize-1/data
properties = pd.read_csv('properties_2017.csv')

#%%
train = pd.read_csv('train_2017.csv')
#train.columns


# In[2]:




#%%
properties.columns
properties.shape

#for col in properties:
#    #pass
#    #print(col)
#    print(col, properties.loc[:,col].isnull().sum())
#    #col.values.isnull()
#    
    
# Finds the number of missing values in each column
num_of_na = [properties.loc[:,col].isnull().sum() for col in properties]

# Divide by rows for proportion 
prop_na = [num / properties.shape[0] for num in num_of_na]

# sanity check
len(prop_na)


#%%
# Put the proporitons and column names into a df
na_df = pd.DataFrame({'prop_na' : prop_na, 'column' : properties.columns})
na_df = na_df.sort_values('prop_na')
print(na_df)


# In[3]:



#%%
ax = sns.barplot(data=na_df, y='column', x='prop_na')
print(ax)


# ## Missing values  
# Create 2 data sets:  
# 1. Remove columns with mostly missing and predict the rest  
# 2. Missing values filled with -99999999999999  

# For NN:  
# - Remove columns with most missing 
# - fill rest with -99999999999
# 
# For Xgboost:  
# - missing values OK  
# 
# pandas fill  

# ## 1.

# ## 2.

# In[4]:


properties.head()

