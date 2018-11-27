
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import pandas as pd
import time
import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[2]:


# Defines
YEAR = 2017
THRESHOLD = .60
FILL_NA = 0
DATE_CONVERSION = 'timestamps'
PREPROCESSING = 'MinMax'
KFOLD_SPLITS = 10


# In[3]:


# Functions
def check_na(train):
    # Finds the number of missing values in each column
    num_of_na = [train.loc[:,col].isnull().sum() for col in train]
    # Divide by rows for proportion 
    prop_na = [num / train.shape[0] for num in num_of_na]
    # Put the proporitons and column names into a df and sort
    na_df = pd.DataFrame({'prop_na' : prop_na, 'column' : train.columns}).sort_values('prop_na')
    return na_df


# In[4]:


# Read csvs
if YEAR == 2016:
    properties = pd.read_csv('properties_2016.csv', low_memory = False)
    train = pd.read_csv('train_2016_v2.csv', low_memory = False)
elif YEAR == 2017:
    properties = pd.read_csv('properties_2017.csv', low_memory = False)
    train = pd.read_csv('train_2017.csv', low_memory = False)
# train has Y and properties has features
# Find row intersection of train and properties
train = train.merge(properties, on = 'parcelid', how = 'left')


# In[5]:


print('BEFORE')
print(check_na(train))
# Remove all columns above the THRESHOLD
train = train.loc[:, (train.isnull().sum(axis=0) <= (train.shape[0]*THRESHOLD))]
print('AFTER')
print(check_na(train))


# TODO: fill the missing values via prediction
# 
# Types of variables:  
# - Continuous  
# - Discrete  
# - Categorical
# 
# Predict variables according to their type:  
# - Linear: continuous
# - Logistic: discrete, categorical
# 

# In[6]:


continuous = ['bathroomcnt',
              'buildingqualitytypeid',
              'calculatedbathnbr',
              'finishedfloor1squarefeet',
              'calculatedfinishedsquarefeet',
              'finishedsquarefeet6',
              'finishedsquarefeet12',
              'finishedsquarefeet13',
              'finishedsquarefeet15',
              'finishedsquarefeet50',
              'garagetotalsqft',
              'latitude',
              'longitude',
              'lotsizesquarefeet',
              'poolsizesum',
              'taxvaluedollarcnt',
              'structuretaxvaluedollarcnt',
              'landtaxvaluedollarcnt',
              'taxamount'
]

discrete = ['bedroomcnt',
            'threequarterbathnbr',
            'fireplacecnt',
            'fullbathcnt',
            'garagecarcnt',
            'numberofstories',
            'poolcnt',
            'roomcnt',
            'unitcnt'
]
categorical = ['airconditioningtypeid', 
               'architecturalstyletypeid',
               'basementsqft',
               'buildingclasstypeid',
               'decktypeid',
               'fips',
               'fireplaceflag',
               'hashottuborspa',
               'heatingorsystemtypeid',
               'pooltypeid10',
               'pooltypeid2',
               'pooltypeid7',
               'propertycountylandusecode',
               'propertylandusetypeid',
               'propertyzoningdesc',
               'rawcensustractandblock',
               'censustractandblock',
               'regionidcounty',
               'regionidcity',
               'regionidzip',
               'regionidneighborhood',
               'storytypeid',
               'typeconstructiontypeid',
               'yardbuildingsqft17',
               'yardbuildingsqft26',
               'yearbuilt',
               'assessmentyear',
               'taxdelinquencyflag',
               'taxdelinquencyyear'
]


# In[7]:


# test linear reg
from sklearn.linear_model import LogisticRegression
# Use saga solver due to large dataset
solver = 'saga' 
lr = LogisticRegression(solver=solver,
#                         multi_class=model,
#                         C=1,
#                         penalty='l1',
#                         fit_intercept=True,
#                         max_iter=this_max_iter,
                        random_state=42,
)


# In[ ]:


# xtrain is the dataset without rows which are missing a value for 
# the variable in question, 
# and ytrain is that value
variable = 'bedroomcnt'
xtrain_isna = properties.loc[:, variable].isnull()
xtrain = properties.drop(variable, axis=1).loc[np.logical_not(xtrain_isna),]
ytrain = properties.loc[np.logical_not(xtrain_isna), variable]
lr.fit(xtrain, ytrain)


# In[ ]:


# # Replace all NAs with number defined in FILL_NA
# train = train.fillna(FILL_NA)
# # Convert transactiondate strings into floats
# date_strings = (train.values[:,2])
# date_converted = []
# if DATE_CONVERSION == 'timestamps':
#     for string in date_strings:
#         date_converted.append(time.mktime(datetime.datetime.strptime(string, "%Y-%m-%d").timetuple()))
# train['transactiondate'] = np.asarray(date_converted)
# # Drop the columns with string and int
# train = train.drop(columns=['propertycountylandusecode', 'propertyzoningdesc'])


# In[ ]:


# # Preprocessing
# y = train.values[:,1]
# y = y.reshape(y.shape[0],1)
# x = train.values[:,2:]
# if PREPROCESSING == 'MinMax':
#     scaler = MinMaxScaler()
#     scaler.fit(x)
#     x = scaler.transform(x)
# # KFolds
# kf = KFold(n_splits = KFOLD_SPLITS, shuffle = True)
# for train_index, test_index in kf.split(x):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]


# In[ ]:


# model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(6, input_dim = 27, activation = 'relu'),
#         tf.keras.layers.Dense(6, activation = 'relu'),
#         tf.keras.layers.Dense(6, activation = 'relu'),
#         tf.keras.layers.Dense(1, activation = 'linear')
#     ])
# sgd = tf.keras.optimizers.SGD(lr=0.1)
# model.compile(loss = 'mse', optimizer = sgd)
# model.fit(x_train, y_train, epochs = 500, batch_size = 32, verbose = 1)


# In[ ]:


# print(model.evaluate(x_test, y_test, verbose = 1))

