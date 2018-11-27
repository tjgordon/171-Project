
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


# In[6]:



# Convert transactiondate strings into floats
date_strings = (train.values[:,2])
date_converted = []
if DATE_CONVERSION == 'timestamps':
    for string in date_strings:
        date_converted.append(time.mktime(datetime.datetime.strptime(string, "%Y-%m-%d").timetuple()))
train['transactiondate'] = np.asarray(date_converted)
# Drop the columns with string and int
train = train.drop(columns=['propertycountylandusecode', 'propertyzoningdesc'])


# In[7]:


# Preprocessing

# scales a column x of a pandas dataframe
def minmaxscaler_dropna(x):
    # Formula from:
    # https://stackoverflow.com/questions/39758449/normalise-between-0-and-1-ignoring-nan
    return((x - x.min()) / (x.max() - x.min()))

# TODO: use another scale function, like normalize, that has mean 0

def normalize_scaler(x):
    return((x - np.mean(x)) / np.std(x))


if PREPROCESSING == 'MinMax':
#     scaler = MinMaxScaler()
#     scaler.fit(x)
#     x = scaler.transform(x)
    train2 = [normalize_scaler(train[col]) for col in train.columns]



# In[8]:


type(train2)


# In[9]:


train2 = pd.DataFrame(train2).transpose()
train2.head()


# In[10]:


y = train2.values[:,1]
y = y.reshape(y.shape[0],1)
x = train2.values[:,2:]


# In[31]:


# x


# In[32]:


# # scales a column x of a pandas dataframe
# def minmaxscaler_dropna(x):
#     # Formula from:
#     # https://stackoverflow.com/questions/39758449/normalise-between-0-and-1-ignoring-nan
#     (x - x.min()) / (x.max() - x.min())


# if PREPROCESSING == 'MinMax':
# #     scaler = MinMaxScaler()
# #     scaler.fit(x)
# #     x = scaler.transform(x)
#     x2 = [minmaxscaler_dropna(x.col) for col in x.columns]
# # x.col =     
# x2


# In[11]:


# Replace all NAs with number defined in FILL_NA
train = train2.fillna(FILL_NA)


# In[12]:


# KFolds
train_index_array = []
test_index_array = []
kf = KFold(n_splits = KFOLD_SPLITS, shuffle = True, random_state = 1)
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_index_array.append(train_index)
    test_index_array.append(test_index)
MY_INDEX = 6
x_train, x_test = x[train_index_array[MY_INDEX]], x[test_index_array[MY_INDEX]]
y_train, y_test = y[train_index_array[MY_INDEX]], y[test_index_array[MY_INDEX]]


# In[ ]:


model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9, input_dim = 27, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'linear')
    ])
sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss = 'mse', optimizer = sgd)
model.fit(x_train, y_train, epochs = 500, batch_size = 32, verbose = 1)


# In[ ]:


model.evaluate(x_test, y_test, verbose = 1)

