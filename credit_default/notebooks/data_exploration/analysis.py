#!/usr/bin/env python
# coding: utf-8

# # Data Exploration for Loan Default Data Set

# In[1]:


import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
output_notebook()


# ## Data Loading

# In[2]:


train_data = pd.read_csv("../data/loan-default-prediction/train_v2.csv")


# In[3]:


target = train_data["loss"]
target.describe()


# In[4]:


train_data.shape


# ### Target Column Analysis

# In[5]:


nonZeroDefaultLoss = train_data[train_data["loss"]!=0]


# In[6]:


nonZeroDefaultLoss["loss"].plot.hist(bins=20,by="loss", log=True)


# ## Categorical/Numerical column separation

# In[7]:


train_data_types = train_data.dtypes


# In[9]:


numeric_var = [key for key in dict(train_data_types)
                   if dict(train_data_types)[key]
                       in ['float64','float32']] # Numeric Variable

int_var = [key for key in dict(train_data_types)
                   if dict(train_data_types)[key]
                       in ['int32','int64']]

cat_var = [key for key in dict(train_data_types)
             if dict(train_data_types)[key] in ['object'] ] # Categorical Varible


# ### Categorical column analysis

# In[16]:


unique_integer_values = [{iv: train_data[iv].unique()} for iv in int_var]

for iv in int_var:
    print(r"{iv} has {distinct} distinct values".format(iv=iv, distinct=len(train_data[iv].unique())))
    


# In[17]:


sorted(unique_integer_values, lambda key,value: value)


# In[12]:


for cv in cat_var:
    print(r"{cv} has {distinct} distinct values".format(cv=cv, distinct=len(train_data[cv].unique())))
    


# In[15]:


train_data["f293"]


# ### Numerical column analysis

# ### Skewness and Standard Devs

# In[10]:


numeric_cols_without_id_loss = train_data[numeric_var].drop(columns=["id", "loss"])


# In[11]:


skewnesses = numeric_cols_without_id_loss.skew(axis=0)


# In[12]:


standard_devs = numeric_cols_without_id_loss.std(axis=0)


# In[13]:


columns_with_no_standard_dev = list(standard_devs[standard_devs<1e-4].index)


# In[14]:


columns_with_no_standard_dev


# ## Missing values analysis

# In[16]:


dictForMissingValues = dict(train_data.isna().any())
columnsWithMissingValues = [key for key in dictForMissingValues if dictForMissingValues[key]]


# In[17]:


r"There are {count} columns with missing values".format(count=len(columnsWithMissingValues))


# In[18]:


numeric_cols_missing_values_ratio = numeric_cols_without_id_loss.isnull().mean()


# In[29]:


columns_with_lot_of_values_missing = list(numeric_cols_missing_values_ratio[numeric_cols_missing_values_ratio > 0.1].index)


# In[30]:


columns_with_lot_of_values_missing


# ### Columns to be deleted
# 
# * All categorical columns because they have a huge cardinality
# * Data columns with more than 10% of missing values
# * Data that have 0 standard deviation

# In[31]:


columns_to_be_deleted= cat_var + columns_with_no_standard_dev + columns_with_lot_of_values_missing


# In[36]:


columns_to_be_deleted = list(set(columns_to_be_deleted))


# In[41]:


dbfile = open('../columns_to_be_deleted', 'ab') 
pickle.dump(columns_to_be_deleted, dbfile)                      
dbfile.close()


# In[ ]:




