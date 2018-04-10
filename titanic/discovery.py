# In[]

import pandas as pd
import numpy as np


train = pd.read_csv('//Users/paul/Sites/kaggle_competitions/titanic/data/train.csv')
train.head()
train['Age'].fillna(train['Age'].mean(), inplace=True)

# In[]
train.describe()

# In[]
train['Sex'].value_counts()

# In[]
foo = train.sort_values(by=["Age"])['Age']
foo.reset_index().Age.plot()

# In[]
def getTitle(name):
    return name[name.find(",")+1:name.find(".")]

train['Title'] = train.Name.apply(getTitle)

# In[]
train['Title'].value_counts()
