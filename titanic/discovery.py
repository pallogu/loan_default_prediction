# In[]

import pandas as pd
import numpy as np


train = pd.read_csv('/home/paul/dev/kaggle_competitions/titanic/data/train.csv')
train.head()

# In[]
train.describe()

# In[]
foo = train.sort_values(by=["Age"])['Age']
foo.reset_index().Age.plot()

# In[]
myString = "Foo, Mr. Bar"
def getTitle(name):
    return name[name.find(",")+1:name.find(".")]

train.title = train.Name.apply(getTitle)
