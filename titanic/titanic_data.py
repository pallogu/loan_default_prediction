import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


source = pd.read_csv('//Users/paul/Sites/kaggle_competitions/titanic/data/train.csv')
source['Age'].fillna(source['Age'].mean(), inplace=True)

def hasAlias(name):
    return True if "(" in name else False

def getTitle(name):
    return name[name.find(",")+1:name.find(".")]

source['Title'] = source.Name.apply(getTitle)
source['Deck'] = source['Cabin'].astype(str).str[0]
source['HasAlias'] = source.Name.apply(hasAlias)

categoryFeatures = [
    'Pclass',
    'Sex',
    'Parch',
    'Embarked',
    'HasAlias',
    'Title',
    'Deck',
]

numberFeatures = [
    "Age", 
    "SibSp", 
    "Fare"
]

train=source.sample(frac=0.8)
test=source.drop(train.index)

train_y = train['Survived'].values
test_y = test['Survived'].values

def getTrainingSet():
    categories = {x: train[x].apply(str).values for x in  categoryFeatures}
    numbers = {x: train[x].values for x in  numberFeatures}
    return tf.data.Dataset.from_tensor_slices(({**categories, **numbers}, train_y))

def getTestSet():
    categories = {x: test[x].apply(str).values for x in  categoryFeatures}
    numbers = {x: test[x].values for x in  numberFeatures}
    return tf.data.Dataset.from_tensor_slices(({**categories, **numbers}, test_y))

def getFeatureDefs():
    categories = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        key = x,
        vocabulary_list=source[x].apply(str).unique()
    )) for x in categoryFeatures]

    numbers = [tf.feature_column.numeric_column(
        key = x
    ) for x in numberFeatures]

    return categories + numbers