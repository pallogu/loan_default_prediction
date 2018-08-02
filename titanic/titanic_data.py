import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
from tensorflow.python.ops import resources

# source = pd.read_csv('//Users/paul/Sites/kaggle_competitions/titanic/data/train.csv')
source = pd.read_csv('//home/paul/dev/kaggle_competitions/titanic/data/train.csv')
# source_predict = pd.read_csv('//Users/paul/Sites/kaggle_competitions/titanic/data/test.csv')
source_predict = pd.read_csv('//home/paul/dev/kaggle_competitions/titanic/data/test.csv')

allData = pd.concat([source, source_predict], axis=0)
dataSets = [source, source_predict]

source['Age'].fillna(source['Age'].mean(), inplace=True)
source_predict['Age'].fillna(source['Age'].mean(), inplace=True)

def hasAlias(name):
    return True if "(" in name else False

def getTitle(name):
    return name[name.find(",")+1:name.find(".")]

allData['Title'] = allData.Name.apply(getTitle)
stat_min = 10 
title_names = (allData['Title'].value_counts() < stat_min)

for dataset in dataSets:
    dataset['Age'].fillna(allData['Age'].mean(), inplace=True)
    dataset['Embarked'].fillna(allData['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(allData['Fare'].mean(), inplace=True)
    dataset['HasAlias'] = dataset.Name.apply(hasAlias)
    dataset['Title'] = dataset.Name.apply(getTitle)
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    dataset['Deck'] = dataset['Cabin'].astype(str).str[0]
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

ageMean = allData['Age'].mean()
ageStd = allData['Age'].std()

fareMean = allData['Fare'].mean()
fareStd = allData['Fare'].std()

for dataset in dataSets:
    dataset['Age'] = (dataset['Age'] - ageMean)/ageStd
    dataset['Fare'] = (dataset['Fare'] - fareMean)/fareStd

categoryFeatures = [
    'Pclass',
    'Sex',
    # 'Parch',
    'Embarked',
    # 'HasAlias',
    'Title',
    'Deck',
    'IsAlone'
]

numberFeatures = [
    "Age",
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

def getPredictSet():
    categories = {x: source_predict[x].apply(str).values for x in  categoryFeatures}
    numbers = {x: source_predict[x].values for x in  numberFeatures}
    return tf.data.Dataset.from_tensor_slices(({**categories, **numbers}, {**categories, **numbers}))

def getPreditIds():
    return source_predict['PassengerId']

def getFeatureDefs():
    categories = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        key = x,
        vocabulary_list=source[x].apply(str).unique()
    )) for x in categoryFeatures]

    # numbers = [tf.feature_column.numeric_column(
    #     key = x
    # ) for x in numberFeatures]

    age = tf.feature_column.numeric_column(key="Age")
    age_bucket = tf.feature_column.bucketized_column(age, list(range(0,100, 5)))

    fare = tf.feature_column.numeric_column(key="Fare")
    fare_bucket = tf.feature_column.bucketized_column(fare, list(range(math.floor(allData['Fare'].min()), math.ceil(allData["Fare"].max()), 50)))

    return categories + [age_bucket]