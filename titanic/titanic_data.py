import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


train = pd.read_csv('//Users/paul/Sites/kaggle_competitions/titanic/data/train.csv')
train['Age'].fillna(train['Age'].mean(), inplace=True)

def hasAlias(name):
    return True if "(" in name else False

def getTitle(name):
    return name[name.find(",")+1:name.find(".")]

train['Title'] = train.Name.apply(getTitle)
train['Deck'] = train['Cabin'].astype(str).str[0]
train['HasAlias'] = train.Name.apply(hasAlias)

countVectorizerMap =[
    {'name':'Pclass', 'analyzer':'char'},
    {'name': 'Sex', 'analyzer': 'word'},
    {'name': 'Parch', 'analyzer': 'char'},
    {'name': 'Embarked', 'analyzer': 'char'},
    {'name': 'HasAlias', 'analyzer': 'word'},
    {'name': 'Title', 'analyzer': 'word'},
    {'name': 'Deck', 'analyzer': 'char'}
]


def _fitTransform(column):
    vectorizer = CountVectorizer(analyzer=column['analyzer'], lowercase=False)
    matrix = vectorizer.fit_transform(train[column['name']].apply(str)).todense()
    return [ matrix, vectorizer ]


fields = [_fitTransform(x) for x in countVectorizerMap]
categoryFeatures, vectorizers = zip(*fields)

x = np.hstack([train.as_matrix(["Age", "SibSp", "Fare"]), np.hstack(list(categoryFeatures))])
# x = np.hstack([train.as_matrix(["Age", "Fare"]), np.hstack(list(categoryFeatures))])

# labelVectorizer = CountVectorizer(analyzer='char', lowercase=False)
# y = labelVectorizer.fit_transform(train['Survived'].apply(str)).todense()
y = train['Survived'].values
def getTrainingSet():
    return (x, y)
