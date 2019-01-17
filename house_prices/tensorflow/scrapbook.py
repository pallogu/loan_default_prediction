from sklearn.preprocessing import StandardScaler
import pandas as pd

import os
cwd = os.getcwd()

train = pd.read_csv(cwd + '/data/train.csv')
cardinalNumericColumns = [
    'LotFrontage',
    'LotArea'
]

for column in cardinalNumericColumns:
    train[column].fillna(0, inplace=True)

scaler = StandardScaler()
train[cardinalNumericColumns] = scaler.fit_transform(train[cardinalNumericColumns])

print(train[cardinalNumericColumns].head())