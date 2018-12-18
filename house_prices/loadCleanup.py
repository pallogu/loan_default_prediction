import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os
cwd = os.getcwd()

print(cwd)

# from sklearn.decomposition import PCA

train = pd.read_csv(cwd + '/data/train.csv')
unknown = pd.read_csv(cwd + '/data/test.csv')

columnsToRemove = ['Id']
classTypeNumericColumns = ['MSSubClass']
cardinalNumericColumns = [
    'LotFrontage',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageCars',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal'
]
timeColumns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
valueColumn = 'SalePrice'

train_Y = train[valueColumn]
del train[valueColumn]

lotFrontageBase = train.groupby(['LotConfig'])['LotFrontage']

ageColumns = ['SaleMonth', 'AgeOfProperty', 'AgeOfRemodel', 'AgeOfGarage']

columnsToRemove = columnsToRemove + ['YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
columnsToRemove = columnsToRemove + ["PoolQC", "Fence", "MiscFeature", "Alley", "FireplaceQu"]

otherColumns = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] not in ['int64', 'float64']]

categoryColumns = [x for x in otherColumns if x not in columnsToRemove] + classTypeNumericColumns

# categoryColumns = ['MSZoning',
#  'Street',
#  'LotShape',
#  'LandContour',
#  'Utilities',
#  'LotConfig',
#  'LandSlope',
#  'Neighborhood',
#  'Condition1',
#  'Condition2',
#  'BldgType',
#  'HouseStyle',
#  'RoofStyle',
#  'RoofMatl',
#  'Exterior1st',
#  'Exterior2nd',
#  'MasVnrType',
#  'ExterQual',
#  'ExterCond',
#  'Foundation',
#  'BsmtQual',
#  'BsmtCond',
#  'BsmtExposure',
#  'BsmtFinType1',
#  'BsmtFinType2',
#  'Heating',
#  'HeatingQC',
#  'CentralAir',
#  'Electrical',
#  'KitchenQual',
#  'Functional',
#  'GarageType',
#  'GarageFinish',
#  'GarageQual',
#  'GarageCond',
#  'PavedDrive',
#  'SaleType',
#  'SaleCondition',
#  'MSSubClass']


def convertTimeColumnsToAgeColumns(train, test):
    for i in [train, test]:
        i['SaleMonth'] = (i['YrSold']-2000)*12 + i['MoSold']
        i['AgeOfProperty'] = i['YrSold'] - i['YearBuilt']
        i['AgeOfRemodel'] = i['YrSold'] - i['YearRemodAdd']
        i['AgeOfGarage'] = i['YrSold'] - i['GarageYrBlt']
        i['AgeOfGarage'].fillna(i['AgeOfProperty'], inplace=True)
    return train, test


def fillCardinalNumericColumns(train, test):
    for i in [train, test]:
        lotFrontageBase = i.groupby(['LotConfig'])['LotFrontage']
        i['LotFrontage'] = lotFrontageBase.transform(lambda x: x.fillna(x.mean()))

        columnsWithMissingValues = ['MasVnrArea',
                                    'BsmtFinSF1',
                                    'BsmtFinSF2',
                                    'BsmtUnfSF',
                                    'TotalBsmtSF',
                                    'BsmtFullBath',
                                    'BsmtHalfBath',
                                    'GarageCars',
                                    'GarageArea']
        for c in columnsWithMissingValues:
            i[c].fillna(0, inplace=True)
    return train, test


def scaleCardinalNumericColumns(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    return scaler.transform(train), scaler.transform(test)


def dummies(train, test, columns=categoryColumns):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix=column, dummy_na=True)[good_cols]), axis=1) 
        test = pd.concat((test, pd.get_dummies(test[column], prefix=column, dummy_na=True)[good_cols]), axis=1)
        del train[column]
        del test[column]

    return train, test       


def removedUnusedColumns(train, test, columns=columnsToRemove):
    for column in columns:
        del train[column]
        del test[column]

    return train, test


def prepareTarget(data):
    return np.array(data.SalePrice, dtype='float64').reshape(-1, 1)


def separate_cardinal_categorical_input(train, test):
    tr_car = train[cardinalNumericColumns]
    test_car = test[cardinalNumericColumns]
    tr_cat, test_cat = removedUnusedColumns(train, test, columns=cardinalNumericColumns)
    return tr_car, tr_cat, test_car, test_cat


def getTrainTestDFs():
    tr, un = convertTimeColumnsToAgeColumns(train, unknown)
    tr, un = fillCardinalNumericColumns(tr, un)
    tr, un = removedUnusedColumns(tr, un)
    tr, un = dummies(tr, un)
    # tr, un = scaleCardinalNumericColumns(tr, un)
    tr_car, tr_cat, un_car, un_cat = separate_cardinal_categorical_input(tr, un)
    return tr_car, tr_cat, un_car, un_cat

def getValueColumn():
    return train_Y
