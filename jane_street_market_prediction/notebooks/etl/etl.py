import pandas as pd
from sklearn.decomposition import PCA

train = pd.read_csv("../../input/train.csv")

train.iloc[3]

test_df = pd.DataFrame(
    data=[
        {
            "feat_0": None,
            "feat_1": 2,
            "feat_2": 3
        },
        {
            "feat_0": 2,
            "feat_1": 1,
            "feat_2": 3
        },
        {
            "feat_0": 1,
            "feat_1": 1,
            "feat_2": 3
        },
        {
            "feat_0": None,
            "feat_1": 1,
            "feat_2": 3
        },
        {
            "feat_0": 1,
            "feat_1": None,
            "feat_2": 3
        }
    ],
    columns = ["feat_0", "feat_1", "feat_2"]
)

test_df

test_mean = test_df.mean()
test_std = test_df.std()


# +
class ETL():
    def __init__(self, **kwargs):
        self.past_row = kwargs.get("initial_row")
        self.columns_to_transform = kwargs.get("columns_to_transform")
        self.mean = kwargs.get("mean")
        self.stddev = kwargs.get("stddev")
        self.pca = kwargs.get("pca")
        
    def fillna(self, row):
        missing_value_columns = row.loc[row.isnull()].index
        if len(missing_value_columns):
            row[missing_value_columns] = self.past_row[missing_value_columns]
        self.past_row = row.copy()
        return row
    
    def normalise(self, row):
        cols = self.columns_to_transform
        row[cols] = (row[cols]-self.mean[cols])/self.stddev[cols]
        return row
    
    def pca_transform(self, row):
        cols = self.columns_to_transform
        row_trans = self.pca.transform(row[cols])
        return row_trans
    
    def transform(self, row):
        missing_value_columns = row.loc[row.isnull()].index
        if len(missing_value_columns):
            row[missing_value_columns] = self.past_row[missing_value_columns]
        self.past_row = row.copy()
        
        cols = self.columns_to_transform
        row[cols] = (row[cols]-self.mean[cols])/self.stddev[cols]
        
#         self.fillna(row)
#         self.normalise(row)
        
        return row
    
etl = ETL(initial_row = test_mean, columns_to_transform=["feat_0", "feat_1"], mean=test_mean, stddev=test_std)
# -

test_df.apply(etl.transform, axis=1)

# %%time
test_df_2.apply(etl.fillna, axis=1)

feats = ["feature_{count}".format(count = count) for count in range(1, 130)]

train_etl = ETL(initial_row=train.mean(), mean = train.mean(), stddev=train.std(), columns_to_transform=feats)

# %%time
train_trans  = train.apply(train_etl.transform, axis=1)

train_trans


