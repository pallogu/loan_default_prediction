import pandas as pd
from pandas.testing import assert_frame_equal

class ETL_1():
    def __init__(self, **kwargs):
        self.columns_to_transform = kwargs.get("columns_to_transform")
        self.mean = kwargs.get("mean")
        self.stddev = kwargs.get("stddev")
        
    def fillna(self, row):
        missing_value_columns = row.loc[row.isnull()].index
        if len(missing_value_columns):
            row[missing_value_columns] = self.mean[missing_value_columns]
        return row
    
    def normalise(self, row):
        cols = self.columns_to_transform
        row[cols] = (row[cols]-self.mean[cols])/self.stddev[cols]
        return row
    
    
    def fillna_normalize(self, row):
        self.fillna(row)
        self.normalise(row)
        return row

# +
inpt_df = pd.DataFrame(data=[
    [1, 10, 100, None],
    [2, 20, 200, 45],
    [3, 30, 600, 55]
], columns=["f_0", "f_1", "f_2", "f_3"])

expected = pd.DataFrame(data=[
    [1 , -1, -0.75592895, 50],
    [2, 0, -0.37796447, 45],
    [3, 1,  1.13389342, 55]
], columns=["f_0", "f_1", "f_2", "f_3"])

inpt_mean = inpt_df.mean()
inpt_stddev = inpt_df.std()

etl_1 = ETL_1(
    initial_row=inpt_mean,
    mean = inpt_mean,
    stddev=inpt_stddev,
    columns_to_transform=["f_1", "f_2"]
)

actual = inpt_df.apply(etl_1.fillna_normalize, axis=1)

assert_frame_equal(actual, expected, check_dtype=False)
# -



class ETL_2():
    def __init__(self, **kwargs):
        self.columns_to_transform = kwargs.get("columns_to_transform")
        self.trans_cols_names = kwargs.get("trans_cols_names")
        self.columns_to_keep_train = kwargs.get("columns_to_keep_train")
        self.columns_to_keep = kwargs.get("columns_to_keep")
        self.pca = kwargs.get("pca")
    
    def pca_transform(self, row):
        cols = self.columns_to_transform
        row_trans = self.pca.transform(row[cols].values.reshape(1, -1))
        return pd.Series(row_trans[0], index=self.trans_cols_names)
        
    def reduce_columns_train(self, row):
        to_keep = row[self.columns_to_keep_train]
        pca_transformed = self.pca_transform(row)

        row_trans = pd.concat([to_keep, pca_transformed])
        
        return row_trans
    
    def reduce_columns(self, row):
        to_keep = row[self.columns_to_keep]
        pca_transformed = self.pca_transform(row)

        row_trans = pd.concat([to_keep, pca_transformed])
        
        return row_trans
