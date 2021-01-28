import pandas as pd

class ETL_1():
    def __init__(self, **kwargs):
        self.past_row = kwargs.get("initial_row")
        self.columns_to_transform = kwargs.get("columns_to_transform")
        self.mean = kwargs.get("mean")
        self.stddev = kwargs.get("stddev")
        
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
    
    
    def fillna_normalize(self, row):
        self.fillna(row)
        self.normalise(row)
        return row


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
