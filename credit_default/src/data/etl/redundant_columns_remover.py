from sklearn.base import BaseEstimator, TransformerMixin

class RedundantColumnsRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, **fit_params):
        df_data_types = X.dtypes
        cat_var = [key for key in dict(df_data_types)
                         if dict(df_data_types)[key] in ['object']]

        tmp = X.drop(columns=cat_var)
        
        column_subset = tmp.columns.values
        groups = []
        redundant_columns = []
        for i in range(len(column_subset)):
            col1 = column_subset[i]
            if col1 in redundant_columns:
                    continue
            same_columns = [col1]

            for j in range(i, len(column_subset)):
                col2 = column_subset[j]
                if col1 == col2:
                    continue
                if (tmp[col1] - tmp[col2]).sum() == 0:
                    same_columns += [col2]
                    redundant_columns += [col2]
            groups+=[same_columns]

        self.columns_to_use = [i[0] for i in groups]
        return self
        
    def transform(self, X, y=None, **transform_params):
        
        return X[self.columns_to_use]