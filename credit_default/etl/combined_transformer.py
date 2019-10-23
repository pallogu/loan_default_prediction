from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import pandas as pd

from .category_encoder import CategoryEncoder

class CombinedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.non_categorical_columns = []
        
        self.categorical_pipeline = Pipeline([
            ("simple_imputer", SimpleImputer(strategy="constant", fill_value=-100)),
            ("categorical encoder", CategoryEncoder(columns_to_encode = categorical_columns))
        ])
        self.numerical_pipeline = Pipeline([
            ("power_transformer", PowerTransformer()),
            ("simple_imputer", SimpleImputer())
        ])
        
    def fit(self, X, **fit_params):
        X_categorical_pd = X[self.categorical_columns]
        X_without_categorical = X.drop(columns = self.categorical_columns)
        
        self.non_categorical_columns = X_without_categorical.columns.values
        
        
        self.categorical_pipeline.fit(X_categorical_pd)
        self.numerical_pipeline.fit(X_without_categorical)
        
        return self
        
    def transform(self, X, y=None, **transform_params):
        X_categorical_pd = X[self.categorical_columns]
        X_without_categorical = X.drop(columns = self.categorical_columns)
        
        X_categorical_transformed = self.categorical_pipeline.transform(X_categorical_pd)
        X_without_categorical_transformed = self.numerical_pipeline.transform(X_without_categorical)

        X_without_categorical_transformed_pd = pd.DataFrame(
            data=X_without_categorical_transformed,
            columns=self.non_categorical_columns
        )
        
        return pd.concat([X_categorical_transformed, X_without_categorical_transformed_pd], axis=1)