from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode
        
    def fit(self, X, y=None, **fit_params):
        return self
        
    def transform(self, X, y=None, **transform_params):
        categorical_pd = pd.DataFrame(data=X, columns=self.columns_to_encode)
        categorical_dummies = pd.get_dummies(categorical_pd, columns=self.columns_to_encode)
        return categorical_dummies