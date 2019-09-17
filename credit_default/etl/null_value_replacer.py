from sklearn.base import BaseEstimator, TransformerMixin

class NullValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, mode):
        assert mode in ["mean", "median"]
        self.mode = mode
        
    def fit(self, X, **fit_params):
        self.means = X.mean()
        self.medians = X.median()
        return self
    
    def transform(self, X, y=None, **transform_params):
        
        filled_in=X.fillna(self.means) if self.mode=="mean" else X.fillna(self.medians)
        return filled_in