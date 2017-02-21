from sklearn.base import TransformerMixin
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


class label_transformer(TransformerMixin):

    def transform(self, X, *_):
        X_1 = pd.DataFrame(X.copy())
        X_1 = X_1.apply(LabelEncoder().fit_transform)
        return X_1

    def fit(self, *_):
        return self

class ordinal_transformer(TransformerMixin):
    def __init__(self, ord_col_map):
        self.ord_col_map = ord_col_map

    def transform(self, X, *_):
        X_1 = X.copy()
        for col in X.columns:
            mp = self.ord_col_map[col]
            print "mp",mp
            X_1[col] = X_1[col].map(lambda x : x.lower() if (type(x) == str) else x)
            X_1[col] = X_1[col].map(lambda x : mp[x] if x in mp else float('nan'))
        return X_1

    def fit(self, *_):
        return self
