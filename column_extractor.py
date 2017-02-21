from sklearn.base import TransformerMixin
import pandas as pd

class column_extractor(TransformerMixin):
    def __init__(self, columns):
        self.columns = [col for col in columns]

    def transform(self, X, *_):
        date_frm = X[self.columns]
        return date_frm

    def fit(self, *_):
        return self
