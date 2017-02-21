from sklearn.base import TransformerMixin
import pandas as pd


class ModelTransformer(TransformerMixin):
    def __init__(self, model) :
        self.model = model

    def transform(self, X, *_):
        return self.model.fit_transform(X)


    def fit(self, X, y =None):
        self.model.fit(X,y)
        return self
