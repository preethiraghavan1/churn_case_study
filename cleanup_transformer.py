from sklearn.base import TransformerMixin
import pandas as pd

class cleanup_transformer(TransformerMixin):

    def transform(self, X, *_):
        df = pd.DataFrame(X.copy())
        df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)
        return df[[col for col in df.columns if (len(df[col].unique()) > 1)]]

    def fit(self, *_):
        return self
