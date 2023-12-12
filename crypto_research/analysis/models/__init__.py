import abc
import pathlib

import joblib
import pandas as pd
from catboost import CatBoostRegressor as _CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
from sklearn.linear_model import Lasso

# Todo: implement grid search for regressor to find best params.
# Todo: should be refactored
class LassoRegressor(Lasso):
    def fit(self, X, y, **kwargs):
        # It does not look good to lasso, but use it just for PoC.
        X = X.ffill().fillna(0)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = X.ffill().fillna(0)
        result = super().predict(X)
        return pd.Series(index=X.index, data=result)

class RandomForestRegressor(_RandomForestRegressor):
    def fit(self, X, y, **kwargs):
        # It does not look good to lasso, but use it just for PoC.
        X = X.ffill().fillna(0)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = X.ffill().fillna(0)
        result = super().predict(X)
        return pd.Series(index=X.index, data=result)


class CatBoostRegressor(_CatBoostRegressor):
    def predict(self, X, **kwargs):
        X = X.ffill().fillna(0)
        result = super().predict(X)
        return pd.Series(index=X.index, data=result)

# Todo: at least in current system it is redundant to have own model class.
# class Model:
#     """Model is used as a unified approach to train various machine learning algorithms on data."""
#     def __init__(self, *args, **kwargs):
#         self.args = args
#         self.kwargs = kwargs
#
#     @abc.abstractmethod
#     def fit(self, X, y):
#         pass
#
#     @abc.abstractmethod
#     def predict(self, X):
#         pass
#
#     def save(self, path: pathlib.Path) -> None:
#         """
#         Save the trained model to the specified path.
#
#         Args:
#         path (pathlib.Path): Path to save the model.
#         """
#         joblib.dump(self.model, path)
#
#     def load(self, path: pathlib.Path) -> None:
#         """
#         Load the model from the specified path.
#
#         Args:
#         path (pathlib.Path): Path to load the model from.
#         """
#         self.model = joblib.load(path)