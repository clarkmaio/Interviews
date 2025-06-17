

from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor
import patsy
import polars as pl
from typing import Union, Optional, Dict
from copy import deepcopy


class LinearSplineModel:
    _core = None
    _splines_design_info = None

    def __init__(self, formula: str, **linear_model_kwargs):
        self.formula = formula
        self._core = LinearRegression(**linear_model_kwargs)


    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        X_splines = patsy.dmatrix(self.formula, X.to_pandas())
        self._splines_design_info = X_splines.design_info
        self._core.fit(X_splines, y.to_pandas())

        
    def predict(self, X: pl.DataFrame, index: str = None) -> pl.Series:
        
        # Generate splines using information learned in training
        X_splines = patsy.build_design_matrices(
            [self._splines_design_info],
            X.to_pandas(),
        )[0]
        y_pred = self._core.predict(X_splines)
        y_pred = pl.Series(values=y_pred, name='pred')

        if index:
            y_pred = pl.concat([X.select(index), y_pred.to_frame()], how='horizontal')

        return y_pred
    

class PassiveAggressiveSplineModel(LinearSplineModel):
    def __init__(self, formula: str,  **linear_model_kwargs):
        super().__init__(formula)
        self._core = PassiveAggressiveRegressor(**linear_model_kwargs)
    
    def partial_fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        # Generate splines using information learned in training
        X_splines = patsy.build_design_matrices(
            [self._splines_design_info],
            X.to_pandas(),
        )[0]       

        self._core.partial_fit(X_splines, y.to_pandas())

    def online_predict(self, X: pl.DataFrame, y: pl.Series, index: str = None) -> pl.Series:
        """Generate rolling predictions using online learning logic."""

        # Prediction first sample
        y_pred = self.predict(X.head(1), index=index)

        # Online learning and prediction all other samples
        if len(X) == 1:
            return y_pred
        for idx_update, idx_test in zip(range(0, len(X)-1), range(1, len(X))):
            self.partial_fit(X[idx_update, :], pl.Series([y[idx_update]]))
            y_pred = pl.concat([y_pred, self.predict(X[idx_test, :], index=index)], how='vertical')
        
        return y_pred