from dataclasses import dataclass
import pandas as pd
from typing import Dict, Union, Tuple, List, Any
import pandas as pd
from datetime import datetime
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

@dataclass
class BackTestARIMA:
    '''
    Class to generate ARIMA prediction with rolling fit/predict methodology.
    At each timestamp fit the model and predict n_periods step ahead

    Very usefull class to perform backtest with Autoregressive models

    -------------------- Example
    bac

    '''

    order: Any
    trend: str = 'c'
    n_periods: int = 10
    train_step_size: Union[int, None] = None
    use_exog: bool = False

    def fit(self, y: pd.Series, test_start: datetime, X: Union[pd.DataFrame, None] = None) -> None:
        '''
        Just store dataset.
        Real fit will be done rolling in predict method
        '''
        self.X = X
        self.y = y
        self.y_test = self.y.loc[test_start:]
        self.test_start = test_start

        # List where all prediction will be stored
        self.prediction_list = []


    def predict(self, X = None, y = None) -> pd.DataFrame:
        '''
        Loop over test set.
        At each time stamp:
            1. Slice rolling training set
            2. Fit arima
            3. Predict n_periods steps ahead
            4. Store result and iterate to next timestamp

        :param X, y: useless. Just for compatibility with sklearn syntax
        '''

        for d in self.y_test.index:

            # Generate forecasts
            y_pred_tmp = self.fit_predict(date_start=d, X=self.X, y=self.y)
            y_pred_tmp = self.postprocess_prediciton(y_pred=y_pred_tmp)

            # Concat to pred list
            self.prediction_list.append(y_pred_tmp)

            '''
            # Check plot
            check_df = pd.concat([self.y.to_frame('actual'), y_pred_tmp[['forecast']]], axis=1)
            check_df.plot()
            '''

        # Concat and pivot
        self.prediction = pd.concat(self.prediction_list, axis=0)
        self.prediction.reset_index(inplace=True)
        prediction_pivot = pd.pivot_table(data=self.prediction, index=['valuedate'], columns=['leadtime'], values=['forecast'])
        prediction_pivot.columns = prediction_pivot.columns.droplevel(0)

        '''
        # Check plot
        check_df = pd.concat([prediction_pivot, self.y.to_frame('target')], axis=1)
        check_df.plot()
        '''

        return prediction_pivot


    def postprocess_prediciton(self, y_pred):
        '''
        Format prediction output and assign leadtime variable
        '''

        # Assign leadtime value to forecast
        y_pred = y_pred.to_frame('forecast')
        y_pred['leadtime'] = range(1, self.n_periods + 1)
        y_pred.index.name = 'valuedate'
        return y_pred


    def fit_predict(self, date_start, y, X) -> pd.Series:
        '''Fit predict according use_exog param'''

        if self.use_exog:
            X_train, X_test, y_train = self.slice_train_exog(timestamp=date_start, y=y, X=X, train_step_size=self.train_step_size)
            y_pred = self._fit_predict_exog(X_train=X_train, X_test=X_test, y_train=y_train)
        else:
            y_train = self.slice_train_endog(timestamp=date_start, y=y, train_step_size=self.train_step_size)
            y_pred = self._fit_predict_endog(y=y_train)

        return y_pred

    def _fit_predict_endog(self, y: pd.Series):
        model = ARIMA(endog=y, order=self.order, trend=self.trend)
        model_fitted = model.fit()

        start = min([self.train_step_size, len(y)]) # Define forecast starting point
        y_pred_tmp = model_fitted.predict(start=start, end=start+self.n_periods-1, dynamic=True)
        return y_pred_tmp

    def _fit_predict_exog(self, X_train, X_test, y_train):
        model = ARIMA(endog=y_train, exog=X_train, order=self.order, trend=self.trend)
        model_fitted = model.fit()

        start = min([self.train_step_size, len(y)]) # Define forecast starting point
        y_pred_tmp = model_fitted.predict(exog=X_test, start=start, end=start+self.n_periods-1, dynamic=True)
        return y_pred_tmp

    def slice_train_endog(self, timestamp: datetime, train_step_size: int, y: pd.Series):
        '''
        Just slice data from (timestamp-train_step_size) to timestamp
        '''

        y_train_slice = y.loc[:timestamp]

        if train_step_size is not None:
            # Slice according to train_step_size
            y_train_slice = y_train_slice.iloc[-train_step_size-1:-1]
        else:
            # If None use the whole (just exclude last value)
            y_train_slice = y_train_slice.iloc[:-1]
        return y_train_slice


    def slice_train_exog(self, timestamp: datetime, train_step_size: int,  X: pd.DataFrame, y: pd.Series):
        '''
        Just slice data from (timestamp-train_step_size) to timestamp.
        Return both train and test exogenous data
        '''

        # Slice trainset
        y_train_slice = y.loc[:timestamp]
        X_train_slice = X.loc[:timestamp]

        if train_step_size is not None:
            # Slice according to train_step_size
            y_train_slice = y_train_slice.iloc[-train_step_size-1:-1]
            X_train_slice = X_train_slice.iloc[-train_step_size-1:-1]
        else:
            # If None use the whole (just exclude last value)
            y_train_slice = y_train_slice.iloc[:-1]
            X_train_slice = X_train_slice.iloc[:-1]


        # Slice test set
        X_test_slice = X.loc[timestamp:]
        X_test_slice = X_test_slice.iloc[:self.n_periods, :]

        return X_train_slice, X_test_slice, y_train_slice

