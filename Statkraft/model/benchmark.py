from dataclasses import dataclass
import pandas as pd
from scipy.ndimage import gaussian_filter1d



@dataclass
class HistoricalMean:

    def fit(self, y: pd.Series = None, X = None):
        '''
        Store target mean for each day of year
        '''
        self.y_train = y.copy()
        self.y_train = self.y_train.to_frame('HistMean')
        self.y_train['dayofyear'] = self.y_train.index.dayofyear
        self.y_train = self.y_train.groupby('dayofyear').mean()
        self.y_train = self.y_train.reset_index()

        # Filter
        self.y_train['HistMeanSmooth'] = gaussian_filter1d(self.y_train['HistMean'], sigma=10)

    def predict(self, X = None):
        X_test = X.copy()
        X_test['dayofyear'] = X_test.index.dayofyear
        X_test = pd.merge(X_test, self.y_train, on = 'dayofyear', how = 'left')
        X_test.index = X.index
        return X_test['HistMeanSmooth']
