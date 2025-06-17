from dataclasses import dataclass
from typing import List, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')

@dataclass
class ForecastSARIMAX:
    '''
    Class to generate price forecast using SARIMAX model.
    :param data: dataframe containing endog and exog columns
    :param endog: target column to forecast
    :param exog: exogenous regressors. Must be a list even if passing just one column
    '''
    data: pd.DataFrame
    endog: str
    exog: Union[List[str], None] = None

    def __post_init__(self):
        self.data_forecast = self.data.copy()
        self.all_columns = [self.endog] + (self.exog if self.exog is not None else [])

        self.data_forecast = self.data_forecast[self.all_columns]
        return

    def backtest(self, st_date: datetime, en_date: datetime,
                 leadtime: int  = 1,
                 window: int = 30, order: Tuple = (1, 1, 0), seasonal_order=None, trend=None,
                 ci_level: List[float] = [0.05, 0.20],
                 verbose: bool = False) -> pd.DataFrame:
        '''
        Generate backtest forecast for a given date range
        :param st_date: starting date
        :param en_date: ending date
        :param leadtime: number of step ahead to forecast
        :param window: train window
        :param order: ARIMA order
        :param seasonal_order: Seasonal order
        :param trend:
        :param ci_level: Confidence interval level
        :param verbose: True to print progress
        :return:
        '''

        print(' ---------------------------------------------------------------------- ')
        print(f'Generate backtest SARIMAX forecast {st_date} - {en_date}')
        print(' ---------------------------------------------------------------------- ')

        data_forecast_output = []

        date_range = pd.date_range(start=st_date, end=en_date, freq='D')
        for d in tqdm(date_range):
            # Compute forecast and store in self.data_forecast
            data_forecast_tmp = self._forecast_core(current_date=d, ci_level=ci_level, leadtime = leadtime, window=window, order=order, seasonal_order=seasonal_order, trend=trend, verbose=verbose)
            data_forecast_output.append(data_forecast_tmp)

        data_forecast_output = pd.concat(data_forecast_output, axis=0)


        '''
        # Plot check
        plt.ion()
        check_df = data_forecast_output.loc[:, ['forecast']]
        check_df = check_df.unstack('leadtime') 
        ax = check_df.plot()
        self.data_forecast.loc[check_df.index,  self.endog].plot(ax=ax, color='k', linewidth = 3)
        '''

        return data_forecast_output

    def _forecast_core(self, current_date: datetime, ci_level: List[float],window: int, leadtime :int = 1,order = None, seasonal_order = None, trend = None, verbose: bool = False) -> None:

        # Endog not available then skip
        if pd.isnull(self.data_forecast.loc[current_date, self.endog]):
            return

        if verbose:
            print(f'Process valuedate {current_date}...')
        # Compute forecast and store in self.data_forecast

        # Slice trainset
        start = current_date-timedelta(days=window)
        end = current_date-timedelta(days=1)
        data_tmp = self.data_forecast.loc[start:end, self.all_columns].dropna()
        endog_df = data_tmp.loc[start:end, self.endog].copy()

        # Prepare exog for fit
        if self.exog is not None:
            exog_df = data_tmp.loc[start:end, self.exog].copy()
        else:
            exog_df = None

        # Fit Model
        model = SARIMAX(endog=endog_df, exog=exog_df, order=order, trend=trend, seasonal_order=seasonal_order)
        result = model.fit(disp=False)

        # Prepare exog for forecast
        if self.exog is not None:
            exog_fcst = self.data_forecast.loc[current_date:(current_date+timedelta(days=leadtime-1)), self.exog].copy()
        else:
            exog_fcst = None


        # Generate prediction
        try:
            fcst_dict = {}
            for alpha in ci_level:
                fcst_dict[alpha] = result.get_forecast(steps=leadtime, exog=exog_fcst).summary_frame(alpha=alpha)
        except ValueError as e:
            print(e)
            return

        # Store results
        key = list(fcst_dict.keys())[0]
        index = pd.MultiIndex.from_arrays((pd.date_range(start=current_date, periods=leadtime, freq='D'), range(1, leadtime+1)), names=['valuedate', 'leadtime'])
        columns = ['forecast', 'forecast_se'] + [f'ci_lower_{alpha}' for alpha in ci_level] + [f'ci_upper_{alpha}' for alpha in ci_level]
        data_forecast_tmp = pd.DataFrame(index=index, columns = columns)
        data_forecast_tmp.loc[:, ['forecast', 'forecast_se']] = fcst_dict[key][['mean', 'mean_se']].values

        # Store quantile
        for alpha, fcst_tmp in fcst_dict.items():
            data_forecast_tmp.loc[:, [f'ci_lower_{alpha}', f'ci_upper_{alpha}']] = fcst_tmp[['mean_ci_lower', 'mean_ci_upper']].values

        return data_forecast_tmp