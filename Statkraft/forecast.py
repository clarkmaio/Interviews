from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import os
from pygam import s, f

from StatkraftAssessment.model.SARIMAX import ForecastSARIMAX
from StatkraftAssessment.model.benchmark import HistoricalMean
from StatkraftAssessment.model.gam import myGAM
from StatkraftAssessment.utils.scorer import ModelsScorer
from xgboost import XGBRegressor

import plotly.express as px
from plotly.offline import plot


@dataclass
class ForecastOrchestrator:
    config: Dict


    def __post_init__(self):
        # Make sure output folder exist
        for p in ['plot_path', 'report_path']:
            if not os.path.exists(self.config[p]):
                os.mkdir(self.config[p])

    def run(self):
        # Load data
        df = pd.read_hdf('./data/processed_data.hdf', 'table')
        df = df.query('discharge_dellas>0').copy()


        # Prepare data
        X = df[df.columns[df.columns != 'total_hydro_columbia']]
        y = df['total_hydro_columbia']
        X_train, y_train, X_val, y_val, X_test, y_test = self.train_test_split(X, y)
        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)

        # Generate forecasts
        static_forecast, static_score = self.generate_static_forecast(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        autoregressive_forecast, autoregressive_score = self.generate_autoregressive_forecast(df=df)

        return

    def generate_static_forecast(self, X_train, y_train, X_test, y_test):
        '''Generate forecast of static models'''


        # ------------------ Static models ------------------

        # Benchmark
        hist_mean_mdl = HistoricalMean()
        hist_mean_mdl.fit(X=None, y=y_train)
        hist_mean_pred = hist_mean_mdl.predict(X_test)
        hist_mean_pred = hist_mean_pred.to_frame('HistMean')

        # Simple GAM
        gam_simple_mdl = myGAM(s(0, n_splines=30), distribution='normal', link='identity')
        gam_simple_mdl.fit(X_train[['discharge_dellas', 'weekday', 'dayofyear']], y_train)
        print(gam_simple_mdl.summary())
        gam_simple_pred = gam_simple_mdl.predict(X_test[['discharge_dellas', 'weekday', 'dayofyear']])
        gam_simple_pred = gam_simple_pred.to_frame('GAM_simple')


        # GAM
        gam_mdl = myGAM(s(0, n_splines=20) + f(1) + s(2, n_splines=10, basis='cp') + s(3, spline_order=3),
                        distribution='normal',
                        link='identity')
        gam_mdl.fit(X_train[['discharge_dellas', 'weekday', 'dayofyear', 'year']], y_train)
        print(gam_mdl.summary())
        gam_pred = gam_mdl.predict(X_test[['discharge_dellas', 'weekday', 'dayofyear', 'year']])
        gam_pred = gam_pred.to_frame('GAM')


        # BDT
        xgb_mdl = XGBRegressor()
        xgb_mdl.fit(X_train[['discharge_dellas', 'weekday', 'dayofyear', 'year']], y_train)
        xgb_pred = pd.DataFrame(xgb_mdl.predict(X_test[['discharge_dellas', 'weekday', 'dayofyear', 'year']]), index=X_test.index, columns=['BDT'])


        # Prepare output
        output = pd.concat([hist_mean_pred, gam_simple_pred, gam_pred, xgb_pred], axis=1)

        # Store fitted model
        self.model = {'hist_mean': hist_mean_mdl, 'gam_simple': gam_simple_mdl, 'gam': gam_mdl, 'xgb': xgb_mdl}

        # ------------------ Plot ------------------
        gam_mdl.plot_splines(spline_names=['discharge dellas', 'weekday', 'dayofyear', 'year'])
        plt.savefig(self.config['plot_path'] + 'splines_gam.png')
        plt.close()


        # ------------------ Score ------------------
        # Compute and save score
        score_data = pd.concat([y_test, output], axis=1).dropna()
        scorer = ModelsScorer(data=score_data, metrics=['mae', 'bias', 'mape', 'rmse', 'bias_percentage'])
        score_result = scorer.compute_score(target='total_hydro_columbia', models=['GAM', 'GAM_simple', 'BDT', 'HistMean'])
        score_result.to_excel(os.path.join(self.config['report_path'], 'static_score.xlsx'))

        fig = px.line(data_frame=score_data, x=score_data.index, y=['total_hydro_columbia', 'GAM', 'GAM_simple', 'BDT', 'HistMean'])
        plot(fig, filename=self.config['plot_path'] + 'static_forecast.html', auto_open=False)


        return output, score_result

    def generate_autoregressive_forecast(self, df, leadtime: int = 30):
        '''Generate forecast of autoregressive models'''

        df_ = df.resample('D').mean()

        # ------------------ Autoregressive models ------------------

        # Simple
        fcst_sarimax_mdl = ForecastSARIMAX(data=df_, endog='total_hydro_columbia')
        backtest_sarimax = fcst_sarimax_mdl.backtest(st_date=self.config['test_start'], en_date=self.config['test_end'], order=(2, 1, 1), seasonal_order=(1, 0, 1, 7), leadtime=leadtime, window = 60, trend=[1])
        sarimax_fcst = backtest_sarimax['forecast'].to_frame('SARIMAX')

        # Autoregreessive applied to GAM bias
        gam_pred = self.model['gam'].predict(df[['discharge_dellas', 'weekday', 'dayofyear', 'year']])
        gam_adj_df = pd.concat([df_[['total_hydro_columbia']], gam_pred], axis=1).dropna()
        gam_adj_df['bias'] = gam_adj_df['total_hydro_columbia'] - gam_adj_df['GAM']
        gam_adj_df = gam_adj_df.resample('D').mean()
        gam_adjustement = ForecastSARIMAX(data=gam_adj_df, endog='bias')
        backtest_sarimax_gam_adj = gam_adjustement.backtest(st_date=self.config['test_start'], en_date=self.config['test_end'], order=(1, 0, 1), seasonal_order=(1, 0, 1, 7), leadtime=leadtime, window = 60)

        sarimax_gam_bias = backtest_sarimax_gam_adj['forecast'].to_frame('SARIMAX_gam_bias')
        sarimax_gam_bias = pd.merge(sarimax_gam_bias.reset_index(), gam_pred.reset_index(), on = 'valuedate', how='left').dropna()
        sarimax_gam_bias['SARIMAX_gam_adj'] = sarimax_gam_bias['SARIMAX_gam_bias'] + sarimax_gam_bias['GAM']
        sarimax_gam_bias = sarimax_gam_bias.set_index(['valuedate', 'leadtime'])

        # Prepare output
        output = pd.concat([sarimax_gam_bias, sarimax_fcst], axis=1)


        # ------------------ Score ------------------
        # Compute and save autoregressive score
        score_data_leadtime = output.reset_index()
        score_data_leadtime = pd.merge(score_data_leadtime, df.reset_index()[['valuedate', 'total_hydro_columbia']], on='valuedate', how='left').dropna()
        scorer_leadtime = ModelsScorer(data = score_data_leadtime, metrics=['mae', 'bias', 'mape', 'rmse'])
        score_result_leadtime = scorer_leadtime.compute_score(target = 'total_hydro_columbia', models = ['SARIMAX', 'SARIMAX_gam_adj', 'GAM'], by='leadtime')
        score_result_leadtime.to_excel(os.path.join(self.config['report_path'], 'autoregressive_score.xlsx'))

        return output, score_result_leadtime

    def train_test_split(self, X, y):
        val_start = self.config['val_start']
        test_start = self.config['test_start']
        test_end = self.config['test_end']

        X_train = X.loc[:val_start]
        y_train = y.loc[:val_start]

        X_val = X.loc[val_start:test_start]
        y_val = y.loc[val_start:test_start]

        X_test = X.loc[test_start:test_end]
        y_test = y.loc[test_start:test_end]
        return X_train, y_train, X_val, y_val, X_test, y_test