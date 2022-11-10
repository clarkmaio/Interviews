import os
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from typing import Dict, List, Tuple, Union
import plotly_express as px

from Utopia.orchestrator.orchestrator import Orchestrator
from Utopia.utils.data_loader import DataLoader
from Utopia.utils.data_utils import pivot_by_trackid
from Utopia.utils.scorer import Scorer, crps

from Utopia.model.autoregressive_model import BackTestARIMA
from Utopia.model.bayesian_model import myPoissonRegressor, myPoissonGAM
import Utopia.utils.plot as plot

from pmdarima import auto_arima, arima
from pygam import s, l, f



class ForecastOrchestrator(Orchestrator):

    def __post_init__(self):
        super().__post_init__()

        # Setup forecast settings
        self.forecast_settings = self.config['forecast_setting']
        self.forecast_settings['calendar_features'] = self.forecast_settings[self.forecast_settings['granularity']]['calendar_features']
        self.forecast_settings['step_ahead'] = self.forecast_settings[self.forecast_settings['granularity']]['step_ahead']
        self.forecast_settings['autoregressive_train_step_size'] = self.forecast_settings[self.forecast_settings['granularity']]['autoregressive_train_step_size']

        self.score_dict = {}

    def run(self):

        # Load data
        data = DataLoader(path=self.config.consumption_path, config=self.config).load(granularity=self.forecast_settings['granularity'], resample_operation=self.forecast_settings['resample_operation'])

        for track_id in self.forecast_settings['track_id_list']:
            self.log_headline(f'Processing track {track_id}')
            self.run_forecast_track(track_id=track_id, data=data)

        # Concat result and save
        score_path = os.path.join(self.config.score_path,  f'score_{self.forecast_settings["granularity"]}.xlsx')
        self.log(f'Saving {score_path}')
        score_output = pd.concat(self.score_dict, axis=0, names=['TrackId']).round(2)
        score_output.to_excel(score_path, engine='openpyxl')



    def run_forecast_track(self, track_id: int, data: pd.DataFrame):
        '''
        Generate forecast for a given track.
        Store forecast and compute
        '''


        # Split data
        X, X_train, X_test, y, y_train, y_test = self.train_test_split(track_id=track_id, data=data, split_date=self.config['date']['test_st'])

        # ---------------- Initialize models ----------------
        # Poisson model
        poisson_model = myPoissonRegressor(fit_intercept=True)

        # Simple AR model
        ar_model = BackTestARIMA(order=(1,0,0), trend='t', n_periods=self.forecast_settings['step_ahead'], train_step_size=self.forecast_settings['autoregressive_train_step_size'], use_exog=False)

        # Optimised ARIMA model
        opt_order = auto_arima(y=y_train, seasonal=True).order
        arima_model = BackTestARIMA(order=opt_order, trend='t', n_periods=self.forecast_settings['step_ahead'], train_step_size=self.forecast_settings['autoregressive_train_step_size'], use_exog=False)
        # ---------------------------------------------------



        # Organize model in ModelContainer
        model_container = ModelContainer()\
            .add_model(model_name='poisson', model=poisson_model,
                       train_set={'X': X_train[self.forecast_settings['calendar_features']], 'y': y_train},
                       test_set=X_test[self.forecast_settings['calendar_features']])\
            .add_model(model_name='ar', model=ar_model,
                       train_set={'X': X[self.forecast_settings['calendar_features']], 'y': y},
                       test_set=y_test,
                       fit_kwargs={'test_start': self.config['date']['test_st']})\
            .add_model(model_name='arima', model=arima_model,
                       train_set={'X': X[self.forecast_settings['calendar_features']], 'y': y},
                       test_set=y_test,
                       fit_kwargs={'test_start': self.config['date']['test_st']})

        # Generate forecast
        model_container.fit_predict()



        # Save interactive plots
        plot_df = pd.concat([model_container.predictions['arima'], y_test.to_frame('Plays'), model_container.predictions['poisson']], axis=1)
        plot_df.columns = [f'ARIMA lag {i}' if isinstance(i, int) else i for i in plot_df.columns]
        fig = px.line(plot_df[['ARIMA lag 1', 'ARIMA lag 2', 'poisson', 'Plays']],
                      title=f'Forecast Track {track_id}',
                      markers=True, color_discrete_map={'Plays': 'red', 'poisson': 'black', 'ARIMA lag 1': 'blue', 'ARIMA lag 2': 'lightskyblue'})
        fig.write_html(os.path.join(self.config.plot_html_path, f'forecast_track_{track_id}_{self.forecast_settings["granularity"]}.html'))


        # Format static model with the same structure of ar mdoels
        model_container.predictions['poisson'] = pd.concat([model_container.predictions['poisson']] * self.forecast_settings['step_ahead'], axis=1)
        model_container.predictions['poisson'].columns = pd.Index(range(1, self.forecast_settings['step_ahead'] + 1), name='leadtime')


        # Compute score
        score_table = self.compute_score(model_container=model_container, y_test=y_test)
        self.score_dict[track_id] = score_table

        score_plot_path = os.path.join(self.config.plot_path, f'score_track_{track_id}_{self.forecast_settings["granularity"]}.png')
        self.log(f'Saving plot {score_plot_path}')
        plot.plot_leadtime_score_curve(score_table=score_table.loc[:, (slice(None), ['mae', 'bias'])], title=f'Score track {track_id} {self.forecast_settings["granularity"]}',
                                       save_path=score_plot_path)



    def compute_score(self, model_container, y_test) -> Dict:

        scorer = Scorer(metrics=('mae', 'bias', 'count'))
        score_dict = {}
        for mdl in model_container.model_list:
            forecast = model_container.predictions[mdl]
            data_score = pd.concat([y_test.to_frame('target'), forecast], axis=1).dropna()
            score_df = scorer.compute_score(data=data_score, target='target', model_list=forecast.columns)
            score_dict[mdl] = score_df
        score_table = pd.concat(score_dict, axis=1, names=['model'])
        score_table.index.names = ['lead_time']

        return score_table



    def train_test_split(self, track_id: int, split_date: datetime, data: pd.DataFrame):
        '''Split data into features/target and train/test set'''

        data_pivot = pivot_by_trackid(df=data)
        y = data_pivot[f'Track {track_id}']
        X = data.query(f'TrackId == {track_id}').set_index('valuedate')
        y_train, y_test = y.loc[:split_date], y.loc[split_date:]
        X_train, X_test = X.loc[:split_date], X.loc[split_date:]

        return X, X_train, X_test, y, y_train, y_test



@dataclass
class BayesianForecastOrchestrator(ForecastOrchestrator):

    def __post_init__(self):
        super(BayesianForecastOrchestrator, self).__post_init__()

        self.log('Bayesian mode works only with granularity W')
        self.forecast_settings['granularity'] = 'W'
        self.forecast_settings['track_id_list'] = [0,1,2,3,4]


    def run(self):

        # Load data
        data = DataLoader(path=self.config.consumption_path, config=self.config).load(granularity=self.forecast_settings['granularity'], resample_operation=self.forecast_settings['resample_operation'])

        for track_id in self.forecast_settings['track_id_list']:
            self.log_headline(f'Processing track {track_id}')
            self.run_forecast_track(track_id=track_id, data=data)


    def run_forecast_track(self, track_id: int, data: pd.DataFrame):
        '''
        Generate forecast for a given track.
        Store forecast and compute
        '''

        # ---------- Hyper parameters
        quantile_list = np.linspace(.1, .9, 17)
        shift_target = 1

        features, formula = self.formula_builder(granularity=self.forecast_settings['granularity'])

        # ---------- Split data
        X, X_train, X_test, y, y_train, y_test = self.train_test_split(track_id=track_id, data=data, split_date=self.config['date']['test_st'])

        # Preprocess data
        X_gam, y_gam = self.preprocess_data(shift=shift_target, features=features, X=X, y=y)
        X_gam_train, y_gam_train = self.preprocess_data(shift=shift_target, features=features, X=X_train, y=y_train)
        X_gam_test, y_gam_test = self.preprocess_data(shift=shift_target, features=features, X=X_test, y=y_test)

        # ---------- Initialize model and fit
        gam_model = myPoissonGAM(formula, fit_intercept=False)
        gam_model.fit(X_gam_train[features], y_gam_train)

        # ---------- Generate predictions
        gam_prediction = gam_model.predict(X_gam[features], quantile=quantile_list, sample_kwargs={'n_draws': 10000, 'n_bootstraps': 5})

        # ---------- Save plots
        self.generate_plots(track_id=track_id, granularity=self.forecast_settings['granularity'], gam_model=gam_model, gam_prediction=gam_prediction, y=y_gam)




    def preprocess_data(self, shift: int, features: List[str], X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Simple preprocess to prepare data'''
        X['Plays_shift'] = X['Plays'].shift(shift).dropna()
        X.dropna(inplace=True)
        y = y.loc[X.index]

        return X[features], y

    def formula_builder(self, granularity: str):
        '''
        Assign features list and gam formula structure depending on granulraity
        '''

        if granularity == 'W':
            features = ['week', 'Plays_shift']
            formula = s(0, basis='cp', n_splines=5) + s(1, n_splines=3, spline_order=2)

        elif granularity == 'D':
            features = ['day_idx', 'doy', 'weekday', 'Plays_shift']
            formula = s(1, basis='cp', n_splines=5) + s(2, basis='cp', n_splines=5) + s(2, n_splines=3, spline_order=2)

        elif granularity == 'MS':
            features = ['day_idx', 'month', 'Plays_shift']
            formula = s(1, basis='cp', n_splines=5) + s(2, basis='cp', n_splines=5)

        return features, formula


    def generate_plots(self, track_id: int, granularity: str, gam_model, gam_prediction: Dict, y) -> None:
        '''
        Create spline splot and forecast timeseries plot
        '''

        gam_mean_pred = gam_prediction['mean']
        gam_quant_pred = gam_prediction['quantile']

        # Plots splines
        gam_model.plot_splines(spline_idx=[0, 1], spline_label=['week', 'last value'])
        plt.suptitle(f'Track {track_id} GAM splines', fontweight='bold')
        plt.savefig(os.path.join(self.config.bayesian_plot_path, f'gam_splines_track_{track_id}_{granularity}.png'))
        plt.close()

        # Plot timeseries
        plt.figure(figsize=(8, 5))
        plt.plot(y.index, y, color='red', label='Target')
        plt.plot(gam_mean_pred.index, gam_mean_pred['pred_mean'], color='black', label='GAM Mean')
        plt.fill_between(x=gam_quant_pred.index, y1=gam_quant_pred[0.1], y2=gam_quant_pred[0.9], color='gray', alpha=.1, label='80% pred interval')
        plt.fill_between(x=gam_quant_pred.index, y1=gam_quant_pred[0.25], y2=gam_quant_pred[0.75], color='blue', alpha=.1, label='50% pred interval')
        plt.axvspan(xmin=self.config['date']['test_st'], xmax=self.config['date']['test_st']+timedelta(days=1), color='green', label='Test start', alpha=1.)
        plt.legend()
        plt.grid(linestyle=':')
        plt.title(f'Bayesian predictions Track {track_id}', fontweight='bold')
        plt.ylabel('Plays', fontweight='bold')
        plt.savefig(os.path.join(self.config.bayesian_plot_path, f'bayesian_pred_track_{track_id}_{granularity}.png'))
        plt.close()



@dataclass
class ModelContainer:
    '''
    Class to manage fit/predict multiple models at once.
    You can add a model using add_model method.

    The method fit_predict will fit and generate prediction on the test set as specified in the
    The prediction will be accessible using 'predictions' class properties

    NOTE: models MUST have fit, predict syntax
    '''

    def __post_init__(self):
        self.model_list = []
        self.model_dict = {}
        self._predictions = {}

    def add_model(self, model_name: str, model, train_set: Dict, test_set: pd.DataFrame, fit_kwargs: Dict = {}):
        self.model_list.append(model_name)
        self.model_dict[model_name] = {'model': model, 'train_set': train_set, 'test_set': test_set, 'fit_kwargs': fit_kwargs}
        return self

    def fit_predict(self):
        '''
        For each model initialised fit on train set and predict on test set.
        Predictions will be stored in self.prediction dictionary.
        '''

        for mdl_name, mdl_dict in self.model_dict.items():
            print(f'> Processing {mdl_name}...')

            # Fit
            X = mdl_dict['train_set']['X']
            y = mdl_dict['train_set']['y']
            mdl_dict['model'].fit(X=X, y=y, **mdl_dict['fit_kwargs'])

            # Predict
            pred_tmp = mdl_dict['model'].predict(X=mdl_dict['test_set'])
            self._predictions[mdl_name] = pred_tmp

    @property
    def predictions(self):
        return self._predictions

