from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from StatkraftAssessment.utils.plot import plot_pacf_acf, plot_seasonality

@dataclass
class AnalysisOrchestrator:
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

        # Plot trend
        self.plot_trend(df)

        # PACF ACF target
        self.plot_pacf_acf_and_save(df, lags=365)
        self.plot_pacf_acf_and_save(df, lags=30)

        # Plot seasnality
        self.plot_seasonality(df)
        
        # Relation discharge/target
        self.plot_discharge_target(df)


        '''
        # Relation regressor/target
        lm = LinearRegression(fit_intercept=True)
        X_train, X_test, y_train, y_test = train_test_split(df[['total_load', 'total_wind', 'total_thermal', 'net_interchange']], df['total_hydro_columbia'], test_size=0.20, shuffle=False)
        lm.fit(X_train, y_train)
        df['lm_pred'] =  lm.predict(X=df[['total_load', 'total_wind', 'total_thermal', 'net_interchange']])

        plt.ion()
        df.loc[y_test.index, :].plot.scatter(x = 'lm_pred', y = 'total_hydro_columbia', alpha=.2, edgecolors='black', color='blue')

        plt.scatter(x = df[['total_load', 'net_interchange']].sum(axis = 1) - df[['total_wind', 'total_thermal']].sum(axis=1), y = df['total_hydro_columbia'], alpha=.2, edgecolors='black', color='blue')
        '''

    def plot_trend(self, df: pd.DataFrame):
        '''Plot trend of target'''

        plt.ion()
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['total_hydro_columbia'], color='blue', alpha=.5)
        plt.plot(df.index, df['total_hydro_columbia'].rolling(365, center=True).mean(), color='red', linewidth = 3, label='365 days rolling mean')
        plt.title('Columbia River Hydropower', fontweight='bold')
        plt.grid(':')
        plt.legend()

        plt.savefig(os.path.join(self.config['plot_path'] , f'trend_target.png'))
        plt.close()

    def plot_pacf_acf_and_save(self, df: pd.DataFrame, lags= 365):
        '''Plot target PACF and ACF'''
        plot_pacf_acf(df['total_hydro_columbia'], lags=lags)
        plt.savefig(os.path.join(self.config['plot_path'] , f'PACF_ACF_{lags}_target.png'))
        plt.close()

    def plot_discharge_target(self, df: pd.DataFrame):
        '''Plot relation between dellas discharge and columbia generation'''

        plt.ion()
        plt.figsize=(15, 10)
        g = sns.lmplot(x='discharge_dellas', y='total_hydro_columbia', data=df, scatter_kws={'alpha':.0}, order=3, hue='year_category')
        g._legend.remove()
        plt.scatter(x='discharge_dellas', y='total_hydro_columbia', data=df, alpha=.05, color = 'black')

        plt.title('Columbia generation vs discharge by year', fontweight = 'bold')
        plt.xlabel('Discharge Dellas')
        plt.ylabel('Generation')
        plt.grid(linestyle=':')
        plt.savefig(os.path.join(self.config['plot_path'], 'columbia_generation_vs_discharge.png'))
        plt.close()

    def plot_seasonality(self, df: pd.DataFrame):
        '''Plot target seasonality'''

        plot_seasonality(df = df, values='total_hydro_columbia', seasonality='Y', title='Columbia normalised generation seasonality year', index='dayofyear')
        plt.savefig(os.path.join(self.config['plot_path'], 'seasonality_Y.png'))
        plt.close()

        plot_seasonality(df = df, values='total_hydro_columbia', seasonality='M', title='Columbia normalised generation seasonality month', index='day')
        plt.savefig(os.path.join(self.config['plot_path'], 'seasonality_M.png'))
        plt.close()

        plot_seasonality(df = df, values='total_hydro_columbia', seasonality='W', title='Columbia normalised generation seasonality week', index='dayofweek')
        plt.savefig(os.path.join(self.config['plot_path'], 'seasonality_W.png'))
        plt.close()
