from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from typing import Dict, List, Tuple, Any
from types import SimpleNamespace
import os

from Utopia.orchestrator.orchestrator import Orchestrator
from Utopia.utils.data_loader import DataLoader, DataContainer
from Utopia.utils.plot import plot_heatmap_correlation, plot_grid_pacf, plot_cross_correlation, plot_grid_acf
from Utopia.utils.data_utils import pivot_by_trackid



class AnalysisOrchestrator(Orchestrator):

    def __post_init__(self):
        return

    def run(self):
        '''
        Simple class to perform analysis and save plots
        '''

        # ------------------ Collect all data you need
        data_container = self.build_data_container()

        # ------------------- Save statistics
        output_path = os.path.join(self.config.output_path, 'data_describe.csv')
        daily_ts_describe = data_container.d_stack.describe().round(1)
        daily_ts_describe.to_csv(output_path)

        # ------------------- Generate plots
        self.generate_correlation_plots(data_container=data_container)
        self.generate_PACF_plots(data_container=data_container)
        self.generate_ACF_plots(data_container=data_container)
        self.generate_timeseries_plots(data_container=data_container)



    def build_data_container(self) -> SimpleNamespace:
        '''Just download all data and organize in a DataContainer class'''

        data_d = DataLoader(path=self.config.consumption_path, config=self.config).load(granularity='D')
        data_w = DataLoader(path=self.config.consumption_path, config=self.config).load(granularity='W', resample_operation='mean')
        data_m = DataLoader(path=self.config.consumption_path, config=self.config).load(granularity='MS', resample_operation='mean')
        data_station = DataLoader(path=self.config.consumption_path, config=self.config).load(granularity='D', aggregate_by_track=False)

        data_dict = {
            'data_station': data_station,
            'd_raw': data_d,
            'w_raw': data_w,
            'm_raw': data_m,

            'd_stack': pivot_by_trackid(data_d),
            'w_stack': pivot_by_trackid(data_w),
            'm_stack': pivot_by_trackid(data_m)

        }

        data_container = SimpleNamespace(**data_dict)
        # data_container = DataContainer(data_dict=data_dict)
        return data_container



    def generate_PACF_plots(self, data_container: Any) -> None:
        '''Plot all plots regarding PACF statistics'''

        self.log_headline('Generate PACF plots...')

        # PACF plot
        plot_grid_pacf(X=data_container.d_stack, save_path=os.path.join(self.config.plot_path, 'pacf_plot_d.png'), title='PACF daily')
        plot_grid_pacf(X=data_container.w_stack, save_path=os.path.join(self.config.plot_path, 'pacf_plot_w.png'), title='PACF weekly')
        plot_grid_pacf(X=data_container.m_stack, save_path=os.path.join(self.config.plot_path, 'pacf_plot_m.png'), title='PACF monthly', pacf_kargs={'lags': 6})


    def generate_ACF_plots(self, data_container: Any) -> None:
        '''Plot all plots regarding PACF statistics'''

        self.log_headline('Generate ACF plots...')

        # PACF plot
        plot_grid_acf(X=data_container.d_stack, save_path=os.path.join(self.config.plot_path, 'acf_plot_d.png'), title='ACF daily')
        plot_grid_acf(X=data_container.w_stack, save_path=os.path.join(self.config.plot_path, 'acf_plot_w.png'), title='ACF weekly')
        plot_grid_acf(X=data_container.m_stack, save_path=os.path.join(self.config.plot_path, 'acf_plot_m.png'), title='ACF monthly', acf_kargs={'lags': 6})


    def generate_correlation_plots(self, data_container: Any) -> None:
        '''Generate all plots about correlation'''

        self.log_headline('Generate correlation plots...')

        # Correlation heatpmaps
        plot_heatmap_correlation(X=data_container.m_stack, save_path=os.path.join(self.config.plot_path, 'correlation_heatmap_m.png'), title='Correlation matrix monthly', kwargs_heatmap={'cbar': False, 'annot': True})
        plot_heatmap_correlation(X=data_container.w_stack, save_path=os.path.join(self.config.plot_path, 'correlation_heatmap_w.png'), title='Correlation matrix weekly', kwargs_heatmap={'cbar': False, 'annot': True})
        plot_heatmap_correlation(X=data_container.d_stack, save_path=os.path.join(self.config.plot_path, 'correlation_heatmap_d.png'), title='Correlation matrix daily', kwargs_heatmap={'cbar': False, 'annot': True})

        # Cross correlation
        plot_cross_correlation(y1=data_container.d_stack['Track 0'], y2=data_container.w_stack['Track 2'], save_path=os.path.join(self.config.plot_path, 'cross_correlation_d_track_0_2.png'), title='Daily cross correlation Track 0 and 2')
        plot_cross_correlation(y1=data_container.d_stack['Track 2'], y2=data_container.w_stack['Track 4'], save_path=os.path.join(self.config.plot_path, 'cross_correlation_d_track_2_4.png'), title='Daily correlation Track 2 and 4')
        plot_cross_correlation(y1=data_container.d_stack['Track 0'], y2=data_container.w_stack['Track 4'], save_path=os.path.join(self.config.plot_path, 'cross_correlation_d_track_0_4.png'), title='Daily correlation Track 0 and 4')

        plot_cross_correlation(y1=data_container.w_stack['Track 0'], y2=data_container.w_stack['Track 2'], save_path=os.path.join(self.config.plot_path, 'cross_correlation_w_track_0_2.png'), title='Weekly cross correlation Track 0 and 2')
        plot_cross_correlation(y1=data_container.w_stack['Track 2'], y2=data_container.w_stack['Track 4'], save_path=os.path.join(self.config.plot_path, 'cross_correlation_w_track_2_4.png'), title='Weekly correlation Track 2 and 4')
        plot_cross_correlation(y1=data_container.w_stack['Track 0'], y2=data_container.w_stack['Track 4'], save_path=os.path.join(self.config.plot_path, 'cross_correlation_w_track_0_4.png'), title='Weekly correlation Track 0 and 4')

        # Correlation station ids
        for i in range(6):
            stack_df = pd.pivot_table(data_container.data_station.query(f'TrackId=={i}'), index='valuedate', columns=['StationId'], values='Plays').fillna(0)
            stack_corr = stack_df.corr()
            plot_heatmap_correlation(X=stack_corr, save_path=os.path.join(self.config.plot_path, f'correlation_station_track{i}.png'), title=f'Correlation matrix track {i}',
                                     kwargs_heatmap={'cbar': True, 'annot': False, 'linewidths': 0.})


    def generate_timeseries_plots(self, data_container: Any) -> None:
        '''Generate plot for each track with different granularity'''

        self.log_headline('Generate time series plots...')

        for track in data_container.d_stack.columns:
            plt.figure(figsize=(15, 5))
            plt.plot(data_container.d_stack.index, data_container.d_stack[track], label='Daily', color='black', alpha=.8)
            plt.plot(data_container.w_stack.index, data_container.w_stack[track], label='Weekly mean', color='red', linewidth=4)
            plt.plot(data_container.m_stack.index, data_container.m_stack[track], label='Monthly mean', color='green', linewidth=4)
            plt.grid(linestyle=':')
            plt.title(f'Timeseries {track}', fontweight='bold')

            plt.ylabel('Plays', fontweight='bold')
            plt.legend()

            plt.savefig(os.path.join(self.config.plot_path, f'{track}_timeseries.png'))
            plt.close()