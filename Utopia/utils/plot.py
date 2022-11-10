import matplotlib.pyplot as plt
from typing import List, Union, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import os
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statsmodels.api as sm

import plotly_express as px

sns.set_theme(style="ticks")
matplotlib.use('TkAgg')


def plot_gam_spline(gam_model, spline_idx: List[int], spline_title: Union[None, List], save_path: str,
                    title: str = 'Splines') -> None:
    '''Plot spline partial dependece of GAM models'''

    assert len(spline_idx) == len(spline_title)

    fig, axs = plt.subplots(1, len(spline_idx))

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, ax in zip(spline_idx, axs):
        XX = gam_model.generate_X_grid(term=i)
        pdep, confi = gam_model.partial_dependence(term=i, width=.95)

        ax.plot(XX[:, i], pdep)
        ax.plot(XX[:, i], confi, c='r', ls='--')
        ax.set_title(spline_title[i])

    plt.title(title, fontweight='bold')
    plt.savefig(save_path)
    plt.close()

def plot_heatmap_correlation(X: pd.DataFrame, save_path: str, title = 'Correlation matrix', kwargs_heatmap: Dict = {}) -> None:
    '''
    Plot correlation matrix as heatmap.
    Correlation computed among X columns
    '''

    # Compute correlation
    corr_matrix = X.corr()

    # Build heatmap default kwargs an update
    kwargs_heatmap_default = {'square': True, 'linewidths': .5, 'annot': True, 'annot_kws': {"size": 8}, 'cbar': False}
    kwargs_heatmap_default.update(kwargs_heatmap)

    # Generate figure
    fig = plt.figure(figsize=(10, 10))
    plt.title(title, fontweight = 'bold')
    sns.heatmap(corr_matrix.round(2), **kwargs_heatmap_default)
    plt.savefig(save_path)
    plt.close()


def plot_cross_correlation(y1, y2, save_path: str, max_lag: int = 50, title: str = 'Cross correlation') -> None:
    '''
    Plot cross correlation between two timeseries
    '''

    # Compute cross correlation
    cross_correlation = [y1.corr(y2.shift(i)) for i in range(max_lag)]
    xaxis = range(len(cross_correlation))

    # Build plot
    fig = plt.figure(figsize=(10, 5))
    plt.stem(range(len(cross_correlation)), cross_correlation, linefmt='k-', markerfmt='ko')
    plt.hlines(xmin=xaxis[0]-1, xmax=xaxis[-1]+1, y=.5, color='red', linestyles='dotted')
    plt.hlines(xmin=xaxis[0]-1, xmax=xaxis[-1]+1, y=-.5, color='red', linestyles='dotted')
    plt.hlines(xmin=xaxis[0]-1, xmax=xaxis[-1]+1, y=0, color='black', linestyles='solid')
    plt.title(title, fontweight='bold')
    plt.ylabel('Correlation', fontweight='bold')
    plt.xlabel('Time lag', fontweight='bold')
    plt.grid(linestyle=':')

    plt.savefig(save_path)
    plt.close()


def plot_grid_pacf(X: pd.DataFrame, save_path: str, title: str = 'PACF', pacf_kargs: Dict = {}) -> None:
    '''
    Plot PACF   for each X column
    '''

    # Deduce needed nrows
    nrows = int(np.ceil(len(X.columns)/2.))

    # Build grid plot
    fig, ax = plt.subplots(nrows, 2, figsize = (10, 8))

    ax_list = ax.flatten()
    for i, col in enumerate(X.columns):
        plot_pacf(x=X[col], ax=ax_list[i], **pacf_kargs)
        ax_list[i].set_title(f'{col}', fontweight='bold')
        ax_list[i].grid(linestyle=':')

        # Remove ticks from upper plots
        if i < nrows * 2 - 2:
            ax_list[i].tick_params(bottom=False, labelbottom=False)
    fig.suptitle(title, fontweight = 'bold')

    plt.savefig(save_path)
    plt.close()




def plot_grid_acf(X: pd.DataFrame, save_path: str, title: str = 'ACF', acf_kargs: Dict = {}) -> None:
    '''
    Plot ACF for each X column
    '''

    # Deduce needed nrows
    nrows = int(np.ceil(len(X.columns)/2.))

    # Build grid plot
    fig, ax = plt.subplots(nrows, 2, figsize = (10, 8))

    ax_list = ax.flatten()
    for i, col in enumerate(X.columns):
        plot_acf(x=X[col], ax=ax_list[i], **acf_kargs)
        ax_list[i].set_title(f'{col}', fontweight='bold')
        ax_list[i].grid(linestyle=':')

        # Remove ticks from upper plots
        if i < nrows * 2 - 2:
            ax_list[i].tick_params(bottom=False, labelbottom=False)
    fig.suptitle(title, fontweight = 'bold')

    plt.savefig(save_path)
    plt.close()



def plot_leadtime_score_curve(score_table: pd.DataFrame, save_path: str, title: str = 'Score') -> None:

    model_list = score_table.columns.get_level_values('model').unique()
    metric_list = score_table.columns.get_level_values('metric').unique()

    fig, ax = plt.subplots(len(metric_list), 1, sharex=True, figsize = (7, 5))
    fig.suptitle(title, fontweight='bold')
    for i, metric in enumerate(metric_list):
        metric_df = score_table.xs(metric, axis=1, level='metric')

        for mdl in model_list:
            ax[i].plot(metric_df.index, metric_df[mdl], marker='.', label=mdl)

        ax[i].set_ylabel(metric.upper(), fontweight='bold')
        ax[i].grid(linestyle=':')
        ax[i].legend()
    ax[i].set_xlabel('Lead time',  fontweight='bold')

    plt.savefig(save_path)
    plt.close()
    return


