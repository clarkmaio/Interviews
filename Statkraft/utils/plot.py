import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

def plot_pacf_acf(v, lags=100, title = 'PACF ACF'):
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title)
    plot_acf(v, lags=lags, ax=ax[0])
    plot_pacf(v, lags=lags, ax=ax[1])
    ax[0].grid(linestyle=':')
    ax[1].grid(linestyle=':')


def plot_seasonality(df: pd.DataFrame, values, seasonality: str, title='Seasonality', index: str = 'day'):
    '''
    Plot seasonality

    :param df:
    :param values:
    :param seasonality:
    :param title:
    :return:
    '''

    plt.ion()
    df_plot = df.copy()
    df_plot['first_day_{}_date'.format(seasonality)] = df_plot.index.to_period(seasonality).to_timestamp(seasonality)
    pivot = pd.pivot_table(df_plot, columns='first_day_{}_date'.format(seasonality), values=values, aggfunc=np.mean, index=index)
    pivot = pivot / pivot.mean()
    avg = pivot.mean(axis=1)
    percentile = pivot.quantile([0.05, 0.95], axis=1).T

    ax = pivot.plot(color='k', alpha=0.05, legend=False)
    avg.plot(ax=ax, color='r', linewidth=3)
    ax.fill_between(percentile.index, percentile[0.05], percentile[0.95], color='r', alpha=0.2)
    ax.grid(linestyle=':')
    ax.set_title(title, fontweight = 'bold')