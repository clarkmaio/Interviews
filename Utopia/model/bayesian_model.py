from dataclasses import dataclass
from typing import List, Union, Tuple, Dict, Any
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
matplotlib.use('TkAgg')

from abc import abstractmethod

from Utopia.utils.data_loader import DataLoader
from Utopia.config.config import Config
from Utopia.utils.scorer import crps

from sklearn.linear_model import PoissonRegressor

import pymc3 as pm
import arviz as az
from pygam import GAM, PoissonGAM, s, te, f, l




class myPoissonRegressor(PoissonRegressor):
    '''Just a simple extension of predict method to return pandas dataframe'''

    def __init__(self, alpha = 1., fit_intercept = True):
        super(myPoissonRegressor, self).__init__(alpha=alpha, fit_intercept=fit_intercept)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            y_pred = pd.DataFrame(super(myPoissonRegressor, self).predict(X), index=X.index, columns=['poisson'])
        else:
            y_pred = self.predict(X)
        return y_pred

class myPoissonGAM(PoissonGAM):

    def __init__(self, *args, **kwargs):
        super(myPoissonGAM, self).__init__(*args, **kwargs)


    def fit(self, X, y, exposure=None, weights=None):
        self.train_set = {'X': X, 'y': y}
        super(myPoissonGAM, self).fit(X=X, y=y, exposure=exposure, weights=weights)


    def predict(self, X, quantile: Union[List[float], None] = None, sample_kwargs: Dict = {}) -> Dict:
        '''
        Always generate mean prediction.
        If quantile is not None itmust be a list of float between 0. and 1.
        It this case return also quantiles obtained from posterior distributiomn sampling.

        ------------
        Useful sample_kwargs parameters:
        n_draws, n_bootstraps
        '''
        prediction_dict = {}

        # Mean prediction
        pred = super(myPoissonGAM, self).predict(X)
        prediction_dict['mean'] = pd.DataFrame(pred, columns=['pred_mean'], index=X.index)

        # Quantile prediction
        if quantile is not None:
            sample_pred = super(myPoissonGAM, self).sample(X=self.train_set['X'], y=self.train_set['y'],
                                                           sample_at_X=X, **sample_kwargs)
            prediction_dict['sample'] = pd.DataFrame(sample_pred.T, index=X.index)
            prediction_dict['quantile'] = prediction_dict['sample'].quantile(quantile, axis=1).T
        return prediction_dict


    def plot_splines(self, spline_idx: List[int], spline_label: List[str]):
        '''
        Generate one plot for each spline_idx
        :param spline_idx: list of spline idx to plot
        :param spline_label: list of names of splines
        '''

        assert len(spline_idx) == len(spline_label)

        fig, axs = plt.subplots(1, len(spline_idx))

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i, ax in zip(spline_idx, axs):
            XX = self.generate_X_grid(term=i)
            pdep, confi = self.partial_dependence(term=i, width=.95)

            ax.plot(XX[:, i], pdep)
            ax.plot(XX[:, i], confi, c='r', ls='--')
            ax.set_title(spline_label[i], fontweight='bold')



@dataclass
class BayesianModel:

    def __post_init__(self):
        self.pm_model = pm.Model()
        self.trace = None
        self.trace_train = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_kwargs = {}):
        '''
        Design pymc3 model.

        :param sample_kwargs: dictionary containing parameters will be passed to pm.sample.
                              Most important keys are 'draws', 'chains', 'cores'
                              Set 'return_inferencedata': True to avoid warning


        **************** NOTE!
        Make sure to define pm.Data('x', X) and use it when defining tje model.
        This is VERY IMPORTANT for prediction phase since the data container will be replaced with outofsample dataset.
        '''
        raise NotImplementedError('Missing fit method')

    @abstractmethod
    def predict(self, X: pd.DataFrame, samples: int = 500, mean: bool = False, quantile: Union[List[float], None] = None):
        '''
        Generate outofsample prediction sampling from posterior prediction

        :param samples: Numer of output will be extract for sample in X
        :param mean: True to return sample mean
        :param quantile: list of values between 0 and 1. If not None return a matrix containing quantile output
        '''

        with self.pm_model:

            # Update data container
            pm.set_data({'x': X})

            # Sample
            # ppc = pm.sample_posterior_predictive(self.trace, samples=samples)
            ppc = pm.fast_sample_posterior_predictive(self.trace, samples=samples)

        # Format output
        y_sample = pd.DataFrame(ppc['pred'].T, index=X.index)
        y_output = {'sample': y_sample}

        if mean:
            y_mean = y_sample.mean(axis=1).to_frame('mean')
            y_output['mean'] = y_mean

        if quantile is not None:
            y_quant = y_sample.quantile(q=quantile, axis=1).T
            y_output['quantile'] = y_quant

        return y_output

    def plot_trace(self):
        ax = az.plot_trace(self.trace_train)


@dataclass
class PoissonLinearRegression(BayesianModel):

    def __init__(self, fit_intercept: bool = True):
        super().__init__()
        self.fit_intercept = fit_intercept


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_kwargs: Any = {}):
        '''
        Initialize linear model using exponential link.
        Coefficient have Normal prior.

        ********* Model design
        y = Poisson(exp(alpha + beta * X))
        alpha, beta = N(0,10)

        ********* Parmas
        :param X: pandas DataFrame of reggressors
        :param y: target
        :param sample_kwargs: kwargs that wll be passed to sample function
        '''

        with self.pm_model:

            # Initialize data container
            X_data = pm.Data('x', X)
            y_data = pm.Data('y', y)

            # Initialize prior coefficients
            beta = pm.Normal('beta', mu=0, sigma=10, shape=len(X.columns))

            if self.fit_intercept:
                alpha = pm.Normal('alpha', mu=0, sigma=10)
                mu = pm.math.exp(alpha + X_data @ beta)
            else:
                mu = pm.math.exp(X_data @ beta)
                #u = pm.Deterministic('mu',  pm.math.exp(X_data @ beta))

            # Build likelihood and run sample
            likelihood = pm.Poisson('pred', mu=mu, observed=y_data)
            self.trace = pm.sample(**sample_kwargs)
            self.trace_train = deepcopy(self.trace)

        return self


class BayesianLinearRegression(BayesianModel):

    def __init__(self, fit_intercept: bool = True):
        super().__init__()
        self.fit_intercept = fit_intercept


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_kwargs={}) -> None:
        '''
        Initialize linear model. Coefficient have Normal prior.

        ********* Model deisgn
        y = Normal (alpha + beta * X, sigma)
        alpha, beta = N(0,10)
        sigma = HalfN(0,10)

        ********* Params
        :param X: pandas DataFrame of reggressors
        :param y: target
        :param sample_kwargs: kwargs that wll be passed to sample function
        '''

        with self.pm_model:

            # Initialize data container
            X_data = pm.Data('x', X)
            y_data = pm.Data('y', y)

            # Initialize prior coefficients
            beta = pm.Normal('beta', mu=0, sigma=10, shape=len(X.columns))
            sigma = pm.HalfNormal('sigma', sigma=10)

            if self.fit_intercept:
                alpha = pm.Normal('alpha', mu=0, sigma=10)
                mu = alpha + X_data @ beta
            else:
                mu = pm.Deterministic('mu', X_data @ beta)

            # Build likelihood and run sample
            likelihood = pm.Normal('pred', mu=mu, sigma=sigma, observed=y_data)
            self.trace = pm.sample(**sample_kwargs)
            self.trace_train = deepcopy(self.trace)

        return self


if __name__ == '__main__':


    # Data
    X = pd.DataFrame({'x1': np.linspace(0, 10, 1000)})
    # y = pd.DataFrame({'y': [np.random.poisson(lam=i**2 / 2.) for i in X['x1']]})
    y = pd.DataFrame({'y': [np.random.poisson(lam=i / 2.) for i in X['x1']]})
    X_train, X_test, y_train, y_test = X.iloc[:700, :], X.iloc[700:, :], y.iloc[:700, :], y.iloc[700:, :]
    quantile_list = np.linspace(.1, .9, 17)

    # Poisson linear
    poisson_linear = PoissonLinearRegression(fit_intercept=False)
    poisson_linear.fit(X=X_train, y=y_train['y'])
    poisson_prediction = poisson_linear.predict(X=X_test, samples=10000, quantile=quantile_list, mean=True)

    print(f'Score Poisson linear {crps(y_true=y_test.values.flatten(), y_pred=poisson_prediction["quantile"].values, return_mean=True)}')



