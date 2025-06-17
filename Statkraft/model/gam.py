from pygam import GAM, s, f, te
import matplotlib.pyplot as plt
from typing import Union, List
import pandas as pd

class myGAM(GAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def predict(self, X):
        '''
        Predict using the GAM model
        :param X: pandas dataframe with the same columns as the training data
        :return: pandas series with the predicted values
        '''
        gam_pred = super().predict(X)
        gam_pred = pd.Series(gam_pred, index=X.index, name='GAM')
        return gam_pred

    def plot_splines(self, spline_names: Union[None, List[str]] = None, title: str = 'Spline plot'):
        '''
        Plot the splines of the GAM model
        :param spline_names: list of spline names to plot. Must have same length as number of terms in the model
        '''

        plt.ion()
        fig, ax = plt.subplots(1, len(self.terms)-1, figsize=(15, 5))
        fig.suptitle(title, fontweight = 'bold')

        for i, term in enumerate(self.terms):
            if term.isintercept:
                continue

            XX = self.generate_X_grid(term=i)
            pdep, confi = self.partial_dependence(term=i, X=XX, width=0.95)

            ax[i].plot(XX[:, term.feature], pdep)
            ax[i].fill_between(x = XX[:, term.feature], y1 = confi[:, 0], y2 = confi[:, 1], alpha=0.1, color = 'r')
            ax[i].grid(linestyle = ':')

            if spline_names is not None:
                ax[i].set_title(spline_names[i])
            else:
                ax[i].set_title(repr(term))
            plt.show()


