from dataclasses import dataclass
from typing import List, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
import CRPS.CRPS as pscore



def crps(y_true, y_pred, return_mean: bool = True) -> np.ndarray:
    '''
    Compute crps, fcrps, acrps for each sample comparing the predicted distribution with true scalar value.


    :param y_true: matrix (n_samples, n_quantiles)
    :param y_pred: vector of true values
    :param return_mean: True to compute
    '''
    assert y_true.shape[0] == y_pred.shape[0]

    result = [ pscore(y_pred[i, :], y_true[i]).compute() for i in range(y_true.shape[0])]
    result = np.array(result)

    # Compute
    if return_mean:
        result = np.mean(result, axis=0)
    return result




@dataclass
class Scorer:
    '''
    Simple class to compute timeseries performance.
    '''

    metrics: Any = ('mae', 'bias')


    def compute_score(self, data: pd.DataFrame, target: Any, model_list: List[Any]) -> pd.DataFrame:
        '''
        Compute score.
        :param data: pandas dataframe that MUST contains target and model_list columns
        '''

        # Prepare score table
        output = pd.DataFrame(np.nan,
                              index=pd.Index(model_list, name='model'),
                              columns=pd.Index(self.metrics, name='metric'))

        # Compute all score for each model
        for mdl in model_list:
            for metric in self.metrics:
                metric_operator = Scorer._metric_map(metric)
                output.loc[mdl, metric] = metric_operator(data[target], data[mdl])

        return output

    @staticmethod
    def _metric_map(metric: str):
        '''Return function associated to metric name'''

        metric_map = {
            'mae': mean_absolute_error,
            'mse': mean_squared_error,
            'bias': lambda y_true, y_pred: np.mean(y_true-y_pred),
            'count': lambda y_true, y_pred: len(y_true),
            'mape': lambda y_true, y_pred: mean_absolute_error(y_true=y_true, y_pred=y_pred) / np.mean(y_true)
        }

        return metric_map[metric]





if __name__ == '__main__':

    df = pd.DataFrame(np.random.randn(100, 5), columns=range(5))

    scorer = Scorer(metrics=('mae', 'bias', 'mse', 'count'))
    results = scorer.compute_score(data=df, target=0, model_list=[1,2,3,4])

    print(results)




