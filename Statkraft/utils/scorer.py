from dataclasses import dataclass, field
import pandas as pd
from typing import List, Union
import numpy as np

@dataclass
class ModelsScorer:
    data: pd.DataFrame
    metrics: List[str] = field(default_factory=['mae', 'rmse', 'mape'])


    def _initiate_score_df(self, by: Union[str, None], models: List):
        '''Create dataframe where scores will be stored'''
        if by is None:
            score_df = pd.DataFrame(index=self.metrics, columns=models)
        else:
            columns = pd.MultiIndex.from_product([self.metrics, models])
            score_df = pd.DataFrame(index=self.data[by].unique(), columns=columns)
        return score_df


    def compute_score(self, target, models: List, by: Union[str, None] = None) -> pd.DataFrame:
        '''Compute scores for each model and each metric'''

        score_df = self._initiate_score_df(by, models)

        if by is None:
            for m in models:
                for metric in self.metrics:
                    score_df.loc[metric, m] = self.compute_metric(metric, self.data[target], self.data[m])
        else:
            for b in score_df.index:
                data_by = self.data.query(f'{by}=={b}')
                for m in models:
                    for metric in self.metrics:
                        score_df.loc[b, (metric, m)] = self.compute_metric(metric, data_by[target], data_by[m])

        return score_df



    def compute_metric(self, metric: str, target, model):
        if metric == 'mae':
            return self._mae(target, model)
        elif metric == 'rmse':
            return self._rmse(target, model)
        elif metric == 'mape':
            return self._mape(target, model)
        elif metric == 'bias':
            return self._bias(target, model)
        elif metric == 'bias_percentage':
            return self._bias_percentage(target, model)
        else:
            raise ValueError(f'Unknown metric: {metric}')

    def _mae(self, target, model):
        return np.mean(np.abs(target - model))

    def _rmse(self, target, model):
        return np.sqrt(np.mean(np.square(target - model)))

    def _mape(self, target, model):
        return np.mean(np.abs(target - model)/target)

    def _bias_percentage(self, target, model):
        return np.mean((target - model)/target)

    def _bias(self, target, model):
        return np.mean(target - model)