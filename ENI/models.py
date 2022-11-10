import pandas as pd
import random
from typing import Dict, Tuple, Union, List
import numpy as np
from dataclasses import dataclass

from utils import avg_class_fbeta_score, avg_single_class_fbeta_score, avg_single_class_recall_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineSMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

class MostFrequentClassifier():
    '''
    Simple model that predict most frequent class in training set
    '''
    def __init__(self):
        self.most_frequent_class = None
        self._estimator_type='classifier'

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        assert len(y.columns) == 1
        self.most_frequent_class = y.iloc[:, 0].mode().values[0]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred = pd.DataFrame(self.most_frequent_class, index=X.index, columns=['MostFrequentClf'])
        return y_pred



class SmartRandomClassifier():
    '''
    Simple model that predict random class.
    Probability to predict a class is proportional to class frequency in training set 
    '''
    def __init__(self):
        self.class_list = None
        self.class_weights = None
        self._estimator_type = 'classifier'

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        assert len(y.columns) == 1

        # Store  class freq
        self.class_list = y.iloc[:, 0].unique()
        self.class_weights = y.iloc[:, 0].value_counts()[self.class_list].tolist()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        extractions = random.choices(population=self.class_list, weights=self.class_weights, k = len(X))
        y_pred = pd.DataFrame(extractions, index=X.index, columns=['SmartRandomClf'])
        return y_pred




@dataclass
class ModelHub:
    model_list: List[str]
    sample_weight_map: Dict = None


    def __post_init__(self):

        self.__check_model_list__(model_list=self. model_list)
        self.model_dict = self._initialize_models()
        self.pred_dict = {}
        return

    def __check_model_list__(self, model_list: List[str]):
        '''Raise error if there is a model in model list that has not been implemented'''
        self._implemented_model_list = ['SmartRandom', 'MostFrequent', 'Logistic', 'LogisticWeight', 'CatBoost', 'CatBoostWeight', 'CatBoostSMOTE']

        unknwon_model = []
        for m in self.model_list:
            if m not in self._implemented_model_list:
                unknwon_model.append(m)

        if len(unknwon_model) > 0:
            raise RuntimeError(f'Unknown model {unknwon_model}. Choose between {self._implemented_model_list}')



    def _initialize_models(self) -> Dict:

        model_build = {
            'SmartRandom': SmartRandomClassifier(),
            'MostFrequent': MostFrequentClassifier(),
            'Logistic': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(penalty='l2'))]),
            'LogisticWeight': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(penalty='l2', class_weight=self.sample_weight_map))]),
            'CatBoost': CatBoostClassifier(early_stopping_rounds=50, num_trees=500),
            'CatBoostWeight': CatBoostClassifier(early_stopping_rounds=50, num_trees=500, class_weights=self.sample_weight_map),
            'CatBoostSMOTE': PipelineSMOTE([('SMOTE', SMOTE()), ('under', RandomUnderSampler()), ('scaler', StandardScaler()), ('model', CatBoostClassifier(early_stopping_rounds=50, num_trees=500))]),
        }

        # Return only models in model list
        model_dict = {m: model_build[m] for m in self.model_list}
        return model_dict


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame):
        '''
        Train each model in model list.
        Here is implemented a series of if/else to deal with different models

        :param X:
        :param y:
        :param X_validaton: this dataset can be eventually concatenated to X in case model do not need validation
        :param y_validation: this dataset can be eventually concatenated to y in case model do not need validation
        :param sample_weight_train:
        :param sample_weight_validation:
        :return:
        '''

        for model in self.model_list:
            print(f'> Training {model}')

            if model in ('SmartRandom', 'MostFrequent'):
                # Normal fit
                X_concat, y_concat = self._concat_train_validation(X=X, y=y, X_validaton=X_validation, y_validation=y_validation)
                self.model_dict[model].fit(X=X_concat, y=y_concat)

            elif model in ('Logistic'):
                # Normal fit
                X_concat, y_concat = self._concat_train_validation(X=X, y=y, X_validaton=X_validation, y_validation=y_validation)
                self.model_dict[model].fit(X=X_concat, y=y_concat.iloc[:, 0])

            elif model in ('LogisticWeight'):
                X_concat, y_concat = self._concat_train_validation(X=X, y=y, X_validaton=X_validation, y_validation=y_validation)
                self.model_dict[model].fit(X=X_concat, y=y_concat.iloc[:, 0])

            elif model in ('CatBoost'):
                self.model_dict[model].fit(X, y, eval_set = (X_validation, y_validation), verbose=0, use_best_model=True)

            elif model in ('CatBoostSMOTE'):
                self.model_dict[model].fit(X, y, model__eval_set = (X_validation, y_validation), model__verbose=0, model__use_best_model=True)

            elif model in ('CatBoostWeight'):
                self.model_dict[model].fit(X, y, eval_set = (X_validation, y_validation), verbose=0, use_best_model=True)


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Predict using every model in model list and return prediction in an unique dataframe
        '''

        pred_df = pd.DataFrame(np.nan, index=X.index, columns=pd.Index(self.model_list, name='model'))
        for model in self.model_list:
            pred_df[model] = self.model_dict[model].predict(X)
            self.pred_dict[model] = pred_df[model]

        return pred_df

    def _concat_train_validation(self,  X: pd.DataFrame, X_validaton: pd.DataFrame, y: Union[pd.DataFrame, None], y_validation: Union[pd.DataFrame, None]) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        '''
        Concat train and validation set (both regressor and target) at once
        '''
        X = pd.concat([X, X_validaton], axis=0)
        if y is None:
            return X
        else:
            y = pd.concat([y, y_validation], axis=0)
            return X, y


    def cross_validation(self, X: pd.DataFrame, y: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
        '''Compute score via cross validation'''

        cv_score = pd.DataFrame(np.nan, columns=['Accuracy'], index=self.model_list)

        for mdl in self.model_list:
            print(f'Compute cv score {mdl}...')

            if mdl == 'CatBoostSMOTE':
                fit_params = {'model__verbose': 0}
            elif mdl == 'CatBoost':
                fit_params = {'verbose': 0}
            elif mdl == 'CatBoostWeight':
                fit_params = {'verbose': 0}
            elif mdl == 'LogisticWeight':
                fit_params = None
            else:
                fit_params = None

            cv_score.loc[mdl, 'Accuracy'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring='accuracy'))
            cv_score.loc[mdl, 'bAccuracy'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring='balanced_accuracy'))
            cv_score.loc[mdl, 'kScore'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(cohen_kappa_score)))
            cv_score.loc[mdl, 'F2Score'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_class_fbeta_score, beta=2)))
            cv_score.loc[mdl, 'F2Score_std'] = np.std(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_class_fbeta_score, beta=2)))
            cv_score.loc[mdl, 'F1Score'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_class_fbeta_score, beta=1)))

            cv_score.loc[mdl, 'F1Score_1'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_single_class_fbeta_score, beta=1, class_label=1)))
            cv_score.loc[mdl, 'F1Score_2'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_single_class_fbeta_score, beta=1, class_label=2)))
            cv_score.loc[mdl, 'F1Score_3'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_single_class_fbeta_score, beta=1, class_label=3)))

            cv_score.loc[mdl, 'Recall_1'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_single_class_recall_score, class_label=1)))
            cv_score.loc[mdl, 'Recall_2'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_single_class_recall_score, class_label=2)))
            cv_score.loc[mdl, 'Recall_3'] = np.mean(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_single_class_recall_score, class_label=3)))
            cv_score.loc[mdl, 'Recall_3_std'] = np.std(cross_val_score(estimator=self.model_dict[mdl], X=X, y=y, cv=cv, fit_params=fit_params, scoring=make_scorer(avg_single_class_recall_score, class_label=3)))

        cv_score = cv_score.round(3)
        return cv_score
