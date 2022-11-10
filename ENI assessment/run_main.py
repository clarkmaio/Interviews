import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Dict, Tuple, Union
import os

import seaborn as sns
import matplotlib
sns.set_theme(style="ticks")
matplotlib.use('TkAgg')

from plot import Plot
from utils import load_raw_data, train_validation_test_split, return_class_weights_map, pprint_table, print_header, avg_single_class_fbeta_score, avg_single_class_recall_score


# Models
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, confusion_matrix, balanced_accuracy_score, classification_report, cohen_kappa_score
from models import ModelHub


class Orchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.__define_path__()

    def __define_path__(self):
        self.folder_path = os.path.dirname(__file__)
        self.plot_path = os.path.join(self.folder_path, 'plot')
        self.distribution_plot_path = os.path.join(self.folder_path, 'plot', 'distribution')
        self.output_path = os.path.join(self.folder_path, 'output')

        # Make sure folder exist
        os.makedirs(self.plot_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.distribution_plot_path, exist_ok=True)


    @abstractmethod
    def run(self):
        raise NotImplementedError('Missing run method')

    def load_data(self, return_X_y: bool = True) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        output = load_raw_data(return_X_y=return_X_y)
        return output

    def print_weight_stability(self, X: pd.DataFrame, y: pd.DataFrame, n_run: int = 10):
        '''
        Compute sample weight multiple times to verify the stability
        '''
        weight_stability = pd.DataFrame(index=range(n_run), columns=['k1', 'k2', 'k3'])
        for i in range(n_run):
            X_train, X_validation, X_test, y_train, y_validation, y_test = train_validation_test_split(X=X, y=y, train_size=self.config['train_size'], validation_size=self.config['validation_size'],
                                                                                                       stratify_by_target=True,
                                                                                                       shuffle = True)
            weight_map = return_class_weights_map(y=y_train['NSP'])
            weight_stability.loc[i, 'k1'] = weight_map[1]
            weight_stability.loc[i, 'k2'] = weight_map[2]
            weight_stability.loc[i, 'k3'] = weight_map[3]

        describe_df = weight_stability.astype(float).describe()
        describe_df  = describe_df.loc[['mean', 'std'], :]
        describe_df.loc['confidence_interval', :] = describe_df.loc['std', :] * 2
        describe_df = describe_df.round(4)

        print('>>>>>>>>>> Sample weight stability')
        pprint_table(describe_df)


class AnalysisOrchestrator(Orchestrator):
    '''
    Just perform some data analysis and generate plot
    '''
    def __init__(self, config: Dict):
        super().__init__(config)

    def run(self):
        print_header('START DATA ANALYSIS')

        # Load data
        X, y = self.load_data(return_X_y=True)

        # Crete plots
        self.generate_plot_analysis(X=X, y=y)



    def generate_plot_analysis(self, X: pd.DataFrame, y: pd.DataFrame) -> None:

        # Count class element
        print('> Plot class frequency')
        Plot.class_frequency(y=y, save_path=self.plot_path)

        # Distribution plots
        print('> Plot single variable class distribution')
        Plot.class_distribution(X=X, y=y, save_path=self.distribution_plot_path)

        # Joint plot
        print('> Plot joint scatter')
        Plot.joint_scatter(X=X, y=y, couple_list = [('Mean', 'Variance'), ('ALTV', 'MLTV'), ('DL', 'DP')], save_path=self.plot_path)

        # Correlation plot
        print('> Plot correlation matrix')
        Plot.heatmap_correlation(X=X, save_path=self.plot_path)
        Plot.heatmap_correlation(X=X[['Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']], title='Correlation matrix histogram features', save_path=self.plot_path)

        # Shap feature importance
        print('> Plot SHAP')
        Plot.shap_values(X=X, y=y, save_path=self.plot_path)






class SingleRunOrchestrator(Orchestrator):
    '''
    Class to run single experiment.
    Useful to play with models and understand performance.
    Final Result is strongly dependent on seed due to data split.
    '''

    def __init__(self, config: Dict):
        super().__init__(config)

    def run(self):
        print_header('START SINGLE RUN')

        # ------------ RETRIEVE DATA
        X, y = self.load_data(return_X_y=True)
        X_train, X_validation, X_test, y_train, y_validation, y_test = train_validation_test_split(X=X, y=y, train_size=self.config['train_size'], validation_size=self.config['validation_size'],
                                                                                                   stratify_by_target=True,
                                                                                                   shuffle = True)

        # ------------ COMPUTE WEIGHTS
        self.print_weight_stability(X=X, y=y, n_run=10)
        weight_map = return_class_weights_map(y=y_train['NSP'])

        # ------------ MODELS
        model_hub = ModelHub(model_list=self.config['model_list'], sample_weight_map=weight_map)

        print('\n\n>>>>>>>>>> TRAINING')
        model_hub.fit(X=X_train[self.config['model_features']], y=y_train, X_validation=X_validation[self.config['model_features']], y_validation=y_validation)
        y_pred = model_hub.predict(X=X_test[self.config['model_features']])


        # ------------ SCORE
        score=self.compute_score(y_test=y_test, y_pred = y_pred, save_path=self.output_path)
        print('\n\n>>>>>>>>>> PERFORMANCE:')
        pprint_table(score.loc[:, ['bAccuracy', 'kScore', 'F2Score_3', 'F2Score_2', 'F2Score_1', 'Recall_3', 'Recall_2', 'Recall_1', 'MeanPrecision', 'MeanRecall', 'MeanF2Score']])



        # ------------ PLOTS
        # Generate confusion matrix of best models
        Plot.confusion_matrix_grid(model_list=['LogisticWeight', 'CatBoost', 'CatBoostSMOTE', 'CatBoostWeight'],
                                   y_pred=y_pred, y_test=y_test,
                                   save_path=self.plot_path)


        self.brute_force_feature_improvement(model_hub=model_hub,
                                             X_train=X_train, y_train=y_train,
                                             X_validation=X_validation, y_validation=y_validation,
                                             X_test=X_test, y_test=y_test)




    def compute_score(self, y_pred: pd.DataFrame, y_test: pd.DataFrame, save_path: Union[str, None] = None) -> pd.DataFrame:
        '''For each model in mdl dict compute score metrics and save dataframe as xlsx'''

        model_list = y_pred.columns
        score = pd.DataFrame(np.nan, columns=['Accuracy'], index=model_list)

        for m in model_list:
            score.loc[m, 'Accuracy'] = accuracy_score(y_test, y_pred[m])
            score.loc[m, 'bAccuracy'] = balanced_accuracy_score(y_test, y_pred[m]) # Average of recalls
            score.loc[m, 'kScore'] = cohen_kappa_score(y_test, y_pred[m])
            score.loc[m, ['Recall_1', 'Recall_2', 'Recall_3']] = recall_score(y_test, y_pred[m], labels=[1, 2, 3], average=None)
            score.loc[m, ['Precision_1', 'Precision_2', 'Precision_3']] = precision_score(y_test, y_pred[m], labels=[1, 2, 3], average=None)
            score.loc[m, ['F2Score_1', 'F2Score_2', 'F2Score_3']] = fbeta_score(y_test, y_pred[m], beta=2, labels=[1, 2, 3], average=None)
            score.loc[m, ['F1Score_1', 'F1Score_2', 'F1Score_3']] = fbeta_score(y_test, y_pred[m], beta=2, labels=[1, 2, 3], average=None)
            score.loc[m, 'MeanRecall'] = score.loc[m, ['Recall_1', 'Recall_2', 'Recall_3']].mean()
            score.loc[m, 'MeanPrecision'] =score.loc[m, ['Precision_1', 'Precision_2', 'Precision_3']].mean()
            score.loc[m, 'MeanF2Score'] = score.loc[m, ['F2Score_1', 'F2Score_2', 'F2Score_3']].mean()

        score = score.round(3) # Round to make it readable

        if save_path is not None:
            score.to_excel(os.path.join(self.output_path, 'singlerun_score.xlsx'), engine='xlsxwriter')
        return score

    def brute_force_feature_improvement(self, model_hub: ModelHub, X_train: pd.DataFrame, y_train: pd.DataFrame, X_validation: pd.DataFrame, y_validation: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
        '''
        Train, predict and compute score adding one feature at the time.
        Plot performance curve to evaluate at which point model reach saturation
        '''

        brute_force_score_dict = {mdl: pd.DataFrame(np.nan, index=pd.Index(range(1, len(self.config['from_best_to_worse_features'])+1), name='n_feature'), columns=['bAccuracy']) for mdl in model_hub.model_list}

        for i in range(1, len(self.config['from_best_to_worse_features'])+1):
            print(f'Training with {i+1} best variables')
            features_list = self.config['from_best_to_worse_features'][:i]

            model_hub.fit(X=X_train[features_list], y=y_train, X_validation=X_validation[features_list], y_validation=y_validation)
            y_pred = model_hub.predict(X=X_test[features_list])

            for mdl in brute_force_score_dict.keys():
                brute_force_score_dict[mdl].loc[i, 'bAccuracy'] = balanced_accuracy_score(y_test, y_pred[mdl])
                brute_force_score_dict[mdl].loc[i, 'kScore'] = cohen_kappa_score(y_test, y_pred[mdl])
                brute_force_score_dict[mdl].loc[i, 'F2Score'] = np.mean(fbeta_score(y_test, y_pred[mdl], beta=2, labels=[1, 2, 3], average=None))
                brute_force_score_dict[mdl].loc[i, 'Recall_3'] = recall_score(y_test, y_pred[mdl], average=None)[2]

        bruce_force_all = pd.concat(brute_force_score_dict, axis=1, names=['model', 'metric'])
        Plot.brute_force_feature_performance(model_list=['LogisticWeight', 'CatBoostWeight', 'CatBoostSMOTE', 'CatBoost'], brute_force_score=bruce_force_all, score_couple=('F2Score', 'Recall_3'), save_path=self.plot_path)



class CrossValidationOrchestrator(Orchestrator):
    '''
    Class to perform CrossValidation scoring.
    use this to compute stable result.
    '''
    def __init__(self, config: Dict):
        super().__init__(config)

    def run(self):
        print_header('START CV SCORE')

        # ------------EXCLUDE NOT COMPATIBLE MODELS
        self.config['model_list'] = [x for x in self.config['model_list'] if x not in ('MostFrequent', 'SmartRandom')]

        # ------------ RETRIEVE DATA
        X, y = self.load_data(return_X_y=True)
        X = X[self.config['model_features']]

        X_train, X_validation, X_test, y_train, y_validation, y_test = train_validation_test_split(X=X, y=y, train_size=self.config['train_size'], validation_size=self.config['validation_size'],
                                                                                                   stratify_by_target=True,
                                                                                                   shuffle = True)


        # ------------ COMPUTE WEIGHTS
        weight_map = return_class_weights_map(y=y_train['NSP'])

        # ------------ MODELS
        model_hub = ModelHub(model_list=self.config['model_list'], sample_weight_map=weight_map)

        # ------------ SCORE
        cv_score = model_hub.cross_validation(X=X, y=y['NSP'], cv=5)
        cv_score.to_excel(os.path.join(self.output_path, 'cv_score.xlsx'))



def LoadConfig() -> Dict:
    config = {
        'model_features': ['AC', 'ASTV', 'ALTV', 'DP', 'MSTV', 'Mean', 'Variance'],
        # 'model_features': ['LB', 'AC', 'FM', 'UC',
        #  'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL', 'DS', 'DP',
        #  'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency'],

        'from_best_to_worse_features': ['AC', 'ASTV', 'ALTV', 'Mean', 'MSTV', 'DP', 'UC', 'Median', 'MLTV', 'Mode', 'Variance', 'Max', 'FM', 'Nmax', 'Min', 'LB', 'Width', 'Nzeros', 'DL', 'Tendency'],

        # List of models will be used. Must be implemented in ModelHub class
        'model_list': ['MostFrequent', 'SmartRandom', 'Logistic', 'LogisticWeight', 'CatBoost', 'CatBoostWeight', 'CatBoostSMOTE'],

        'train_size': 0.5,
        'validation_size': 0.2
    }
    return config


if __name__ == '__main__':

    config = LoadConfig()

    # Generate all plots for data analysis
    AnalysisOrchestrator(config=config).run()

    # Single experiment (useful to play with model design and understand performance)
    SingleRunOrchestrator(config=config).run()

    # Compute performance as average of multiple experiment using cross validation
    CrossValidationOrchestrator(config=config).run()

    print_header('FIN.')

