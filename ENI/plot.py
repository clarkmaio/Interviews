import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import numpy as np
import itertools

from typing import Tuple, Union, List

from catboost import CatBoostClassifier
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


sns.set_theme(style="ticks")
matplotlib.use('TkAgg')





class Plot:

    @staticmethod
    def heatmap_correlation(X: pd.DataFrame, save_path: str, title = 'Correlation matrix') -> None:
        '''
        Compute correlation matrix as plot as heatmap
        '''
        # compute corre
        corr_matrix = X.corr()

        fig = plt.figure(figsize=(10, 10))
        plt.title(title, fontweight = 'bold')
        sns.heatmap(corr_matrix.round(2), square=True, linewidths=.5, annot=True, annot_kws={"size": 8})
        plt.savefig(os.path.join(os.path.join(save_path, f'{title}.png')))
        plt.close()

    @staticmethod
    def shap_values(X: pd.DataFrame, y: pd.DataFrame, save_path: str) -> None:
        '''
        Plot Shapley feature importance using tree model
        '''

        # Feat simple tree model
        mdlCatBoost = CatBoostClassifier()
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=.7, shuffle=True, stratify=y['NSP'])
        mdlCatBoost.fit(X, y, verbose=0, eval_set=(X_val, y_val), use_best_model=True)

        # Fit Shapley explainer
        explainer = shap.TreeExplainer(mdlCatBoost)
        shap_values = explainer.shap_values(X)

        # Generate and save shapley summary plot
        shap.summary_plot(shap_values=shap_values, features=X, feature_names=X.columns, class_names = mdlCatBoost.classes_, plot_type='bar', show=False, plot_size=(7, 8))
        plt.grid(linestyle=':')
        legend = plt.legend()
        legend.set_title('NSP')
        plt.savefig(os.path.join(os.path.join(save_path, 'shapley_feature_importance.png')))
        plt.close()

    @staticmethod
    def class_frequency(y: pd.DataFrame, save_path: str) -> None:
        '''Simple bar plot of classes frequency'''

        class_list = y['NSP'].unique()

        frequency_df = pd.DataFrame(np.nan, index=pd.Index(class_list, name='class'), columns=['count', 'percentage']).sort_index()
        for c in frequency_df.index:
            frequency_df.loc[c, 'count'] = len(y.query(f'NSP == {c}'))
            frequency_df.loc[c, 'percentage'] = 100 * len(y.query(f'NSP == {c}')) / len(y)
            frequency_df = frequency_df.round(1)


        ax = sns.countplot(x = y['NSP'])
        plt.text(0, frequency_df.loc[1, "count"] - 130, f'{frequency_df.loc[1, "percentage"]}%', color='white', bbox={'facecolor':'none', 'edgecolor':'white', 'boxstyle':'round'}, horizontalalignment='center')
        plt.text(1, frequency_df.loc[2, "count"] + 50, f'{frequency_df.loc[2, "percentage"]}%', color='red', bbox={'facecolor':'none', 'edgecolor':'red', 'boxstyle':'round'}, horizontalalignment='center')
        plt.text(2, frequency_df.loc[3, "count"] + 50, f'{frequency_df.loc[3, "percentage"]}%', color='red', bbox={'facecolor':'none', 'edgecolor':'red', 'boxstyle':'round'}, horizontalalignment='center')
        plt.grid(linestyle = ':')
        plt.title('Class frequency', fontweight = 'bold')


        plt.savefig(os.path.join(save_path, 'class_frequency.png'))
        plt.close()



    @staticmethod
    def class_distribution(X: pd.DataFrame, y: pd.DataFrame, save_path: str) -> None:
        '''
        Plot normalised distribution of features clustered by class
        :param X:
        :return:
        '''

        data = pd.concat([X, y], axis=1)

        sns.displot(data=data, x='DP', kind='kde', hue='NSP', fill=True)
        plt.grid(linestyle=':')
        plt.title('Density by class', fontweight='bold')


        for v in X.columns:
            # Plot dist for each class
            ax = sns.kdeplot(data=data.query('NSP==1'), x=v, fill=True, color='blue', label=1)
            sns.kdeplot(data=data.query('NSP==2'), x=v, fill=True, color='orange', ax=ax, label=2)
            sns.kdeplot(data=data.query('NSP==3'), x=v, fill=True, color='green', ax=ax, label=3)

            legend = plt.legend()
            legend.set_title('NSP')
            plt.grid(linestyle=':')
            plt.title(f'{v}: class distribution', fontweight = 'bold')

            plt.savefig(os.path.join(save_path, f'{v}_distribution_plot.png'))
            plt.close()

    @staticmethod
    def joint_scatter(X: pd.DataFrame, y: pd.DataFrame, couple_list: List[Tuple], save_path: str, alpha: float = .7) -> None:
        '''Generate joint plot using columns in couple list'''

        data = pd.concat([X, y], axis=1)

        for (c1, c2) in couple_list:
            JointGrid = sns.jointplot(data=data, x = c1, y = c2, hue='NSP', alpha=alpha)
            JointGrid.ax_joint.grid(linestyle=':')
            JointGrid.fig.suptitle(f'Joint plot: {c1} - {c2}', fontweight='bold')
            plt.savefig(os.path.join(save_path, f'jointplot_{c1}_{c2}.png'))
            plt.close()


    @staticmethod
    def confusion_matrix_grid(model_list: List[str], y_pred: pd.DataFrame, y_test: pd.DataFrame, save_path: str)  -> None:
        '''
        Plot confusion matrix of model prediction applied to test sets.
        Use sklearn function plot_confusion_matrix and organize in a grid

        :param model_list: list of models to process
        :param model_dict: dictionary containing trained models.
        :param X_test, y_test: test set
        '''

        assert len(model_list) <= 4 # 4 plot at most
        list_plot_position = list(itertools.product([0, 1], [0, 1]))
        list_plot_position = list_plot_position[:len(model_list)]

        fig, ax = plt.subplots(2,2, figsize = (10, 10))
        fig.suptitle('Confusion matrix', fontweight='bold')
        for mdl, (posx, posy) in zip(model_list, list_plot_position):
            ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred[mdl], ax=ax[posx][posy])
            ax[posx][posy].set_title(mdl, fontweight='bold')

            if posx == 0:
                ax[posx][posy].set_xlabel('')

            if posy == 1:
                ax[posx][posy].set_ylabel('')

        plt.savefig(os.path.join(save_path, 'confusion_matrix_grid.png'))
        plt.close()



    @staticmethod
    def brute_force_feature_performance(model_list: List[str], score_couple: Tuple[str, str], brute_force_score: pd.DataFrame, save_path: Union[str, None] = None) -> None:
        '''

        :param brute_force_score:
        :param save_path:
        :return:
        '''

        # Color palette
        color_dict = {'Logistic': 'gray', 'LogisticWeight': 'black', 'CatBoost': 'green', 'CatBoostWeight': 'red', 'CatBoostSMOTE': 'blue'}

        fig, ax = plt.subplots(2, 1, figsize=(7, 8))
        fig.suptitle('Brute force feature impact', fontweight='bold')
        for mdl in model_list:
            ax[0].plot(brute_force_score.index, brute_force_score.loc[:, (mdl, score_couple[0])], label=mdl, color=color_dict[mdl], marker='.')
        ax[0].grid(linestyle=':')
        ax[0].set_title(score_couple[0])
        ax[0].legend()

        for mdl in model_list:
            ax[1].plot(brute_force_score.index, brute_force_score.loc[:, (mdl, score_couple[1])], label=mdl, color=color_dict[mdl], marker='.')
        ax[1].grid(linestyle=':')
        ax[1].set_xlabel('Number of features', fontweight='bold')
        ax[1].set_title(score_couple[1])
        ax[1].legend()

        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'Brute_force_feature_impact.png'))
            plt.close()











