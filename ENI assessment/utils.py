import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from typing import Dict, Tuple, Union
from sklearn.metrics import recall_score, fbeta_score

LINK = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls'


def load_raw_data(link: str = LINK, return_X_y = True) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    '''
    Download raw dataset from url link
    :param link: link to xls file
    :param return_X_y: True to return two dataframe (features and target)
    '''
    df = pd.read_excel(link, sheet_name='Raw Data')
    df.dropna(subset='NSP', inplace=True)
    df['NSP'] = df['NSP'].astype(int).astype("category")

    regressor_column = ['LB', 'AC', 'FM', 'UC',
                        'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL', 'DS', 'DP',
                        'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    target_column = ['NSP']

    df['DR'].unique()

    if return_X_y:
        X = df[regressor_column]
        y = df[target_column]
        return X, y
    else:
        return df.loc[:, regressor_column + target_column]


def train_validation_test_split(X: pd.DataFrame, y: Union[pd.DataFrame, None] = None,
                                train_size: float = .50, validation_size: float = .30, stratify_by_target: bool = True, **kwargs):
    '''
    Split X (and y if not None) into 3 dataframe: Train, validation and test

    :param X:
    :param y:
    :param train_size: percentage of train data
    :param validation_size: percentage of test data
    :param stratify_by_target: True to stratify by value in y data frame
    :return:
    '''

    # Make sure train and validation size are less then 1
    if train_size + validation_size >= 1:
        raise ValueError('Train size + Validation size must be less than 1')

    test_size = 1-train_size-validation_size

    # Split training set
    if stratify_by_target:
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=train_size, stratify=y,**kwargs)
    else:
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=train_size, **kwargs)

    # Split validation and test
    validation_size_calib = validation_size / (validation_size + test_size)
    if stratify_by_target:
        X_validation, X_test, y_validation, y_test = train_test_split(X_tmp, y_tmp, train_size=validation_size_calib, stratify=y_tmp, **kwargs)
    else:
        X_validation, X_test, y_validation, y_test = train_test_split(X_tmp, y_tmp, train_size=validation_size_calib, **kwargs)

    return X_train, X_validation, X_test, y_train, y_validation, y_test




def return_class_weights_map(y: pd.Series) -> Dict:
    '''
    Given a vector of classes return a map with weight for each calss.

    ____________________
    The weights are built in the following way:
    Let ci = # sample of class i
    K = 1/c1 + 1/c2 + 1/c3 ...

    Weight class i: K_i = 1/ci / K

    ____________________
    Note:
    - Sum_i K_i = 1
    - the less sample for a class, bigger the weight
    '''

    class_count = y.value_counts()
    K = np.sum(1/class_count)

    weights = 1/class_count/K
    weights = dict(weights)
    return weights


def pprint_table(df: pd.DataFrame) -> None:
    print(tabulate(df, headers='keys', tablefmt='psql'))


def print_header(header: str) -> None:
    print('\n\n')
    print('*************************************************************')
    print(f'{header}')
    print('*************************************************************')



def avg_class_fbeta_score(y_true, y_pred, beta: float = 1.):
    output = fbeta_score(y_true=y_true, y_pred=y_pred, labels=[1, 2, 3], average=None, beta=beta)
    return np.mean(output)

def avg_single_class_fbeta_score(y_true, y_pred, class_label: int, beta: float = 1.):
    labels = [1, 2, 3]
    label_idx = labels.index(class_label)
    output = fbeta_score(y_true=y_true, y_pred=y_pred, labels=[1, 2, 3], average=None, beta=beta)
    return output[label_idx]

def avg_single_class_recall_score(y_true, y_pred, class_label: int):
    labels = [1,2,3]
    label_idx = labels.index(class_label)
    output = recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average=None)
    return output[label_idx]




if __name__ == '__main__':

    X, y = load_raw_data()
    print(X.describe())




