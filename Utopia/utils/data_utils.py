'''
Simple functions to manage data
'''

import pandas as pd

def pivot_by_trackid(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Just pivot data w.r.t. TrackId column

    :param df: pandas DataFrame, must contains columns: valuedate, TrackId, Plays
    '''
    df_pivot = pd.pivot_table(data=df, index='valuedate', columns='TrackId', values='Plays')
    df_pivot.fillna(0, inplace=True)
    df_pivot.columns = [f'Track {c}' for c in df_pivot.columns]
    return df_pivot


