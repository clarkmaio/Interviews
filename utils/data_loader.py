import pandas as pd
from dataclasses import dataclass
import numpy as np
from typing import Dict

from Utopia.utils.time_utils import load_calendar
from Utopia.config.config import Config

@dataclass
class DataLoader:
    path: str
    config: Config

    def load(self, granularity: str = 'D', resample_operation: str = 'sum', aggregate_by_track: bool = True) -> pd.DataFrame:
        '''

        :param granularity: Choose between D, W, MS
        :return:
        '''
        df = self._load_csv(self.path)
        df = self._postprocess(data=df, granularity=granularity, resample_operation=resample_operation, aggregate_by_track=aggregate_by_track)
        return df


    def _load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df


    def _postprocess(self, data: pd.DataFrame, granularity: str, resample_operation: str = 'sum', aggregate_by_track: bool = True) -> pd.DataFrame:
        '''
        Postprocess data:
        - rename columns
        - additional features
        '''

        # Build trackid and stationid map and rename track id for simplicity
        self._build_trackid_map(data=data)
        self._build_stationid_map(data=data)
        data['TrackId'] = data['TrackId'].map(self.trackid_map)
        data['StationId'] = data['StationId'].map(self.stationid_map)

        # Rename columns
        data.rename(columns={'Day': 'valuedate'}, inplace=True)

        # Format columns
        data['valuedate'] = pd.to_datetime(data['valuedate'])

        # Aggregate by StationId
        if aggregate_by_track:
            data = data.groupby(['valuedate', 'TrackId']).sum()['Plays'].reset_index()
            groupby_var = ['TrackId']
        else:
            groupby_var = ['TrackId', 'StationId']

        # Resample w.r.t. granularity input
        resample_operation_fun = self.operation_map(operation=resample_operation)
        data = data.groupby(groupby_var).resample(granularity, on='valuedate').apply(resample_operation_fun)['Plays'].reset_index()

        # Build calendar and merge
        calendar = load_calendar(st_date=data['valuedate'].min(), en_date=data['valuedate'].max(), freq='D')
        data = pd.merge(data, calendar, how='left', on=['valuedate'])

        # Assign Holiday weight
        data['holiday_weight'] = self.config['holiday_weight']['default']
        data.loc[data['isSummer'], 'holiday_weight'] = self.config['holiday_weight']['summer']
        data.loc[data['isChristmas'], 'holiday_weight'] = self.config['holiday_weight']['christmas']

        return data


    def _build_trackid_map(self, data: pd.DataFrame):
        trackid_list = data['TrackId'].unique()
        self.trackid_map = {trackid: id for (id, trackid) in enumerate(trackid_list)}


    def _build_stationid_map(self, data: pd.DataFrame):
        stationid_list = data['StationId'].unique()
        self.stationid_map = {stationid: id for (id, stationid) in enumerate(stationid_list)}


    def operation_map(self, operation: str):
        map = {'sum': np.sum, 'mean': np.mean}
        return map[operation]


@dataclass
class DataContainer:
    data_dict: Dict

    def __post_init__(self):
        self._values = {}
        self._values.update(self.data_dict)

    def __setattr__(self, key, value):
        self._values[key] = value

    def __getattr__(self, item):
        return self._values[item]

    def __setitem__(self, key, value):
        self._values[key] = value

    def __getitem__(self, item):
        return self._values[item]

    def __repr__(self):
        return f'{self._values.keys()}'







if __name__ == '__main__':
    config = Config().load()
    df = DataLoader(path=config.consumption_path, config=config).load(granularity='W')
