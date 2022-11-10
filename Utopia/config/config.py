from typing import Dict
from dataclasses import dataclass
import os
import yaml
import pathlib
from datetime import datetime

@dataclass
class Config:
    '''Simple class to load yaml config and define main path'''

    def __post_init__(self):
        self._values = {}

    def load(self):
        self.__build_path__()
        self._load_yaml(os.path.join(self.config_path, 'config.yaml'))

        self._format_values()
        return self

    def __build_path__(self):

        # Retrieve package path
        self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        # Deduce all other paths
        self.output_path = os.path.join(self.folder_path, 'output')
        self.score_path = os.path.join(self.output_path, 'score')
        self.plot_path = os.path.join(self.output_path, 'plot')
        self.plot_html_path = os.path.join(self.output_path, 'plot', 'plot_html')
        self.bayesian_plot_path = os.path.join(self.output_path, 'plot', 'bayesian_plot')
        self.config_path = os.path.join(self.folder_path, 'config')
        self.data_path = os.path.join(self.folder_path, 'data')
        self.consumption_path = os.path.join(self.folder_path, 'data', 'consumption.csv')

        # Make sure folders exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.score_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)
        os.makedirs(self.plot_html_path, exist_ok=True)
        os.makedirs(self.bayesian_plot_path, exist_ok=True)


    def _load_yaml(self, path: str) -> None:
        '''Load yaml and update values dictionary'''
        with open(path) as file:
            config_yaml = yaml.load(file, Loader=yaml.FullLoader)

        self._values.update(config_yaml)

    def _format_values(self) -> None:
        '''Format dictionary keys'''

        # Convert datetime string into datetime
        if 'date' in self._values:
            for k, date_str in self._values['date'].items():
                self._values['date'][k] = datetime.strptime(date_str, '%Y%m%d')



    def __getitem__(self, item):
        return self._values[item]

    def __setitem__(self, key, value):
        self._values[key] = value


    def __repr__(self):
        return f'{self._values}'