import os
from typing import Union
import numpy as np
import random

from Utopia.config.config import Config
from Utopia.orchestrator.forecast_orchestrator import ForecastOrchestrator, BayesianForecastOrchestrator
from Utopia.orchestrator.analysis_orchestrator import AnalysisOrchestrator

import warnings
warnings.simplefilter('ignore')

def print_start_module(module: str) -> None:
    print('********************************')
    print(f'* Start module {module}')
    print('********************************')

def set_seed(seed: Union[int, None]) -> None:

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

if __name__ == '__main__':
    config = Config().load()
    set_seed(config['SEED'])

    # **************************************
    # Run modules according to config params
    # **************************************

    if config['run_modules']['analysis']:
        print_start_module('Analysis')
        analysis_orchestrator = AnalysisOrchestrator(config=config)
        analysis_orchestrator.run()

    if config['run_modules']['forecast']:
        print_start_module('Forecast')
        forecast_orchestrator = ForecastOrchestrator(config=config)
        forecast_orchestrator.run()

    if config['run_modules']['forecast_bayesian']:
        print_start_module('Bayesian Forecast')
        bayesian_forecast_orchestrator = BayesianForecastOrchestrator(config=config)
        bayesian_forecast_orchestrator.run()






