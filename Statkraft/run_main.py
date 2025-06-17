
from StatkraftAssessment.settings.parser import return_parser
import yaml


def load_yaml(yaml_path: str):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)




if __name__ == "__main__":

    # Load settings
    parser =  return_parser()
    settings = load_yaml('./settings/settings.yaml')
    config = {**parser, **settings}

    if config['mode'] == 'analysis':
        from StatkraftAssessment.analysis import AnalysisOrchestrator
        AnalysisOrchestrator(config).run()
    elif config['mode'] == 'forecast':
        from StatkraftAssessment.forecast import ForecastOrchestrator
        ForecastOrchestrator(config).run()
