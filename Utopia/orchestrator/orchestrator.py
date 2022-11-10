from dataclasses import dataclass
from abc import abstractmethod

from Utopia.config.config import Config

@dataclass
class Orchestrator:
    config: Config

    def __post_init__(self):
        return

    @abstractmethod
    def run(self):
        raise NotImplementedError('Missing run method')


    def log(self, msg: str) -> None:
        print(msg)

    def log_headline(self, msg: str) -> None:
        print(f'>>> {msg}')
