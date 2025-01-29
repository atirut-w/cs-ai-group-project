from typing import Optional
from abc import abstractmethod


class Module:
    def __init__(self):
        # self.name: Optional[str] = None
        self.description: Optional[str] = None

    @abstractmethod
    def prepare_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def train_model(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self):
        raise NotImplementedError
