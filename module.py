from typing import Optional


class Module:
    def __init__(self):
        # self.name: Optional[str] = None
        self.description: Optional[str] = None

    def prepare_dataset(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def evaluate_model(self):
        raise NotImplementedError
