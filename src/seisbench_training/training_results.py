from pathlib import Path


class TrainedModel:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint


class SeisbenchTrainingResults:
    def __init__(self, path: Path):
        self.path = path

    def get_best_model(self) -> TrainedModel:
        pass
