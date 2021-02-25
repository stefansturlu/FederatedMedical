import torch
from aggregators.Aggregator import Aggregator, allAggregators
from typing import List


class DefaultExperimentConfiguration:
    def __init__(self):
        # DEFAULT PARAMETERS
        self.name: str = None

        # Federated learning parameters
        self.rounds: int = 50  # Total number of training rounds
        self.epochs: int = (
            2  # Epochs num locally run by clients before sending back the model update
        )
        self.batchSize: int = 200  # Local training  batch size
        self.learningRate: float = 0.1
        self.Loss = torch.nn.CrossEntropyLoss
        self.Optimizer = torch.optim.SGD

        # Big datasets size tuning param: (trainSize, testSize); (None, None) interpreted as full dataset
        self.datasetSize = (None, None)

        # Clients setup
        self.percUsers = torch.tensor(
            [0.2, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1]
        )  # Client data partition
        self.labels = torch.tensor(range(10))  # Considered dataset labels
        self.faulty: List[int] = []  # List of noisy clients
        self.malicious: List[int] = []  # List of (malicious) clients with flipped labels

        # AFA Parameters:
        self.alpha = 3
        self.beta = 3
        self.xi = 2
        self.deltaXi = 0.5

        # FedMGDA+ Parameters:
        self.threshold = 0.001
        self.innerLR = 0.001

        # Client privacy preserving module setup
        self.privacyPreserve: bool = False  # if None, run with AND without DP
        self.releaseProportion: float = 0.1
        self.epsilon1: float = 1
        self.epsilon3: float = 1
        self.needClip: bool = False
        self.clipValue: float = 0.001
        self.needNormalization: bool = False

        # Anonymization of datasets for k-anonymity
        self.requireDatasetAnonymization: bool = False

        self.aggregators: List[Aggregator] = allAggregators()  # Aggregation strategies

        self.plotResults: bool = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
