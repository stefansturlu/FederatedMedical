from aggregators.FedAvg import FAAggregator
import torch
from aggregators.Aggregator import Aggregator, allAggregators
from typing import List, Tuple, Union
import torch.optim as optim
import torch.nn as nn


class DefaultExperimentConfiguration:
    def __init__(self):
        # DEFAULT PARAMETERS
        self.name: str = None

        # Federated learning parameters
        self.rounds: int = 30  # Total number of training rounds
        self.epochs: int = (
            2  # Epochs num locally run by clients before sending back the model update
        )
        self.batchSize: int = 200  # Local training  batch size
        self.learningRate: float = 0.1
        self.Loss = nn.CrossEntropyLoss
        self.Optimizer: optim.Optimizer = torch.optim.SGD

        # Big datasets size tuning param: (trainSize, testSize); (None, None) interpreted as full dataset
        self.datasetSize: Tuple[int, int] = (None, None)

        # Clients setup
        self.percUsers = torch.tensor(
            [0.2, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1]
        )  # Client data partition
        self.labels = torch.tensor(range(10))  # Considered dataset labels
        self.faulty: List[int] = []  # List of noisy clients
        self.malicious: List[int] = []  # List of (malicious) clients with flipped labels
        self.freeRiding: List[int] = []  # List of free-riding clients

        # AFA Parameters:
        self.alpha: float = 4
        self.beta: float = 4
        self.xi: float = 2
        self.deltaXi: float = 0.25

        # FedMGDA+ Parameters:
        self.innerLR: float = 0.1

        # Client privacy preserving module setup
        self.privacyPreserve: Union[bool, None] = False  # if None, run with AND without DP
        self.releaseProportion: float = 0.1
        self.epsilon1: float = 1
        self.epsilon3: float = 1
        self.needClip: bool = False
        self.clipValue: float = 0.001
        self.needNormalization: bool = False
        self.privacyAmplification = False
        self.amplificationP = 0.3

        # Anonymization of datasets for k-anonymity
        self.requireDatasetAnonymization: bool = False

        self.aggregators: List[Aggregator] = allAggregators()  # Aggregation strategies

        self.plotResults: bool = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        # Pipeline config
        self.freeRiderDetect: bool = False
        self.clustering: bool = False

        # Group-Wise config
        self.internalAggregator: Aggregator = FAAggregator
        self.externalAggregator: Aggregator = FAAggregator
