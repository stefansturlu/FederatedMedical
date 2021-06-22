from aggregators.COMED import COMEDAggregator
from aggregators.MKRUM import MKRUMAggregator
from experiment.AggregatorConfig import AggregatorConfig
from aggregators.FedAvg import FAAggregator
import torch
from aggregators.Aggregator import Aggregator, allAggregators
from typing import List, Tuple, Type, Union
import torch.optim as optim
import torch.nn as nn


class DefaultExperimentConfiguration:
    """
    Base configuration for the federated learning setup.
    """

    def __init__(self):
        # DEFAULT PARAMETERS
        self.name: str = ""

        self.aggregatorConfig = AggregatorConfig()

        # Epochs num locally run by clients before sending back the model update
        self.epochs: int = 2

        self.batchSize: int = 200  # Local training  batch size
        self.learningRate: float = 0.1
        self.Loss = nn.CrossEntropyLoss
        self.Optimizer: Type[optim.Optimizer] = optim.SGD

        # Big datasets size tuning param: (trainSize, testSize); (None, None) interpreted as full dataset
        self.datasetSize: Tuple[int, int] = (0, 0)

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

        # Client privacy preserving module setup
        self.privacyPreserve: Union[bool, None] = False  # if None, run with AND without DP
        self.releaseProportion: float = 0.1
        self.epsilon1: float = 1
        self.epsilon3: float = 1
        self.needClip: bool = False
        self.clipValue: float = 0.001
        self.needNormalization: bool = False

        # Anonymization of datasets for k-anonymity
        self.requireDatasetAnonymization: bool = False

        self.aggregators: List[Type[Aggregator]] = allAggregators()  # Aggregation strategies

        self.plotResults: bool = True

        # Group-Wise config
        self.internalAggregator: Union[
            Type[FAAggregator], Type[MKRUMAggregator], Type[COMEDAggregator]
        ] = FAAggregator
        self.externalAggregator: Union[
            Type[FAAggregator], Type[MKRUMAggregator], Type[COMEDAggregator]
        ] = COMEDAggregator
            
        # Data splitting config
        self.nonIID = False
        self.alphaDirichlet = 0.1 # Parameter for Dirichlet sampling
        self.serverData = 0 # Used for distillation.
