from utils.typings import AttacksType
from aggregators.Aggregator import allAggregators
from aggregators.GroupWise import GroupWiseAggregation
from typing import List
import torch
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
# Naked imports for allAggregators function
from aggregators.FedAvg import FAAggregator
from aggregators.COMED import COMEDAggregator
from aggregators.MKRUM import MKRUMAggregator
from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAPlus import FedMGDAPlusAggregator


class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super().__init__()
        self.scenarios: AttacksType = [
            # ([], [], [], "no_free"),
            # ([], [], [2], "1_free"),
            # ([], [], [2, 5], "2_free"),
            # ([], [], [2, 5, 8], "3_free"),
            # ([], [], [2, 5, 8, 11], "4_free"),
            # ([], [], [2, 5, 8, 11, 14], "5_free"),
            # ([], [], [2, 5, 8, 11, 14, 17], "6_free"),
            # ([], [], [2, 5, 8, 11, 14, 17, 20], "7_free"),
            # ([], [], [2, 5, 8, 11, 14, 17, 20, 23], "8_free"),
            # ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26], "9_free"),
            # ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], "10_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0], "11_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3], "12_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6], "13_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9], "14_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12], "15_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15], "16_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18], "17_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21], "18_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24], "19_free"),
            ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27], "20_free"),
        ]
        self.percUsers = torch.tensor(PERC_USERS, device=self.aggregatorConfig.device)

        self.aggregatorConfig.detectFreeRiders = True
        # self.aggregatorConfig.privacyAmplification = True

        self.aggregators = [FAAggregator, COMEDAggregator, MKRUMAggregator, AFAAggregator, FedMGDAPlusAggregator]

    def scenario_conversion(self):
        for faulty, malicious, freeRider, attackName in self.scenarios:

            self.faulty = faulty
            self.malicious = malicious
            self.freeRiding = freeRider
            self.name = attackName

            yield attackName


# Determines how much data each client gets (normalised)
PERC_USERS: List[float] = [
    0.1,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.1,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.1,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.1,
    0.15,
    0.2,
    0.2,
]
