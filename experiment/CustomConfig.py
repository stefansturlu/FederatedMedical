from utils.typings import AttacksType, FreeRiderAttack
from aggregators.Aggregator import allAggregators
from typing import List
import torch
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
# Naked imports for allAggregators function
from aggregators.FedAvg import FAAggregator
from aggregators.COMED import COMEDAggregator
from aggregators.MKRUM import MKRUMAggregator
from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAPlusPlus import FedMGDAPlusPlusAggregator
from aggregators.FedPADRC import FedPADRCAggregator


class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super().__init__()
        self.scenarios: AttacksType = [
            # # ([], [], [], "no_attack"),
            # ([], [2], [], "1_mal"),
            # ([], [2, 5], [], "2_mal"),
            # ([], [2, 5, 8], [], "3_mal"),
            # # ([], [2, 5, 8, 11], [], "4_mal"),
            # ([], [2, 5, 8, 11, 14], [], "5_mal"),
            # # ([], [2, 5, 8, 11, 14, 17], [], "6_mal"),
            # ([], [2, 5, 8, 11, 14, 17, 20], [], "7_mal"),
            ([], [2, 5, 8, 11, 14, 17, 20, 23], [], "8_mal"),
            # # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26], [], "9_mal"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], "10_mal"),
            # # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0], [], "11_mal"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3], [], "12_mal"),
            # # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6], [], "13_mal"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9], [], "14_mal"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27], [], "20_mal"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 25], [], "22_mal"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 25, 22, 19], [], "24_mal"),
            # ([], [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 25, 22, 19, 16, 13], "26_free"),
        ]
        self.percUsers = torch.tensor(PERC_USERS, device=self.aggregatorConfig.device)

    def scenario_conversion(self):
        """
        Sets the faulty, malicious and free-riding clients appropriately.

        Sets the config's and aggregatorConfig's names to be the attackName.
        """
        for faulty, malicious, freeRider, attackName in self.scenarios:

            self.faulty = faulty
            self.malicious = malicious
            self.freeRiding = freeRider
            self.name = attackName
            self.aggregatorConfig.attackName = attackName

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
