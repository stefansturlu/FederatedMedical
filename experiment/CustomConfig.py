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
            ([], [], [], "no_attack IID"),
            ([], [2, ], [], "1_mal IID"),
            ([], [2, 5], [], "2_mal IID"),
            ([], [2, 5, 8], [], "3_mal IID"),
            ([], [2, 5, 8, 11], [], "4_mal IID"),
            ([], [2, 5, 8, 11, 14], [], "5_mal IID"),
            ([], [2, 5, 8, 11, 14, 17], [], "6_mal IID"),
            ([], [2, 5, 8, 11, 14, 17, 20], [], "7_mal IID"),
            ([], [2, 5, 8, 11, 14, 17, 20, 23], [], "8_mal IID"),
            ([], [2, 5, 8, 11, 14, 17, 20, 23, 26], [], "9_mal IID"),
            ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], "10_mal IID"),
            # ([], [], [], "no_attack non-IID alpha=1.0"),
            # ([], [2, ], [], "1_mal non-IID alpha=1.0"),
            # ([], [2, 5], [], "2_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8], [], "3_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8, 11], [], "4_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8, 11, 14], [], "5_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8, 11, 14, 17], [], "6_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8, 11, 14, 17, 20], [], "7_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23], [], "8_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26], [], "9_mal non-IID alpha=1.0"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], "10_mal non-IID alpha=1.0"),
            # ([], [2], [], "1_mal"),
            # ([], [2, 5], [], "2_mal"),
            # ([], [2, 5, 8], [], "3_mal"),
            # ([], [2, 5, 8, 11], [], "4_mal"),
        ]
        self.percUsers = torch.tensor(PERC_USERS, device=self.aggregatorConfig.device)

        self.aggregators = [FAAggregator, COMEDAggregator, MKRUMAggregator, AFAAggregator, FedMGDAPlusAggregator]
        
        self.nonIID = False
        self.alphaDirichlet = 1.0

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
