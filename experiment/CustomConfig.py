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
from aggregators.FedMGDAPlus import FedMGDAPlusAggregator
from aggregators.FedMGDAPlusPlus import FedMGDAPlusPlusAggregator
from aggregators.FedPADRC import FedPADRCAggregator
from aggregators.FedPADRC import FedPADRCAggregator
from aggregators.FedBE import FedBEAggregator


class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super().__init__()
        
        self.nonIID = False
        self.alphaDirichlet = 0.5
        self.serverData = 1/6
        
        self.scenarios: AttacksType = [
            ([], [], [], "no_attack IID FedBE"),
            #([2, ], [], [], f"GPU 1_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5], [], [], f"2_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8], [], [], f"3_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8, 11], [], [], f"4_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8, 11, 14], [], [], f"5_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8, 11, 14, 17], [], [], f"6_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8, 11, 14, 17, 20], [], [], f"7_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8, 11, 14, 17, 20, 23], [], [], f"8_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8, 11, 14, 17, 20, 23, 26], [], [], f"9_faulty non-IID alpha={self.alphaDirichlet}"),
            #([2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], [], f"10_faulty non-IID alpha={self.alphaDirichlet}"),
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

        self.aggregators = [FedBEAggregator]
        

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
