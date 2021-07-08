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
from aggregators.FedBE import FedBEAggregator
from aggregators.FedDF import FedDFAggregator


class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super().__init__()
        
        self.nonIID = True
        self.alphaDirichlet = 0.1 # For sampling
        self.serverData = 1.0/6
        
        if self.nonIID:
            iidString = f'non-IID alpha={self.alphaDirichlet}'
        else:
            iidString = 'IID'

        experimentString = 'Fed-Avg-BE-DF'
        
        self.scenarios: AttacksType = [
            ([], [], [], f"no_attack {iidString} {experimentString}"),
            ([2, ], [], [], f"1_faulty {iidString} {experimentString}"),
            #([2, 5], [], [], f"2_faulty {iidString}"),
            #([2, 5, 8], [], [], f"3_faulty {iidString}"),
            #([2, 5, 8, 11], [], [], f"4_faulty {iidString}"),
            #([2, 5, 8, 11, 14], [], [], f"5_faulty {iidString}"),
            #([2, 5, 8, 11, 14, 17], [], [], f"6_faulty {iidString}"),
            #([2, 5, 8, 11, 14, 17, 20], [], [], f"7_faulty {iidString}"),
            ([2, 5, 8, 11, 14, 17, 20, 23], [], [], f"8_faulty {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20, 23, 26], [], [], f"9_faulty {iidString}"),
            #([2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], [], f"10_faulty {iidString}"),
            ([], [2, ], [], f"1_mal {iidString} {experimentString}"),
            #([], [2, 5], [], f"2_mal {iidString} Gaussian"),
            #([], [2, 5, 8], [], f"3_mal {iidString} Gaussian"),
            #([], [2, 5, 8, 11], [], f"4_mal {iidString} Gaussian"),
#           ([], [2, 5, 8, 11, 14], [], f"5_mal {iidString} Gaussian"),
            #([], [2, 5, 8, 11, 14, 17], [], f"6_mal {iidString} Gaussian"),
            #([], [2, 5, 8, 11, 14, 17, 20], [], f"7_mal {iidString} Gaussian"),
            ([], [2, 5, 8, 11, 14, 17, 20, 23], [], f"8_mal {iidString} {experimentString}"),
            #([], [2, 5, 8, 11, 14, 17, 20, 23, 26], [], f"9_mal {iidString} Gaussian"),
#           ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], f"10_mal {iidString} Gaussian"),
            # ([], [2], [], "1_mal"),
            # ([], [2, 5], [], "2_mal"),
            # ([], [2, 5, 8], [], "3_mal"),
            # ([], [2, 5, 8, 11], [], "4_mal"),
        ]
        self.percUsers = torch.tensor(PERC_USERS, device=self.aggregatorConfig.device)

        self.aggregators = [FAAggregator, FedBEAggregator, FedDFAggregator, ]
        #self.aggregators = [FAAggregator, COMEDAggregator, MKRUMAggregator, AFAAggregator, FedMGDAPlusPlusAggregator, FedBEAggregator]
        

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
