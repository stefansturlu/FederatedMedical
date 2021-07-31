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
from aggregators.FedMGDAPlusDF import FedMGDAPlusDFAggregator
from aggregators.FedPADRC import FedPADRCAggregator
from aggregators.FedBE import FedBEAggregator
from aggregators.FedDF import FedDFAggregator
from aggregators.FedABE import FedABEAggregator
from aggregators.FedABED import FedABEDAggregator
from aggregators.FedADF import FedADFAggregator
from aggregators.FedDFmed import FedDFmedAggregator
from aggregators.FedRAD import FedRADAggregator


class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super().__init__()
        
        self.nonIID = False
        self.alphaDirichlet = 0.5 # For sampling
        self.serverData = 1.0/6
        
        if self.nonIID:
            iidString = f'non-IID alpha={self.alphaDirichlet}'
        else:
            iidString = 'IID'

        experimentString = 'everything'
        
        self.scenarios: AttacksType = [
            ([], [], [], f"no_attack {iidString} {experimentString}"),
            ([2, ], [], [], f"faulty_1 {iidString} {experimentString}"),
            #([2, 5], [], [], f"faulty_2 {iidString} {experimentString}"),
            #([2, 5, 8], [], [], f"faulty_3 {iidString} {experimentString}"),
            #([2, 5, 8, 11], [], [], f"faulty_4 {iidString} {experimentString}"),
            ([2, 5, 8, 11, 14], [], [], f"faulty_5 {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17], [], [], f"faulty_6 {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20], [], [], f"faulty_7 {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20, 23], [], [], f"faulty_8 {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20, 23, 26], [], [], f"faulty_9 {iidString} {experimentString}"),
            ([2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], [], f"faulty_10 {iidString} {experimentString}"),
            ([], [2, ], [], f"mal_1 {iidString} {experimentString}"),
            #([], [2, 5], [], f"mal_2 {iidString} {experimentString}"),
            #([], [2, 5, 8], [], f"mal_3 {iidString} {experimentString}"),
            #([], [2, 5, 8, 11], [], f"mal_4 {iidString} {experimentString}"),
            ([], [2, 5, 8, 11, 14], [], f"mal_5 {iidString} {experimentString}"),
            #([], [2, 5, 8, 11, 14, 17], [], f"mal_6 {iidString} {experimentString}"),
            #([], [2, 5, 8, 11, 14, 17, 20], [], f"mal_7 {iidString} {experimentString}"),
            #([], [2, 5, 8, 11, 14, 17, 20, 23], [], f"mal_8 {iidString} {experimentString}"),
            #([], [2, 5, 8, 11, 14, 17, 20, 23, 26], [], f"mal_2 {iidString} {experimentString}"),
            ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], f"mal_10 {iidString} {experimentString}"),
            #([], [1,2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], f"mal_11 {iidString} {experimentString}"),
            #([], [1,2, 4,5, 8, 11, 14, 17, 20, 23, 26, 29], [], f"mal_12 {iidString} {experimentString}"),
            #([], [1,2, 4,5, 7,8, 11, 14, 17, 20, 23, 26, 29], [], f"mal_13 {iidString} {experimentString}"),
            #([], [1,2, 4,5, 7,8, 10,11, 14, 17, 20, 23, 26, 29], [], f"mal_14 {iidString} {experimentString}"),
            #([], [1,2, 4,5, 7,8, 11, 13,14, 16,17, 20, 23, 26, 29], [], f"mal_15 {iidString} {experimentString}"),
            #([], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 28,29], [], f"mal_15 {iidString} {experimentString}"),
            ([2, 5, ], [17, 20, ], [], f"dual_faulty2_mal2 {iidString} {experimentString}"),
            ([1, 4, 7, 10, 13], [17, 20, 23, 26, 29], [], f"dual_faulty5_mal5 {iidString} {experimentString}"),
        ]
            
        self.percUsers = torch.tensor(PERC_USERS, device=self.aggregatorConfig.device)

        #self.aggregators = [FedRADAggregator, FedABEDAggregator, ]
        self.aggregators = [FAAggregator, COMEDAggregator, MKRUMAggregator, AFAAggregator, FedMGDAPlusPlusAggregator, 
                            FedDFAggregator, FedDFmedAggregator, FedADFAggregator, FedMGDAPlusDFAggregator, 
                            FedBEAggregator, 
                            FedRADAggregator, FedABEDAggregator]
        

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
