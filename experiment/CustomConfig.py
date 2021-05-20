from typing import List, Tuple
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration

MaliciousList = List[int]
FaultyList = List[int]
FreeRiderList = List[int]
AttackName = str
AttacksType = List[Tuple[FaultyList, MaliciousList, FreeRiderList, AttackName]]


class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super(CustomConfig, self).__init__()
        self.scenarios: AttacksType = (
            # ([], [], [3], "1_free"),
            # ([], [], [3, 6], "2_free"),
            # ([1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16], [], "3_free"),
            # ([], [], [3, 6, 9, 12], "4_free"),
            # ([], [], [3, 6, 9, 12, 15], "5_free"),
            # ([], [], [3, 6, 9, 12, 15, 18], "6_free"),
            # ([], [], [3, 6, 9, 12, 15, 18, 21], "7_free"),
            # ([], [], [3, 6, 9, 12, 15, 18, 21, 24], "8_free"),
            # ([], [], [3, 6, 9, 12, 15, 18, 21, 24, 27], "9_free"),
            # ([], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], "10_free"),
            ([], [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20], [], "20_mal"),
            # ([], [], [], "nothing"),
        )

        self.epochs = 10



    def scenario_conversion(self):
        for faulty, malicious, freeRider, attackName in self.scenarios:

            self.faulty = faulty
            self.malicious = malicious
            self.freeRiding = freeRider

            yield attackName