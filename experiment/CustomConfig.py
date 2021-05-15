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
            # ([], [], [3, 6, 9], "3_free"),
            # ([], [], [3, 6, 9, 12], "4_free"),
            # ([], [], [3, 6, 9, 12, 15], "5_free"),
            # ([], [], [3, 6, 9, 12, 15, 18], "6_free"),
            # ([], [], [3, 6, 9, 12, 15, 18, 21], "7_free"),
            # ([], [], [3, 6, 9, 12, 15, 18, 21, 24], "8_free"),
            # ([], [], [3, 6, 9, 12, 15, 18, 21, 24, 27], "9_free"),
            ([], [], [3, 6, 9, 12, 15, 18, 21, 24, 27, 30], "10_free"),
            ([], [], [], "nothing"),
        )


    def scenario_conversion(self):
        for faulty, malicious, freeRider, attackName in self.scenarios:

            self.faulty = faulty
            self.malicious = malicious
            self.freeRiding = freeRider

            yield attackName
