from typing import List, NewType, Tuple
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration

MaliciousList = List[int]
FaultyList = List[int]
FreeRiderList = List[int]
AttackName = str
AttacksType = List[Tuple[FaultyList, MaliciousList, FreeRiderList, AttackName]]


class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        self.scenarios: AttacksType = [
            ([], [2], "1_malicious"),
            ([], [2, 4], "2_malicious"),
            ([], [2, 4, 6], "3_malicious"),
            ([], [2, 4, 6, 8], "4_malicious"),
            ([], [2, 4, 6, 8, 10], "5_malicious"),
            ([], [2, 4, 6, 8, 10, 12], "6_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14], "7_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16], "8_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16, 18], "9_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "10_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], "11_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], "12_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26], "13_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], "14_malicious"),
            ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], "15_malicious"),
            ([1], [], "1_faulty"),
            ([1, 3], [], "2_faulty"),
            ([1, 3, 5], [], "3_faulty"),
            ([1, 3, 5, 7], [], "4_faulty"),
            ([1, 3, 5, 7, 9], [], "5_faulty"),
            ([1, 3, 5, 7, 9, 11], [], "6_faulty"),
            ([1, 3, 5, 7, 9, 11, 13], [], "7_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15], [], "8_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17], [], "9_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], [], "10_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21], [], "11_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], [], "12_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25], [], "13_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27], [], "14_faulty"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], [], "15_faulty"),
            ([1], [2], "1_faulty, 1_malicious"),
            ([1, 3], [2, 4], "2_faulty, 2_malicious"),
            ([1, 3, 5], [2, 4, 6], "3_faulty, 3_malicious"),
            ([1, 3, 5, 7], [2, 4, 6, 8], "4_faulty, 4_malicious"),
            ([1, 3, 5, 7, 9], [2, 4, 6, 8, 10], "5_faulty, 5_malicious"),
            ([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12], "6_faulty, 6_malicious"),
            ([1, 3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12, 14], "7_faulty, 7_malicious"),
            ([1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16], "8_faulty, 8_malicious"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17], [2, 4, 6, 8, 10, 12, 14, 16, 18], "9_faulty, 9_malicious"),
            ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "10_faulty, 10_malicious"),
        ],


    def scenario_conversion(self):
        for scenario in self.scenarios:
            faulty, malicious, freeRider, attackName = scenario

            self.faulty = faulty
            self.malicious = malicious
            self.freeRiding = freeRider

            yield attackName
