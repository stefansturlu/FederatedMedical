from torch import Tensor
from typing import NewType, Dict, List, Tuple
from enum import Enum

# New Types
Errors = NewType("Errors", Tensor)
IdRoundPair = NewType("IdRoundPair", Tuple[int, int])
BlockedLocations = NewType("BlockedLocations", Dict[str, List[IdRoundPair]])


# Type Aliases
MaliciousList = List[int]
FaultyList = List[int]
FreeRiderList = List[int]
AttackName = str
AttacksType = List[Tuple[FaultyList, MaliciousList, FreeRiderList, AttackName]]


# Enum Classes
class FreeRiderAttack(Enum):
    BASIC=0
    NOISY=1
    DELTA=2