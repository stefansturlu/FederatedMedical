from utils.typings import FreeRiderAttack
from torch import device, cuda

class AggregatorConfig:
    def __init__(self):

        # Total number of training rounds
        self.rounds: int = 30

        self.device = device("cuda" if cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        # Pipeline config
        self.detectFreeRiders: bool = False
        self.freeRiderAttack: FreeRiderAttack = FreeRiderAttack.BASIC

        # Privacy Amplification settings
        self.privacyAmplification = False
        self.amplificationP = 0.3

        # FedMGDA+ Parameters:
        self.innerLR: float = 0.1

        # AFA Parameters:
        self.xi: float = 2
        self.deltaXi: float = 0.25
