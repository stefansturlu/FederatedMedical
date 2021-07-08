from utils.typings import FreeRiderAttack, PersonalisationMethod
from torch import device, cuda


class AggregatorConfig:
    """
    Configuration for the aggregators.

    Use this for information that you want the aggregator to know about.
    """

    def __init__(self):

        # Total number of training rounds
        self.rounds: int = 30

        self.device = device("cuda" if cuda.is_available() else "cpu")
        #self.device = device("cpu")

        # Name of attack being employed
        self.attackName = ""

        # Pipeline config
        self.detectFreeRiders: bool = False
        self.freeRiderAttack: FreeRiderAttack = FreeRiderAttack.BASIC

        # Privacy Amplification settings  (Sets how many clients are sampled)
        self.privacyAmplification = False
        self.amplificationP = 0.3

        # FedMGDA+ Parameters:
        self.innerLR: float = 0.1

        # AFA Parameters:
        self.xi: float = 2
        self.deltaXi: float = 0.25

        # Clustering Config:
        self.cluster_count: int = 5

        self.personalisation: PersonalisationMethod = PersonalisationMethod.NO_GLOBAL
        self.threshold: bool = False
            
            
        # FedBE Parameters
        self.sampleSize = 15
        self.samplingMethod = 'dirichlet' # gaussian, dirichlet, dirichlet_elementwise
        self.samplingDirichletAlpha = 0.1

