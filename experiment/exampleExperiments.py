from experiment.CustomConfig import CustomConfig
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from itertools import product
import torch

from aggregators.Aggregator import allAggregators
from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAPlus import FedMGDAPlusAggregator
# Naked imports for allAggregators function
from aggregators.FedAvg import FAAggregator
from aggregators.COMED import COMEDAggregator
from aggregators.MKRUM import MKRUMAggregator

from main import __experimentOnMNIST, experiment, PERC_USERS, __experimentOnCOVIDx


@experiment
def noDP_noByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    __experimentOnMNIST(configuration, "MNIST", "mnist")


@experiment
def withoutDP_withByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1])
    configuration.labels = torch.tensor([0, 2, 5, 8])
    configuration.faulty = [2, 6]
    configuration.malicious = [1]

    __experimentOnMNIST(configuration, "MNIST - Byzantine Clients", "mnist_byz")


@experiment
def withDP_noByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    configuration.faulty = []
    configuration.malicious = []
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration, "MNIST - DP", "mnist_dp")


@experiment
def withAndWithoutDP_noByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.privacyPreserve = None

    __experimentOnMNIST(configuration)


@experiment
def withDP_withByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1])
    configuration.labels = torch.tensor([0, 2, 5, 8])
    configuration.faulty = [2, 6]
    configuration.malicious = [1]
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration, "MNIST - Byzantine Clients with DP", "mnist_byz_dp")


@experiment
def withDP_fewNotByzClient_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor([0.3, 0.25, 0.45])
    configuration.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration)


@experiment
def noDP_30notByzClients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor(PERC_USERS)

    __experimentOnMNIST(configuration)


@experiment
def withDP_30Clients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor(PERC_USERS)
    configuration.privacyPreserve = True

    __experimentOnMNIST(configuration)


@experiment
def withAndWithoutDP_30notByzClients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor(PERC_USERS)
    configuration.privacyPreserve = None

    __experimentOnMNIST(configuration)


@experiment
def withAndWithoutDP_30withByzClients_onMNIST():
    configuration = DefaultExperimentConfiguration()

    configuration.percUsers = torch.tensor(PERC_USERS)
    configuration.faulty = [2, 10, 13]
    configuration.malicious = [15, 18]
    configuration.privacyPreserve = None
    configuration.rounds = 7
    configuration.plotResults = True
    __experimentOnMNIST(configuration)


@experiment
def noDP_noByzClient_fewRounds_onMNIST():
    configuration = DefaultExperimentConfiguration()
    configuration.rounds = 3
    configuration.plotResults = True
    __experimentOnMNIST(configuration)


@experiment
def withMultipleDPconfigsAndWithout_30notByzClients_onMNIST():
    releaseProportion = {0.1, 0.4}
    epsilon1 = {1, 0.01, 0.0001}
    epsilon3 = {1, 0.01, 0.0001}
    clipValues = {0.01, 0.0001}
    needClip = {False, True}
    needNormalise = {False, True}

    percUsers = torch.tensor(PERC_USERS)
    # Without DP
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = allAggregators()
    noDPconfig.percUsers = percUsers
    __experimentOnMNIST(noDPconfig)

    # With DP
    for config in product(
        needClip, clipValues, epsilon1, epsilon3, needNormalise, releaseProportion
    ):
        (
            needClip,
            clipValues,
            epsilon1,
            epsilon3,
            needNormalise,
            releaseProportion,
        ) = config

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = allAggregators()

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.needNormalise = needNormalise
        expConfig.clipValues = clipValues
        expConfig.needClip = needClip
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3

        __experimentOnMNIST(expConfig)


@experiment
def Aggregator_Limitations_Test_MNIST():
    percUsers = torch.tensor(PERC_USERS)

    config = CustomConfig()
    config.aggregators = allAggregators()
    config.percUsers = percUsers
    config.innerLR = 0.1

    for attackName in config.scenario_conversion():

        errors = __experimentOnMNIST(
            config,
            title=f"Aggregator Limitations Test MNIST \n Attacks: {attackName}",
            filename=f"{attackName}",
            folder="MNIST_Limitations",
        )



@experiment
def withMultipleDPandByzConfigsAndWithout_30ByzClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, "low"), (0.4, 1, 1, "high")]

    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [
        ([1], [], "1_faulty"),
        ([], [2], "1_malicious"),
        ([1], [2], "1_faulty, 1_malicious"),
        ([1, 3, 5, 7, 9], [2, 4, 6, 8, 10], "5_faulty, 5_malicious"),
        ([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], [], "10_faulty"),
        ([], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "10_malicious"),
        (
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            "15_faulty, 14_malicious",
        ),
    ]

    percUsers = torch.tensor(PERC_USERS)

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = allAggregators()
    noDPconfig.percUsers = percUsers
    __experimentOnMNIST(
        noDPconfig, title="MNIST - 30 Clients", filename="mnist_30", folder="mnist_experiments"
    )

    # Without DP
    for scenario in attacks:
        faulty, malicious, attackName = scenario
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = allAggregators()
        noDPconfig.percUsers = percUsers

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        # noDPconfig.name = "altered:{}".format(attackName)

        __experimentOnMNIST(
            noDPconfig,
            title=f"MNIST - Byzantine Clients, 30 Clients, With Attacks (Flipping - {attackName})",
            filename=f"mnist_byz_30_attacks_{attackName}",
            folder="mnist_experiments",
        )

    # With DP
    for budget, attack in product(privacyBudget, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = allAggregators()

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        # expConfig.name = f" altered:{attackName};"
        # expConfig.name += f" privacyBudget:{budgetName};"

        __experimentOnMNIST(
            expConfig,
            title=f"MNIST - Byzantine Clients with DP (budget: {budgetName}), 30 Clients, With Attacks ({attackName})",
            filename=f"mnist_byz_dp_budget_{budgetName}_30_attacks_{attackName}",
            folder="mnist_experiments",
        )


@experiment
def byz_FedMGDA_MNIST():
    config = DefaultExperimentConfiguration()
    config.rounds = 70

    config.percUsers = torch.tensor(PERC_USERS)

    attacks = [
        ([len(config.percUsers) - (2 * i + 1) for i in range(1)], [], "1_faulty"),
        ([len(config.percUsers) - (2 * i + 1) for i in range(2)], [], "2_faulty"),
        ([], [len(config.percUsers) - (2 * i + 2) for i in range(1)], "1_malicious"),
        ([], [len(config.percUsers) - (2 * i + 2) for i in range(2)], "2_malicious"),
        (
            [len(config.percUsers) - (2 * i + 1) for i in range(1)],
            [len(config.percUsers) - (2 * i + 2) for i in range(1)],
            "1_faulty, 1_malicious",
        ),
        (
            [len(config.percUsers) - (2 * i + 1) for i in range(2)],
            [len(config.percUsers) - (2 * i + 2) for i in range(2)],
            "2_faulty, 2_malicious",
        ),
    ]
    config.aggregators = [FedMGDAPlusAggregator]
    __experimentOnMNIST(config, title=f"MNIST - 30 Clients, MGDA+", filename=f"mnist_30_MGDA+")

    for scenario in attacks:
        faulty, malicious, attackName = scenario
        config.aggregators = [FedMGDAPlusAggregator]

        config.faulty = faulty
        config.malicious = malicious
        config.name = " altered:{}".format(attackName)

        config.name += " FedMGDA"

        __experimentOnMNIST(
            config,
            title=f"MNIST - Byzantine Clients 30 Clients, MGDA+, With Attacks ({attackName})",
            filename=f"mnist_byz_30_MGDA+_attacks_{attackName}",
        )


@experiment
def withAndWithoutDP_AFA_30ByzAndNotClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, "low")]
    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [
        ([2 * i + 1 for i in range(2)], [], "2_faulty"),
        ([2 * i + 1 for i in range(4)], [], "4_faulty"),
        ([2 * i + 1 for i in range(6)], [], "6_faulty"),
        ([2 * i + 1 for i in range(8)], [], "8_faulty"),
        ([2 * i + 1 for i in range(9)], [], "9_faulty"),
        ([2 * i + 1 for i in range(10)], [], "10_faulty"),
        ([2 * i + 1 for i in range(12)], [], "12_faulty"),
        ([2 * i + 1 for i in range(14)], [], "14_faulty"),
        ([2 * i + 1 for i in range(15)], [], "15_faulty"),
        ([], [2 * i + 2 for i in range(2)], "2_malicious"),
        ([], [2 * i + 2 for i in range(4)], "4_malicious"),
        ([], [2 * i + 2 for i in range(6)], "6_malicious"),
        ([], [2 * i + 2 for i in range(8)], "8_malicious"),
        ([], [2 * i + 2 for i in range(9)], "9_malicious"),
        ([], [2 * i + 2 for i in range(10)], "10_malicious"),
        ([], [2 * i + 2 for i in range(12)], "12_malicious"),
        ([], [2 * i + 2 for i in range(14)], "14_malicious"),
        ([], [2 * i + 2 for i in range(15)], "15_malicious"),
        (
            [2 * i + 1 for i in range(1)],
            [2 * i + 2 for i in range(1)],
            "1_faulty,1_malicious",
        ),
        (
            [2 * i + 1 for i in range(2)],
            [2 * i + 2 for i in range(2)],
            "2_faulty,2_malicious",
        ),
        (
            [2 * i + 1 for i in range(3)],
            [2 * i + 2 for i in range(3)],
            "3_faulty,3_malicious",
        ),
        (
            [2 * i + 1 for i in range(4)],
            [2 * i + 2 for i in range(4)],
            "4_faulty,4_malicious",
        ),
        (
            [2 * i + 1 for i in range(5)],
            [2 * i + 2 for i in range(5)],
            "5_faulty,5_malicious",
        ),
        (
            [2 * i + 1 for i in range(6)],
            [2 * i + 2 for i in range(6)],
            "6_faulty,6_malicious",
        ),
        (
            [2 * i + 1 for i in range(7)],
            [2 * i + 2 for i in range(7)],
            "7_faulty,7_malicious",
        ),
        (
            [2 * i + 1 for i in range(8)],
            [2 * i + 2 for i in range(8)],
            "8_faulty,8_malicious",
        ),
    ]

    percUsers = torch.tensor(PERC_USERS)

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = [AFAAggregator]
    noDPconfig.percUsers = percUsers
    __experimentOnMNIST(noDPconfig)

    # Without DP
    for scenario in attacks:
        faulty, malicious, attackName = scenario
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = [AFAAggregator]
        noDPconfig.percUsers = percUsers

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        noDPconfig.name = "altered:{}".format(attackName)

        __experimentOnMNIST(noDPconfig)

    # With DP
    for budget, attack in product(privacyBudget, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = [AFAAggregator]

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        expConfig.name = "altered:{};".format(attackName)
        expConfig.name += "privacyBudget:{};".format(budgetName)

        __experimentOnMNIST(expConfig)


@experiment
def withAndWithoutDP_manyAlphaBetaAFA_30ByzAndNotClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, "low")]
    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [
        ([2 * i + 1 for i in range(2)], [], "2_faulty"),
        ([2 * i + 1 for i in range(6)], [], "6_faulty"),
        ([2 * i + 1 for i in range(8)], [], "8_faulty"),
        ([2 * i + 1 for i in range(10)], [], "10_faulty"),
        ([], [2 * i + 2 for i in range(2)], "2_malicious"),
        ([], [2 * i + 2 for i in range(6)], "6_malicious"),
        ([], [2 * i + 2 for i in range(8)], "8_malicious"),
        ([], [2 * i + 2 for i in range(10)], "10_malicious"),
        (
            [2 * i + 1 for i in range(1)],
            [2 * i + 2 for i in range(1)],
            "1_faulty,1_malicious",
        ),
        (
            [2 * i + 1 for i in range(4)],
            [2 * i + 2 for i in range(4)],
            "4_faulty,4_malicious",
        ),
    ]

    # # Workaround to run experiments in parallel runs:
    # e = 4  # experiment index
    # nAttacks = 2  # number of attack scenarios considered per experiement
    # attacks = attacks[e * nAttacks: e * nAttacks + nAttacks]

    alphaBetas = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]

    percUsers = torch.tensor(PERC_USERS)

    # Without DP without attacks
    for alphaBeta in alphaBetas:
        alpha, beta = alphaBeta

        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = [AFAAggregator]
        noDPconfig.percUsers = percUsers

        noDPconfig.alpha = alpha
        noDPconfig.beta = beta

        noDPconfig.name = "alphaBeta:{};".format(alphaBeta)

        __experimentOnMNIST(noDPconfig)

    # Without DP
    for alphaBeta, attack in product(alphaBetas, attacks):
        faulty, malicious, attackName = attack
        alpha, beta = alphaBeta
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = [AFAAggregator]
        noDPconfig.percUsers = percUsers

        noDPconfig.alpha = alpha
        noDPconfig.beta = beta

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        noDPconfig.name = "alphaBeta:{};".format(alphaBeta)
        noDPconfig.name += "altered:{};".format(attackName)

        __experimentOnMNIST(noDPconfig)

    # With DP
    for budget, alphaBeta, attack in product(privacyBudget, alphaBetas, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack
        alpha, beta = alphaBeta

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = [AFAAggregator]

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        expConfig.alpha = alpha
        expConfig.beta = beta

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        expConfig.name = "alphaBeta:{};".format(alphaBeta)
        expConfig.name += "altered:{};".format(attackName)
        expConfig.name += "privacyBudget:{};".format(budgetName)

        __experimentOnMNIST(expConfig)


@experiment
def withAndWithoutDP_manyXisAFA_30ByzAndNotClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, "low")]
    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [
        ([2 * i + 1 for i in range(2)], [], "2_faulty"),
        # ([2 * i + 1 for i in range(4)], [], "4_faulty"),
        ([2 * i + 1 for i in range(6)], [], "6_faulty"),
        # ([2 * i + 1 for i in range(7)], [], "7_faulty"),
        ([2 * i + 1 for i in range(8)], [], "8_faulty"),
        # ([2 * i + 1 for i in range(9)], [], "9_faulty"),
        ([2 * i + 1 for i in range(10)], [], "10_faulty"),
        # ([2 * i + 1 for i in range(12)], [], "12_faulty"),
        # ([2 * i + 1 for i in range(14)], [], "14_faulty"),
        # ([2 * i + 1 for i in range(15)], [], "15_faulty"),
        ([], [2 * i + 2 for i in range(2)], "2_malicious"),
        # ([], [2 * i + 2 for i in range(4)], "4_malicious"),
        ([], [2 * i + 2 for i in range(6)], "6_malicious"),
        # ([], [2 * i + 2 for i in range(7)], "7_malicious"),
        ([], [2 * i + 2 for i in range(8)], "8_malicious"),
        # ([], [2 * i + 2 for i in range(9)], "9_malicious"),
        ([], [2 * i + 2 for i in range(10)], "10_malicious"),
        # ([], [2 * i + 2 for i in range(12)], "12_malicious"),
        # ([], [2 * i + 2 for i in range(14)], "14_malicious"),
        # ([], [2 * i + 2 for i in range(15)], "15_malicious"),
        (
            [2 * i + 1 for i in range(1)],
            [2 * i + 2 for i in range(1)],
            "1_faulty,1_malicious",
        ),
        # ([2 * i + 1 for i in range(2)], [2 * i + 2 for i in range(2)], "2_faulty,2_malicious"),
        # ([2 * i + 1 for i in range(3)], [2 * i + 2 for i in range(3)], "3_faulty,3_malicious"),
        (
            [2 * i + 1 for i in range(4)],
            [2 * i + 2 for i in range(4)],
            "4_faulty,4_malicious",
        ),
        # ([2 * i + 1 for i in range(5)], [2 * i + 2 for i in range(5)], "5_faulty,5_malicious"),
        # ([2 * i + 1 for i in range(6)], [2 * i + 2 for i in range(6)], "6_faulty,6_malicious"),
        # ([2 * i + 1 for i in range(7)], [2 * i + 2 for i in range(7)], "7_faulty,7_malicious"),
        # ([2 * i + 1 for i in range(8)], [2 * i + 2 for i in range(8)], "8_faulty,8_malicious")
    ]

    # Workaround to run experiments in parallel runs:
    e = 4  # experiment index
    nAttacks = 2  # number of attack scenarios considered per experiement
    attacks = attacks[e * nAttacks : e * nAttacks + nAttacks]

    xis = [
        (1, 0.25),
        (1, 0.5),
        (1, 0.75),
        (2, 0.25),
        (2, 0.5),
        (2, 0.75),
        (3, 0.25),
        (3, 0.5),
        (3, 0.75),
    ]

    percUsers = torch.tensor(PERC_USERS)

    # Only run the vanilla experiment once
    if not e:
        # Without DP without attacks
        for xiTuple in xis:
            xi, deltaXi = xiTuple

            noDPconfig = DefaultExperimentConfiguration()
            noDPconfig.aggregators = [AFAAggregator]
            noDPconfig.percUsers = percUsers

            noDPconfig.xi = xi
            noDPconfig.deltaXi = deltaXi

            noDPconfig.name = "xis:{};".format(xiTuple)

            __experimentOnMNIST(noDPconfig)

    # Without DP
    for xiTuple, attack in product(xis, attacks):
        faulty, malicious, attackName = attack
        xi, deltaXi = xiTuple
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = [AFAAggregator]
        noDPconfig.percUsers = percUsers

        noDPconfig.xi = xi
        noDPconfig.deltaXi = deltaXi

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        noDPconfig.name = "xis:{};".format(xiTuple)
        noDPconfig.name += "altered:{};".format(attackName)

        __experimentOnMNIST(noDPconfig)

    # With DP
    for budget, xiTuple, attack in product(privacyBudget, xis, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack
        xi, deltaXi = xiTuple

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = [AFAAggregator]

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        noDPconfig.xi = xi
        noDPconfig.deltaXi = deltaXi

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        noDPconfig.name = "xis:{};".format(xiTuple)
        expConfig.name += "altered:{};".format(attackName)
        expConfig.name += "privacyBudget:{};".format(budgetName)

        __experimentOnMNIST(expConfig)


@experiment
def withLowAndHighAndWithoutDP_30ByzClients_onMNIST():
    # Privacy budget = (releaseProportion, epsilon1, epsilon3)
    privacyBudget = [(0.1, 0.0001, 0.0001, "low"), (0.4, 1, 1, "high")]

    # Attacks: Malicious/Flipping - flips labels to 0; Faulty/Byzantine - noisy
    attacks = [([1, 3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12, 14], "7_faulty,7_malicious")]

    percUsers = torch.tensor(PERC_USERS)

    # Without DP
    for scenario in attacks:
        faulty, malicious, attackName = scenario
        noDPconfig = DefaultExperimentConfiguration()
        noDPconfig.aggregators = allAggregators()
        noDPconfig.percUsers = percUsers

        noDPconfig.faulty = faulty
        noDPconfig.malicious = malicious
        noDPconfig.name = "altered:{}".format(attackName)

        __experimentOnMNIST(noDPconfig)

    # With DP
    for budget, attack in product(privacyBudget, attacks):
        releaseProportion, epsilon1, epsilon3, budgetName = budget
        faulty, malicious, attackName = attack

        expConfig = DefaultExperimentConfiguration()
        expConfig.percUsers = percUsers
        expConfig.aggregators = allAggregators()

        expConfig.privacyPreserve = True
        expConfig.releaseProportion = releaseProportion
        expConfig.epsilon1 = epsilon1
        expConfig.epsilon3 = epsilon3
        expConfig.needClip = True

        expConfig.faulty = faulty
        expConfig.malicious = malicious

        expConfig.name = "altered:{};".format(attackName)
        expConfig.name += "privacyBudget:{};".format(budgetName)

        __experimentOnMNIST(expConfig)


@experiment
def withAndWithoutDP_withAndWithoutByz_10ByzClients_onCOVIDx():
    epsilon1 = 0.0001
    epsilon3 = 0.0001
    releaseProportion = 0.1

    learningRate = 0.00002
    batchSize = 2
    rounds = 25

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = allAggregators()
    noDPconfig.learningRate = learningRate
    noDPconfig.batchSize = batchSize
    noDPconfig.rounds = rounds

    noDPconfig.percUsers = percUsers

    __experimentOnCOVIDx(noDPconfig)

    # With DP without attacks
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    noDPconfig.percUsers = percUsers

    __experimentOnCOVIDx(DPconfig)

    # With DP with one attacker
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    noDPconfig.percUsers = percUsers

    DPconfig.malicious = [3]
    DPconfig.name = "altered:1_malicious"

    __experimentOnCOVIDx(DPconfig)

    # With DP with more attackers
    DPbyzConfig = DefaultExperimentConfiguration()
    DPbyzConfig.aggregators = allAggregators()
    DPbyzConfig.learningRate = learningRate
    DPbyzConfig.batchSize = batchSize
    DPbyzConfig.rounds = rounds

    DPbyzConfig.privacyPreserve = True
    DPbyzConfig.releaseProportion = releaseProportion
    DPbyzConfig.epsilon1 = epsilon1
    DPbyzConfig.epsilon3 = epsilon3
    DPbyzConfig.needClip = True

    noDPconfig.percUsers = percUsers

    DPbyzConfig.faulty = [1]
    DPbyzConfig.malicious = [2, 4]

    DPbyzConfig.name = "altered:1_faulty,2_malicious"

    __experimentOnCOVIDx(DPbyzConfig)


@experiment
def withAndWithoutDP_withAndWithoutByz_5ByzClients_resnet_onCOVIDx():
    epsilon1 = 0.0001
    epsilon3 = 0.0001
    releaseProportion = 0.1

    learningRate = 0.00002
    batchSize = 2
    rounds = 10

    percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1])

    # With DP without attacks
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    DPconfig.percUsers = percUsers

    __experimentOnCOVIDx(DPconfig, model="resnet18")

    # With DP with one attacker
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.aggregators = allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    DPconfig.percUsers = percUsers

    DPconfig.malicious = [3]
    DPconfig.name = "altered:1_malicious"

    __experimentOnCOVIDx(DPconfig, model="resnet18")

    # With DP with more attackers
    DPbyzConfig = DefaultExperimentConfiguration()
    DPbyzConfig.aggregators = allAggregators()
    DPbyzConfig.learningRate = learningRate
    DPbyzConfig.batchSize = batchSize
    DPbyzConfig.rounds = rounds

    DPbyzConfig.privacyPreserve = True
    DPbyzConfig.releaseProportion = releaseProportion
    DPbyzConfig.epsilon1 = epsilon1
    DPbyzConfig.epsilon3 = epsilon3
    DPbyzConfig.needClip = True

    DPbyzConfig.percUsers = percUsers

    DPbyzConfig.faulty = [3]
    DPbyzConfig.malicious = [5]

    DPbyzConfig.name = "altered:1_faulty,1_malicious"

    __experimentOnCOVIDx(DPbyzConfig, model="resnet18")

    # Without DP without attacks
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = allAggregators()
    noDPconfig.learningRate = learningRate
    noDPconfig.batchSize = batchSize
    noDPconfig.rounds = rounds

    noDPconfig.percUsers = percUsers

    __experimentOnCOVIDx(noDPconfig, model="resnet18")


@experiment
def noDP_noByzClient_onCOVIDx():
    configuration = DefaultExperimentConfiguration()
    configuration.batchSize = 64
    configuration.learningRate = 0.0002
    __experimentOnCOVIDx(configuration)


@experiment
def noDP_singleClient_onCOVIDx_100train11test():
    configuration = DefaultExperimentConfiguration()
    configuration.percUsers = torch.tensor([1.0, 2.0])
    configuration.datasetSize = (100, 11)
    configuration.batchSize = 20
    configuration.epochs = 3
    configuration.learningRate = 0.0002
    __experimentOnCOVIDx(configuration)


# @experiment
# def noDP_noByz_onDiabetes():
#     configuration = DefaultExperimentConfiguration()
#     configuration.aggregators = allAggregators()
#     configuration.batchSize = 10
#     configuration.learningRate = 0.001
#     configuration.Optimizer = torch.optim.Adam

#     __experimentOnDiabetes(configuration)


# @experiment
# def withAndWithoutDPandKAnonymization_withAndWithoutByz_10ByzClients_onDiabetes():
#     epsilon1 = 0.0001
#     epsilon3 = 0.0001
#     releaseProportion = 0.1

#     learningRate = 0.00001
#     batchSize = 10
#     epochs = 5
#     rounds = 50

#     percUsers = torch.tensor([0.1, 0.15, 0.2, 0.2, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2])

#     # Vanilla
#     noDPconfig = DefaultExperimentConfiguration()
#     noDPconfig.aggregators = allAggregators()
#     noDPconfig.Optimizer = torch.optim.Adam
#     noDPconfig.learningRate = learningRate
#     noDPconfig.batchSize = batchSize
#     noDPconfig.epochs = epochs
#     noDPconfig.rounds = rounds
#     noDPconfig.percUsers = percUsers
#     __experimentOnDiabetes(noDPconfig)

#     # With DP
#     DPconfig = DefaultExperimentConfiguration()
#     DPconfig.Optimizer = torch.optim.Adam
#     DPconfig.aggregators = allAggregators()
#     DPconfig.learningRate = learningRate
#     DPconfig.batchSize = batchSize
#     DPconfig.epochs = epochs
#     DPconfig.rounds = rounds
#     DPconfig.privacyPreserve = True
#     DPconfig.releaseProportion = releaseProportion
#     DPconfig.epsilon1 = epsilon1
#     DPconfig.epsilon3 = epsilon3
#     DPconfig.needClip = True
#     DPconfig.percUsers = percUsers
#     __experimentOnDiabetes(DPconfig)

#     # With k-anonymity
#     kAnonConfig = DefaultExperimentConfiguration()
#     kAnonConfig.Optimizer = torch.optim.Adam
#     kAnonConfig.aggregators = allAggregators()
#     kAnonConfig.learningRate = learningRate
#     kAnonConfig.batchSize = batchSize
#     kAnonConfig.epochs = epochs
#     kAnonConfig.rounds = rounds
#     kAnonConfig.requireDatasetAnonymization = True
#     kAnonConfig.name = "k:4;"
#     kAnonConfig.percUsers = percUsers
#     __experimentOnDiabetes(kAnonConfig)

#     # With DP with one attacker
#     DPconfig = DefaultExperimentConfiguration()
#     DPconfig.Optimizer = torch.optim.Adam
#     DPconfig.aggregators = allAggregators()
#     DPconfig.learningRate = learningRate
#     DPconfig.batchSize = batchSize
#     DPconfig.epochs = epochs
#     DPconfig.rounds = rounds

#     DPconfig.privacyPreserve = True
#     DPconfig.releaseProportion = releaseProportion
#     DPconfig.epsilon1 = epsilon1
#     DPconfig.epsilon3 = epsilon3
#     DPconfig.needClip = True

#     DPconfig.percUsers = percUsers

#     DPconfig.malicious = [1]
#     DPconfig.name = "altered:1_malicious"

#     __experimentOnDiabetes(DPconfig)
#     # With k-anonymity with one attacker
#     kAnonByzConfig = DefaultExperimentConfiguration()
#     kAnonByzConfig.Optimizer = torch.optim.Adam
#     kAnonByzConfig.aggregators = allAggregators()
#     kAnonByzConfig.learningRate = learningRate
#     kAnonByzConfig.batchSize = batchSize
#     kAnonByzConfig.rounds = rounds
#     kAnonByzConfig.epochs = epochs

#     kAnonByzConfig.requireDatasetAnonymization = True

#     kAnonByzConfig.percUsers = percUsers

#     kAnonByzConfig.malicious = [1]
#     kAnonByzConfig.name = "k:4;"
#     kAnonByzConfig.name += "altered:1_malicious"

#     __experimentOnDiabetes(kAnonByzConfig)

#     # With DP with more attackers
#     DPbyzConfig = DefaultExperimentConfiguration()
#     DPbyzConfig.Optimizer = torch.optim.Adam
#     DPbyzConfig.aggregators = allAggregators()
#     DPbyzConfig.learningRate = learningRate
#     DPbyzConfig.batchSize = batchSize
#     DPbyzConfig.epochs = epochs
#     DPbyzConfig.rounds = rounds

#     DPbyzConfig.privacyPreserve = True
#     DPbyzConfig.releaseProportion = releaseProportion
#     DPbyzConfig.epsilon1 = epsilon1
#     DPbyzConfig.epsilon3 = epsilon3
#     DPbyzConfig.needClip = True

#     DPbyzConfig.percUsers = percUsers

#     DPbyzConfig.malicious = [2, 4]
#     DPbyzConfig.faulty = [1]
#     DPbyzConfig.name = "altered:1_faulty,2_malicious"

#     __experimentOnDiabetes(DPbyzConfig)

#     # With k-anonymity with more attackers
#     kAnonByzConfig = DefaultExperimentConfiguration()
#     kAnonByzConfig.Optimizer = torch.optim.Adam
#     kAnonByzConfig.aggregators = allAggregators()
#     kAnonByzConfig.learningRate = learningRate
#     kAnonByzConfig.batchSize = batchSize
#     kAnonByzConfig.epochs = epochs
#     kAnonByzConfig.rounds = rounds

#     kAnonByzConfig.requireDatasetAnonymization = True

#     kAnonByzConfig.percUsers = percUsers

#     kAnonByzConfig.malicious = [2, 4]
#     kAnonByzConfig.faulty = [3]
#     kAnonByzConfig.name = "k:4;"
#     kAnonByzConfig.name += "altered:1_faulty,2_malicious"

#     __experimentOnDiabetes(kAnonByzConfig)


# @experiment
# def withAndWithoutDPandKAnonymization_withAndWithoutByz_3ByzClients_onDiabetes():
#     epsilon1 = 0.0001
#     epsilon3 = 0.0001
#     releaseProportion = 0.1

#     learningRate = 0.00001
#     batchSize = 10
#     epochs = 5
#     rounds = 10

#     percUsers = torch.tensor([0.3, 0.3, 0.4])

#     # Vanilla
#     noDPconfig = DefaultExperimentConfiguration()
#     noDPconfig.aggregators = allAggregators()
#     noDPconfig.Optimizer = torch.optim.Adam
#     noDPconfig.learningRate = learningRate
#     noDPconfig.batchSize = batchSize
#     noDPconfig.epochs = epochs
#     noDPconfig.rounds = rounds
#     noDPconfig.percUsers = percUsers
#     __experimentOnDiabetes(noDPconfig)

#     # With DP
#     DPconfig = DefaultExperimentConfiguration()
#     DPconfig.Optimizer = torch.optim.Adam
#     DPconfig.aggregators = allAggregators()
#     DPconfig.learningRate = learningRate
#     DPconfig.batchSize = batchSize
#     DPconfig.epochs = epochs
#     DPconfig.rounds = rounds
#     DPconfig.privacyPreserve = True
#     DPconfig.releaseProportion = releaseProportion
#     DPconfig.epsilon1 = epsilon1
#     DPconfig.epsilon3 = epsilon3
#     DPconfig.needClip = True
#     DPconfig.percUsers = percUsers
#     __experimentOnDiabetes(DPconfig)

#     # With k-anonymity
#     kAnonConfig = DefaultExperimentConfiguration()
#     kAnonConfig.Optimizer = torch.optim.Adam
#     kAnonConfig.aggregators = allAggregators()
#     kAnonConfig.learningRate = learningRate
#     kAnonConfig.batchSize = batchSize
#     kAnonConfig.epochs = epochs
#     kAnonConfig.rounds = rounds
#     kAnonConfig.requireDatasetAnonymization = True
#     kAnonConfig.name = "k:4;"
#     kAnonConfig.percUsers = percUsers
#     __experimentOnDiabetes(kAnonConfig)

#     # With DP with one attacker
#     DPconfig = DefaultExperimentConfiguration()
#     DPconfig.Optimizer = torch.optim.Adam
#     DPconfig.aggregators = allAggregators()
#     DPconfig.learningRate = learningRate
#     DPconfig.batchSize = batchSize
#     DPconfig.epochs = epochs
#     DPconfig.rounds = rounds

#     DPconfig.privacyPreserve = True
#     DPconfig.releaseProportion = releaseProportion
#     DPconfig.epsilon1 = epsilon1
#     DPconfig.epsilon3 = epsilon3
#     DPconfig.needClip = True

#     DPconfig.percUsers = percUsers

#     DPconfig.malicious = [1]
#     DPconfig.name = "altered:1_malicious"

#     __experimentOnDiabetes(DPconfig)

#     # With k-anonymity with one attacker
#     kAnonByzConfig = DefaultExperimentConfiguration()
#     kAnonByzConfig.Optimizer = torch.optim.Adam
#     kAnonByzConfig.aggregators = allAggregators()
#     kAnonByzConfig.learningRate = learningRate
#     kAnonByzConfig.batchSize = batchSize
#     kAnonByzConfig.rounds = rounds
#     kAnonByzConfig.epochs = epochs

#     kAnonByzConfig.requireDatasetAnonymization = True

#     kAnonByzConfig.percUsers = percUsers

#     kAnonByzConfig.malicious = [1]
#     kAnonByzConfig.name = "k:4;"
#     kAnonByzConfig.name += "altered:1_malicious"

#     __experimentOnDiabetes(kAnonByzConfig)

#     # With DP with more attackers
#     DPbyzConfig = DefaultExperimentConfiguration()
#     DPbyzConfig.Optimizer = torch.optim.Adam
#     DPbyzConfig.aggregators = allAggregators()
#     DPbyzConfig.learningRate = learningRate
#     DPbyzConfig.batchSize = batchSize
#     DPbyzConfig.epochs = epochs
#     DPbyzConfig.rounds = rounds

#     DPbyzConfig.privacyPreserve = True
#     DPbyzConfig.releaseProportion = releaseProportion
#     DPbyzConfig.epsilon1 = epsilon1
#     DPbyzConfig.epsilon3 = epsilon3
#     DPbyzConfig.needClip = True

#     noDPconfig.percUsers = percUsers

#     DPbyzConfig.malicious = [2, 4]
#     DPbyzConfig.faulty = [1]
#     DPbyzConfig.name = "altered:1_faulty,2_malicious"

#     __experimentOnDiabetes(DPbyzConfig)

#     # With k-anonymity with more attackers
#     kAnonByzConfig = DefaultExperimentConfiguration()
#     kAnonByzConfig.Optimizer = torch.optim.Adam
#     kAnonByzConfig.aggregators = allAggregators()
#     kAnonByzConfig.learningRate = learningRate
#     kAnonByzConfig.batchSize = batchSize
#     kAnonByzConfig.epochs = epochs
#     kAnonByzConfig.rounds = rounds

#     kAnonByzConfig.requireDatasetAnonymization = True

#     kAnonByzConfig.percUsers = percUsers

#     kAnonByzConfig.malicious = [2, 4]
#     kAnonByzConfig.faulty = [3]
#     kAnonByzConfig.name = "k:4;"
#     kAnonByzConfig.name += "altered:1_malicious"

#     __experimentOnDiabetes(kAnonByzConfig)


def __groupedExperiments_SyntacticVsDP(
    batchSize,
    epochs,
    epsilon1,
    epsilon3,
    learningRate,
    percUsers,
    releaseProportion,
    rounds,
    experimentMethod,
):
    # Vanilla
    noDPconfig = DefaultExperimentConfiguration()
    noDPconfig.aggregators = allAggregators()
    noDPconfig.Optimizer = torch.optim.Adam
    noDPconfig.learningRate = learningRate
    noDPconfig.batchSize = batchSize
    noDPconfig.epochs = epochs
    noDPconfig.rounds = rounds
    noDPconfig.percUsers = percUsers
    experimentMethod(noDPconfig)

    # With DP
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.Optimizer = torch.optim.Adam
    DPconfig.aggregators = allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.epochs = epochs
    DPconfig.rounds = rounds
    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True
    DPconfig.percUsers = percUsers
    experimentMethod(DPconfig)

    # With k-anonymity
    kAnonConfig = DefaultExperimentConfiguration()
    kAnonConfig.Optimizer = torch.optim.Adam
    kAnonConfig.aggregators = allAggregators()
    kAnonConfig.learningRate = learningRate
    kAnonConfig.batchSize = batchSize
    kAnonConfig.epochs = epochs
    kAnonConfig.rounds = rounds
    kAnonConfig.requireDatasetAnonymization = True
    kAnonConfig.name = "k:4;"
    kAnonConfig.percUsers = percUsers
    experimentMethod(kAnonConfig)

    # With DP with one attacker
    DPconfig = DefaultExperimentConfiguration()
    DPconfig.Optimizer = torch.optim.Adam
    DPconfig.aggregators = allAggregators()
    DPconfig.learningRate = learningRate
    DPconfig.batchSize = batchSize
    DPconfig.epochs = epochs
    DPconfig.rounds = rounds

    DPconfig.privacyPreserve = True
    DPconfig.releaseProportion = releaseProportion
    DPconfig.epsilon1 = epsilon1
    DPconfig.epsilon3 = epsilon3
    DPconfig.needClip = True

    DPconfig.percUsers = percUsers

    DPconfig.malicious = [3]
    DPconfig.name = "altered:1_malicious"

    experimentMethod(DPconfig)
    # With k-anonymity with one attacker
    kAnonByzConfig = DefaultExperimentConfiguration()
    kAnonByzConfig.Optimizer = torch.optim.Adam
    kAnonByzConfig.aggregators = allAggregators()
    kAnonByzConfig.learningRate = learningRate
    kAnonByzConfig.batchSize = batchSize
    kAnonByzConfig.rounds = rounds
    kAnonByzConfig.epochs = epochs

    kAnonByzConfig.requireDatasetAnonymization = True

    kAnonByzConfig.percUsers = percUsers

    kAnonByzConfig.malicious = [3]
    kAnonByzConfig.name = "k:4;"
    kAnonByzConfig.name += "altered:1_malicious"

    experimentMethod(kAnonByzConfig)
    # With DP with more attackers
    DPbyzConfig = DefaultExperimentConfiguration()
    DPbyzConfig.Optimizer = torch.optim.Adam
    DPbyzConfig.aggregators = allAggregators()
    DPbyzConfig.learningRate = learningRate
    DPbyzConfig.batchSize = batchSize
    DPbyzConfig.epochs = epochs
    DPbyzConfig.rounds = rounds

    DPbyzConfig.privacyPreserve = True
    DPbyzConfig.releaseProportion = releaseProportion
    DPbyzConfig.epsilon1 = epsilon1
    DPbyzConfig.epsilon3 = epsilon3
    DPbyzConfig.needClip = True

    noDPconfig.percUsers = percUsers

    DPbyzConfig.malicious = [2, 4]
    DPbyzConfig.faulty = [1]
    DPbyzConfig.name = "altered:1_faulty,2_malicious"

    experimentMethod(DPbyzConfig)

    # With k-anonymity with more attackers
    kAnonByzConfig = DefaultExperimentConfiguration()
    kAnonByzConfig.Optimizer = torch.optim.Adam
    kAnonByzConfig.aggregators = allAggregators()
    kAnonByzConfig.learningRate = learningRate
    kAnonByzConfig.batchSize = batchSize
    kAnonByzConfig.epochs = epochs
    kAnonByzConfig.rounds = rounds

    kAnonByzConfig.requireDatasetAnonymization = True

    kAnonByzConfig.percUsers = percUsers

    kAnonByzConfig.malicious = [2, 4]
    kAnonByzConfig.faulty = [3]
    kAnonByzConfig.name = "k:4;"
    kAnonByzConfig.name += "altered:1_malicious"

    experimentMethod(kAnonByzConfig)


# @experiment
# def noByz_HeartDisease():
#     config = DefaultExperimentConfiguration()
#     config.percUsers = torch.tensor([0.3, 0.3, 0.4])

#     config.requireDatasetAnonymization = True

#     config.Optimizer = torch.optim.Adam
#     config.learningRate = 0.0001
#     config.batchSize = 20
#     config.rounds = 100
#     config.epochs = 10

#     __experimentOnHeartDisease(config)


# @experiment
# def withAndWithoutDPandKAnonymization_withAndWithoutByz_3ByzClients_onHeartDisease():
#     percUsers = torch.tensor([0.3, 0.3, 0.4])

#     epsilon1 = 0.0001
#     epsilon3 = 0.0001
#     releaseProportion = 0.1

#     learningRate = 0.0001
#     batchSize = 20
#     epochs = 10
#     rounds = 100

#     # Vanilla
#     noDPconfig = DefaultExperimentConfiguration()
#     noDPconfig.aggregators = allAggregators()
#     noDPconfig.Optimizer = torch.optim.Adam
#     noDPconfig.learningRate = learningRate
#     noDPconfig.batchSize = batchSize
#     noDPconfig.epochs = epochs
#     noDPconfig.rounds = rounds
#     noDPconfig.percUsers = percUsers
#     __experimentOnHeartDisease(noDPconfig)

#     # With DP
#     DPconfig = DefaultExperimentConfiguration()
#     DPconfig.Optimizer = torch.optim.Adam
#     DPconfig.aggregators = allAggregators()
#     DPconfig.learningRate = learningRate
#     DPconfig.batchSize = batchSize
#     DPconfig.epochs = epochs
#     DPconfig.rounds = rounds
#     DPconfig.privacyPreserve = True
#     DPconfig.releaseProportion = releaseProportion
#     DPconfig.epsilon1 = epsilon1
#     DPconfig.epsilon3 = epsilon3
#     DPconfig.needClip = True
#     DPconfig.percUsers = percUsers
#     __experimentOnHeartDisease(DPconfig)

#     # With k-anonymity
#     kAnonConfig = DefaultExperimentConfiguration()
#     kAnonConfig.Optimizer = torch.optim.Adam
#     kAnonConfig.aggregators = allAggregators()
#     kAnonConfig.learningRate = learningRate
#     kAnonConfig.batchSize = batchSize
#     kAnonConfig.epochs = epochs
#     kAnonConfig.rounds = rounds
#     kAnonConfig.requireDatasetAnonymization = True
#     kAnonConfig.name = "k:2;"
#     kAnonConfig.percUsers = percUsers
#     __experimentOnHeartDisease(kAnonConfig)

#     # With DP with one attacker
#     DPconfig = DefaultExperimentConfiguration()
#     DPconfig.Optimizer = torch.optim.Adam
#     DPconfig.aggregators = allAggregators()
#     DPconfig.learningRate = learningRate
#     DPconfig.batchSize = batchSize
#     DPconfig.epochs = epochs
#     DPconfig.rounds = rounds

#     DPconfig.privacyPreserve = True
#     DPconfig.releaseProportion = releaseProportion
#     DPconfig.epsilon1 = epsilon1
#     DPconfig.epsilon3 = epsilon3
#     DPconfig.needClip = True

#     DPconfig.percUsers = percUsers

#     DPconfig.malicious = [1]
#     DPconfig.name = "altered:1_malicious"

#     __experimentOnHeartDisease(DPconfig)

#     # With k-anonymity with one attacker
#     kAnonByzConfig = DefaultExperimentConfiguration()
#     kAnonByzConfig.Optimizer = torch.optim.Adam
#     kAnonByzConfig.aggregators = allAggregators()
#     kAnonByzConfig.learningRate = learningRate
#     kAnonByzConfig.batchSize = batchSize
#     kAnonByzConfig.rounds = rounds
#     kAnonByzConfig.epochs = epochs

#     kAnonByzConfig.requireDatasetAnonymization = True

#     kAnonByzConfig.percUsers = percUsers

#     kAnonByzConfig.malicious = [1]
#     kAnonByzConfig.name = "k:2;"
#     kAnonByzConfig.name += "altered:1_malicious"

#     __experimentOnHeartDisease(kAnonByzConfig)

#     # With DP with more attackers
#     DPbyzConfig = DefaultExperimentConfiguration()
#     DPbyzConfig.Optimizer = torch.optim.Adam
#     DPbyzConfig.aggregators = allAggregators()
#     DPbyzConfig.learningRate = learningRate
#     DPbyzConfig.batchSize = batchSize
#     DPbyzConfig.epochs = epochs
#     DPbyzConfig.rounds = rounds

#     DPbyzConfig.privacyPreserve = True
#     DPbyzConfig.releaseProportion = releaseProportion
#     DPbyzConfig.epsilon1 = epsilon1
#     DPbyzConfig.epsilon3 = epsilon3
#     DPbyzConfig.needClip = True

#     noDPconfig.percUsers = percUsers

#     DPbyzConfig.malicious = [2, 4]
#     DPbyzConfig.faulty = [1]
#     DPbyzConfig.name = "altered:1_faulty,2_malicious"

#     __experimentOnHeartDisease(DPbyzConfig)

#     # With k-anonymity with more attackers
#     kAnonByzConfig = DefaultExperimentConfiguration()
#     kAnonByzConfig.Optimizer = torch.optim.Adam
#     kAnonByzConfig.aggregators = allAggregators()
#     kAnonByzConfig.learningRate = learningRate
#     kAnonByzConfig.batchSize = batchSize
#     kAnonByzConfig.epochs = epochs
#     kAnonByzConfig.rounds = rounds

#     kAnonByzConfig.requireDatasetAnonymization = True

#     kAnonByzConfig.percUsers = percUsers

#     kAnonByzConfig.malicious = [2, 4]
#     kAnonByzConfig.faulty = [3]
#     kAnonByzConfig.name = "k:4;"
#     kAnonByzConfig.name += "altered:1_malicious"

#     __experimentOnHeartDisease(kAnonByzConfig)


# @experiment
# def customExperiment():
#     config = DefaultExperimentConfiguration()
#     config.percUsers = torch.tensor([1.0])

#     config.learningRate = 0.0001
#     config.batchSize = 20
#     config.epochs = 10
#     config.rounds = 100

#     # config.requireDatasetAnonymization = True
#     __experimentOnDiabetes(config)


# customExperiment()
