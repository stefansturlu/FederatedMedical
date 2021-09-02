from torch import optim
from utils.typings import BlockedLocations, Errors, FreeRiderAttack, PersonalisationMethod
from datasetLoaders.DatasetInterface import DatasetInterface
from experiment.CustomConfig import CustomConfig
import os
from typing import Callable, Dict, List, NewType, Optional, Tuple, Dict, Type
import json
from loguru import logger

from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from datasetLoaders.MNIST import DatasetLoaderMNIST
from datasetLoaders.COVIDx import DatasetLoaderCOVIDx
from datasetLoaders.COVID19 import DatasetLoaderCOVID19
from datasetLoaders.Pneumonia import DatasetLoaderPneumonia


from classifiers import MNIST, CovidNet, CNN, Pneumonia
from logger import logPrint
from client import Client

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import time
import gc
from torch import cuda, Tensor, nn

from aggregators.Aggregator import Aggregator, allAggregators
from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAPlusPlus import FedMGDAPlusPlusAggregator
from aggregators.FedAvg import FAAggregator
from aggregators.COMED import COMEDAggregator
from aggregators.Clustering import ClusteringAggregator
from aggregators.MKRUM import MKRUMAggregator
from aggregators.FedPADRC import FedPADRCAggregator
from aggregators.FedBE import FedBEAggregator
from aggregators.FedDF import FedDFAggregator
from aggregators.FedABE import FedABEAggregator

SEED = 0

# Colours used for graphing, add more if necessary
COLOURS: List[str] = [
    "midnightblue",
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:cyan",
    "tab:purple",
    "tab:pink",
    "tab:olive",
    "tab:brown",
    "tab:gray",
    "chartreuse",
    "lightcoral",
    "saddlebrown",
    "blueviolet",
    "navy",
    "cornflowerblue",
    "thistle",
    "dodgerblue",
    "crimson",
    "darkseagreen",
    "maroon",
    "mediumspringgreen",
    "burlywood",
    "olivedrab",
    "linen",
    "mediumorchid",
    "teal",
    "black",
    "gold",
]


def __experimentOnMNIST(
    config: DefaultExperimentConfiguration, title="", filename="", folder="DEFAULT"
) -> Dict[str, Errors]:
    """
    MNIST Experiment with default settings
    """
    dataLoader = DatasetLoaderMNIST().getDatasets
    classifier = MNIST.Classifier
    return __experimentSetup(config, dataLoader, classifier, title, filename, folder)


def __experimentOnCOVID19(
    config: DefaultExperimentConfiguration, title="", filename="", folder="DEFAULT"
) -> Dict[str, Errors]:
    """
    COVID19 Experiment with default settings
    """
    dataLoader = DatasetLoaderCOVID19().getDatasets
    classifier = CNN.Classifier
    return __experimentSetup(config, dataLoader, classifier, title, filename, folder)


def __experimentOnCOVIDx(
    config: DefaultExperimentConfiguration,
    model="COVIDNet",
    title="",
    filename="",
    folder="DEFAULT",
) -> Dict[str, Errors]:
    """
    COVIDx Experiment with default settings
    """
    datasetLoader = DatasetLoaderCOVIDx().getDatasets
    if model == "COVIDNet":
        classifier = CovidNet.Classifier
    elif model == "resnet18":
        classifier = CNN.Classifier
    else:
        raise Exception("Invalid Covid model name.")
    return __experimentSetup(config, datasetLoader, classifier, title, filename, folder)


def __experimentOnPneumonia(
    config: DefaultExperimentConfiguration, title="", filename="", folder="DEFAULT"
) -> Dict[str, Errors]:
    """
    Pneumonia Experiment with extra settings in place to incorporate the necessary changes
    """
    datasetLoader = DatasetLoaderPneumonia().getDatasets
    classifier = Pneumonia.Classifier
    # Each client now only has like 80-170 images so a batch size of 200 is pointless
    config.batchSize = 30
    config.labels = torch.tensor([0, 1])
    config.Loss = nn.BCELoss
    config.Optimizer = optim.RMSprop

    return __experimentSetup(config, datasetLoader, classifier, title, filename, folder)


# def __experimentOnDiabetes(config: DefaultExperimentConfiguration):
#     datasetLoader = DatasetLoaderDiabetes(
#         config.requireDatasetAnonymization
#     ).getDatasets
#     classifier = Diabetes.Classifier
#     __experimentSetup(config, datasetLoader, classifier)


# def __experimentOnHeartDisease(config: DefaultExperimentConfiguration):
#     dataLoader = DatasetLoaderHeartDisease(
#         config.requireDatasetAnonymization
#     ).getDatasets
#     classifier = HeartDisease.Classifier
#     __experimentSetup(config, dataLoader, classifier)


def __experimentSetup(
    config: DefaultExperimentConfiguration,
    datasetLoader: Callable[
        [Tensor, Tensor, Optional[Tuple[int, int]]], Tuple[List[DatasetInterface], DatasetInterface]
    ],
    classifier,
    title: str = "DEFAULT_TITLE",
    filename: str = "DEFAULT_NAME",
    folder: str = "DEFAULT_FOLDER",
) -> Dict[str, Errors]:
    __setRandomSeeds()
    gc.collect()
    cuda.empty_cache()
    errorsDict: Dict[str, Errors] = {}
    blocked: Dict[str, BlockedLocations] = {}

    for aggregator in config.aggregators:
        name = aggregator.__name__.replace("Aggregator", "")
        name = name.replace("Plus", "+")
        logPrint("TRAINING {}".format(name))

        if config.privacyPreserve is not None:
            errors, block = __runExperiment(
                config, datasetLoader, classifier, aggregator, config.privacyPreserve, folder
            )
        else:
            errors, block = __runExperiment(
                config, datasetLoader, classifier, aggregator, False, folder
            )
            logPrint("TRAINING {} with DP".format(name))
            errors, block = __runExperiment(
                config, datasetLoader, classifier, aggregator, True, folder
            )

        errorsDict[name] = errors.tolist()
        blocked[name] = block

    # Writing the blocked lists and errors to json files for later inspection.
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.isdir(f"{folder}/json"):
        os.mkdir(f"{folder}/json")
    if not os.path.isdir(f"{folder}/graphs"):
        os.mkdir(f"{folder}/graphs")
    with open(f"{folder}/json/{filename} blocked.json", "w+") as outfile:
        json.dump(blocked, outfile)
    with open(f"{folder}/json/{filename} errors (Seed: {SEED}).json", "w+") as outfile:
        json.dump(errorsDict, outfile)

    # Plots the individual aggregator errors
    if config.plotResults:
        plt.figure()
        i = 0
        for name, err in errorsDict.items():
            # plt.plot(err.numpy(), color=COLOURS[i])
            plt.plot(err, color=COLOURS[i])
            i += 1
        plt.legend(errorsDict.keys())
        plt.xlabel(f"Rounds - {config.epochs} Epochs per Round")
        plt.ylabel("Error Rate (%)")
        plt.title(title, loc="center", wrap=True)
        plt.ylim(0, 1.0)
        plt.savefig(f"{folder}/graphs/{filename}.png", dpi=400)

    return errorsDict


def __runExperiment(
    config: DefaultExperimentConfiguration,
    datasetLoader: Callable[
        [Tensor, Tensor, Optional[Tuple[int, int]]], Tuple[List[DatasetInterface], DatasetInterface]
    ],
    classifier: nn.Module,
    agg: Type[Aggregator],
    useDifferentialPrivacy: bool,
    folder: str = "test",
) -> Tuple[Errors, BlockedLocations]:
    """
    Sets up the experiment to be run.

    Initialises each aggregator appropriately
    """
    serverDataSize = config.serverData
    # if not (agg is FedBEAggregator or agg is FedDFAggregator or agg is FedABEAggregator):
    if not agg.requiresData():
        print("Type of agg:", type(agg))
        print("agg:", agg)
        serverDataSize = 0

    trainDatasets, testDataset, serverDataset = datasetLoader(
        config.percUsers,
        config.labels,
        config.datasetSize,
        config.nonIID,
        config.alphaDirichlet,
        serverDataSize,
    )
    # TODO: Print client data partition, i.e. how many of each class they have. Plot it and put it in report.
    clientPartitions = torch.stack([torch.bincount(t.labels, minlength=10) for t in trainDatasets])
    logPrint(
        f"Client data partition (alpha={config.alphaDirichlet}, percentage on server: {100*serverDataSize:.2f}%)"
    )
    # logPrint(f"Data per client: {clientPartitions.sum(dim=1)}")
    logPrint(f"Number of samples per class for each client: \n{clientPartitions}")
    clients = __initClients(config, trainDatasets, useDifferentialPrivacy)
    # Requires model input size update due to dataset generalisation and categorisation
    if config.requireDatasetAnonymization:
        classifier.inputSize = testDataset.getInputSize()
    model = classifier().to(config.aggregatorConfig.device)
    name = agg.__name__.replace("Aggregator", "")

    aggregator = agg(clients, model, config.aggregatorConfig)

    if isinstance(aggregator, AFAAggregator):
        aggregator.xi = config.aggregatorConfig.xi
        aggregator.deltaXi = config.aggregatorConfig.deltaXi
    elif isinstance(aggregator, FedMGDAPlusPlusAggregator):
        aggregator.reinitialise(config.aggregatorConfig.innerLR)
    elif isinstance(aggregator, FedPADRCAggregator) or isinstance(aggregator, ClusteringAggregator):
        aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
    # elif isinstance(aggregator, FedBEAggregator) or isinstance(aggregator, FedDFAggregator) or isinstance(aggregator, FedABEAggregator):
    elif aggregator.requiresData():
        serverDataset.data = serverDataset.data.to(aggregator.config.device)
        serverDataset.labels = serverDataset.labels.to(aggregator.config.device)
        aggregator.distillationData = serverDataset

    errors: Errors = aggregator.trainAndTest(testDataset)
    blocked = BlockedLocations(
        {
            "benign": aggregator.benignBlocked,
            "malicious": aggregator.maliciousBlocked,
            "faulty": aggregator.faultyBlocked,
            "freeRider": aggregator.freeRidersBlocked,
        }
    )

    # Plot mean and std values from the clients
    if config.aggregatorConfig.detectFreeRiders:

        if not os.path.exists(f"{folder}/std/{name}"):
            os.makedirs(f"{folder}/std/{name}")
        if not os.path.exists(f"{folder}/mean/{name}"):
            os.makedirs(f"{folder}/mean/{name}")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(30):
            if clients[i].free or clients[i].byz or clients[i].flip:
                ax.plot(aggregator.means[i].detach().numpy(), color="red", label="free")
            else:
                ax.plot(aggregator.means[i].detach().numpy(), color="grey", label="normal")
        handles, labels = ax.get_legend_handles_labels()
        plt.legend([handles[1], handles[2]], [labels[1], labels[2]])
        plt.xlabel(f"Rounds - {config.epochs} Epochs per Round")
        plt.ylabel("Mean of Weights")
        plt.title("Mean of Weights over Time", loc="center", wrap=True)
        plt.xlim(0, 30)

        plt.savefig(f"{folder}/mean/{name}/{config.name}.png")
        # plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(30):
            if clients[i].free or clients[i].byz or clients[i].flip:
                ax.plot(aggregator.stds[i].detach().numpy(), color="red", label="free")
            else:
                ax.plot(aggregator.stds[i].detach().numpy(), color="grey", label="normal")
        handles, labels = ax.get_legend_handles_labels()
        plt.legend([handles[1], handles[2]], [labels[1], labels[2]])
        plt.xlabel(f"Rounds - {config.epochs} Epochs per Round")
        plt.ylabel("StD of Weights")
        plt.title("Standard Deviation of Weights over Time", loc="center", wrap=True)
        plt.xlim(0, 30)
        plt.savefig(f"{folder}/std/{name}/{config.name}.png")
        # plt.show()

    return errors, blocked


def __initClients(
    config: DefaultExperimentConfiguration,
    trainDatasets: List[DatasetInterface],
    useDifferentialPrivacy: bool,
) -> List[Client]:
    """
    Initialises each client with their datasets, weights and whether they are not benign
    """
    usersNo = config.percUsers.size(0)
    p0 = 1 / usersNo
    logPrint("Creating clients...")
    clients: List[Client] = []
    for i in range(usersNo):
        clients.append(
            Client(
                idx=i,
                trainDataset=trainDatasets[i],
                epochs=config.epochs,
                batchSize=config.batchSize,
                learningRate=config.learningRate,
                p=p0,
                alpha=config.alpha,
                beta=config.beta,
                Loss=config.Loss,
                Optimizer=config.Optimizer,
                device=config.aggregatorConfig.device,
                useDifferentialPrivacy=useDifferentialPrivacy,
                epsilon1=config.epsilon1,
                epsilon3=config.epsilon3,
                needClip=config.needClip,
                clipValue=config.clipValue,
                needNormalization=config.needNormalization,
                releaseProportion=config.releaseProportion,
            )
        )

    nTrain = sum([client.n for client in clients])
    # Weight the value of the update of each user according to the number of training data points
    for client in clients:
        client.p = client.n / nTrain

    # Create malicious (byzantine) and faulty users
    for client in clients:
        if client.id in config.faulty:
            client.byz = True
            logPrint("User", client.id, "is faulty.")
        if client.id in config.malicious:
            client.flip = True
            logPrint("User", client.id, "is malicious.")
            client.trainDataset.zeroLabels()
        if client.id in config.freeRiding:
            client.free = True
            logPrint("User", client.id, "is Free-Riding.")
    return clients


def __setRandomSeeds(seed=SEED) -> None:
    """
    Sets random seeds for all of the relevant modules.

    Ensures consistent and deterministic results from experiments.
    """
    print(f"Setting seeds to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)


def experiment(exp: Callable[[], None]):
    """
    Decorator for experiments so that time can be known and seeds can be set

    Logger catch is set for better error catching and printing but is not necessary
    """

    @logger.catch
    def decorator():
        __setRandomSeeds()
        logPrint("Experiment {} began.".format(exp.__name__))
        begin = time.time()
        exp()
        end = time.time()
        logPrint("Experiment {} took {}".format(exp.__name__, end - begin))

    return decorator


@experiment
def program() -> None:
    """
    Main program for running the experiments that you want run.
    """
    config = CustomConfig()

    if (
        FedPADRCAggregator in config.aggregators or FedMGDAPlusPlusAggregator in config.aggregators
    ) and config.aggregatorConfig.privacyAmplification:
        print("Currently doesn't support both at the same time.")
        print("Size of clients is very likely to be smaller than or very close to cluster_count.")
        print(
            "FedMGDA+ relies on every client being present and training at every federated round."
        )
        exit(-1)

    errorsDict = {}

    for attackName in config.scenario_conversion():
        errors = __experimentOnMNIST(
            config,
            title=f"Basic CustomConfig Test \n Attack: {attackName}",
            filename=f"{attackName}",
            folder=f"test",
        )


# Running the program here
program()
