from utils.typings import BlockedLocations, Errors, FreeRiderAttack
from datasetLoaders.DatasetInterface import DatasetInterface
from experiment.CustomConfig import CustomConfig
import os
from typing import Callable, Dict, List, Literal, NewType, Tuple, Dict, Type, Union
import json
from loguru import logger

from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from datasetLoaders.MNIST import DatasetLoaderMNIST
from datasetLoaders.COVIDx import DatasetLoaderCOVIDx
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
from aggregators.FedMGDAPlus import FedMGDAPlusAggregator
from aggregators.FedAvg import FAAggregator
from aggregators.COMED import COMEDAggregator
from aggregators.Clustering import ClusteringAggregator
from aggregators.MKRUM import MKRUMAggregator
from aggregators.GroupWise import GroupWiseAggregation



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
):
    dataLoader = DatasetLoaderMNIST().getDatasets
    classifier = MNIST.Classifier
    return __experimentSetup(config, dataLoader, classifier, title, filename, folder)


def __experimentOnCOVIDx(
    config: DefaultExperimentConfiguration,
    model="COVIDNet",
    title="",
    filename="",
    folder="DEFAULT",
):
    datasetLoader = DatasetLoaderCOVIDx().getDatasets
    if model == "COVIDNet":
        classifier = CovidNet.Classifier
    elif model == "resnet18":
        classifier = CNN.Classifier
    else:
        raise Exception("Invalid Covid model name.")
    __experimentSetup(config, datasetLoader, classifier)


def __experimentOnPneumonia(
    config: DefaultExperimentConfiguration, title="", filename="", folder="DEFAULT"
):
    datasetLoader = DatasetLoaderPneumonia().getDatasets
    classifier = Pneumonia.Classifier
    # Each client now only has like 80-170 images so a batch size of 200 is pointless
    config.batchSize = 30

    __experimentSetup(config, datasetLoader, classifier)


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
    datasetLoader: Callable[[Tensor, Tensor, int], Tuple[List[DatasetInterface], DatasetInterface]],
    classifier,
    title: str = "DEFAULT_TITLE",
    filename: str = "DEFAULT_NAME",
    folder: str = "DEFAULT_FOLDER",
):
    print(title)
    print(filename)
    __setRandomSeeds()
    gc.collect()
    cuda.empty_cache()
    errorsDict: Dict[str, Errors] = {}
    blocked: Dict[str, BlockedLocations] = {}

    for aggregator in config.aggregators:
        name = aggregator.__name__.replace("Aggregator", "")
        name = name.replace("Plus", "+")
        # name += ":" + config.name if config.name else ""
        logPrint("TRAINING {}".format(name))
        if config.privacyPreserve is not None:
            errors, block = __runExperiment(
                config, datasetLoader, classifier, aggregator, config.privacyPreserve, folder
            )
        else:
            errors, block = __runExperiment(
                config,
                datasetLoader,
                classifier,
                aggregator,
                False,
                folder
            )
            logPrint("TRAINING {} with DP".format(name))
            errors, block = __runExperiment(
                config,
                datasetLoader,
                classifier,
                aggregator,
                True,
                folder
            )

        errorsDict[name] = errors
        blocked[name] = block

    # Writing the blocked lists to json file for later inspection
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.isdir(f"{folder}/json"):
        os.mkdir(f"{folder}/json")
    if not os.path.isdir(f"{folder}/graphs"):
        os.mkdir(f"{folder}/graphs")
    with open(f"{folder}/json/{filename}.json", "w+") as outfile:
        json.dump(blocked, outfile)

    if config.plotResults:
        plt.figure()
        i = 0
        for name, err in errorsDict.items():
            plt.plot(err.numpy(), color=COLOURS[i])
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
    datasetLoader,
    classifier: nn.Module,
    agg: Type[Aggregator],
    useDifferentialPrivacy: bool,
    folder:str="test"
) -> Tuple[Errors, BlockedLocations]:

    trainDatasets, testDataset = datasetLoader(config.percUsers, config.labels, config.datasetSize, config.nonIID, config.alpha)
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
    elif isinstance(aggregator, FedMGDAPlusAggregator):
        aggregator.reinitialise(config.aggregatorConfig.innerLR)
    elif isinstance(aggregator, GroupWiseAggregation) or isinstance(aggregator, ClusteringAggregator):
        aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)


    errors: Errors = aggregator.trainAndTest(testDataset)
    blocked = BlockedLocations({
        "benign": aggregator.benignBlocked,
        "malicious": aggregator.maliciousBlocked,
        "faulty": aggregator.faultyBlocked,
        "freeRider": aggregator.freeRidersBlocked,
    })

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
    config: DefaultExperimentConfiguration, trainDatasets, useDifferentialPrivacy
) -> List[Client]:
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


def __setRandomSeeds(seed=0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)


# Experiment decorator
def experiment(exp: Callable[[], None]):
    @logger.catch  # Not necessarily needed but catches errors really nicely
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
    config = CustomConfig()

    if (GroupWiseAggregation in config.aggregators or FedMGDAPlusAggregator in config.aggregators) and config.aggregatorConfig.privacyAmplification:
        print("Currently doesn't support both at the same time.")
        print("Size of clients is very likely to be smaller than or very close to cluster_count.")
        print("FedMGDA+ relies on every client being present and training at every federated round.")
        exit(-1)



    for attackName in config.scenario_conversion():
        print(f"Hi! Running this: {attackName}")
        errors = __experimentOnMNIST(
            config,
            title=f"Basic CustomConfig Test \n Attack: {attackName}",
            filename=f"{attackName}",
            folder=f"test/",
        )




# Running the program here
program()
