from experiment.AggregatorConfig import AggregatorConfig
from utils.FreeRider import FreeRider
from datasetLoaders.DatasetInterface import DatasetInterface
from torch import Tensor, nn, device
from client import Client
import copy
from logger import logPrint
from threading import Thread
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from typing import List, NewType, Tuple
import torch
import matplotlib.pyplot as plt
from random import uniform

IdRoundPair = NewType("IdRoundPair", Tuple[int, int])

class Aggregator:
    def __init__(self, clients: List[Client], model: nn.Module, config: AggregatorConfig, useAsyncClients: bool = False):
        self.model = model.to(config.device)
        self.clients: List[Client] = clients
        self.rounds: int = config.rounds
        self.config = config

        self.device = config.device
        self.useAsyncClients = useAsyncClients
        self.detectFreeRiders = config.detectFreeRiders

        # Used for free-rider detection
        self.stds = torch.zeros((len(clients), self.rounds))
        self.means = torch.zeros((len(clients), self.rounds))
        self.prev_model: nn.Module = None

        self.round = 0

        # List of malicious users blocked in tuple of client_id and iteration
        self.maliciousBlocked: List[IdRoundPair] = []
        # List of benign users blocked
        self.benignBlocked: List[IdRoundPair] = []
        # List of faulty users blocked
        self.faultyBlocked: List[IdRoundPair] = []
        # List of free-riding users blocked
        self.freeRidersBlocked: List[IdRoundPair] = []

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        raise Exception(
            "Train method should be overridden by child class, "
            "specific to the aggregation strategy."
        )

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        raise Exception(
            "Aggregation method should be overridden by child class, "
            "specific to the aggregation strategy."
        )


    def _shareModelAndTrainOnClients(self, models:List[nn.Module]=None, labels:List[int]=None):
        if models == None and labels == None:
            models = [self.model]
            labels = [0]*len(self.clients)

        if self.config.privacyAmplification:
            chosen_indices = [i for i in range(len(self.clients)) if uniform(0, 1) <= self.config.amplificationP]
            print(chosen_indices)
            print(len(chosen_indices))

        if self.useAsyncClients:
            threads = []
            for client in self.clients:
                model = models[labels[client.id]]
                t = Thread(target=(lambda: self.__shareModelAndTrainOnClient(client, model)))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()
        else:
            for client in self.clients:
                model = models[labels[client.id]]
                self.__shareModelAndTrainOnClient(client, model)


    def __shareModelAndTrainOnClient(self, client: Client, model: nn.Module) -> None:
        broadcastModel = copy.deepcopy(model)
        client.updateModel(broadcastModel)
        error, pred = client.trainModel()

    def _retrieveClientModelsDict(self):
        models: List[nn.Module] = []
        for client in self.clients:
            # If client blocked return an the unchanged version of the model
            if not client.blocked:
                models.append(client.retrieveModel())
            else:
                models.append(client.model)

        if self.detectFreeRiders:
            self.handle_free_riders(models)
        return models

    def test(self, testDataset) -> float:
        dataLoader = DataLoader(testDataset, shuffle=False)
        with torch.no_grad():
            predLabels, testLabels = zip(*[(self.predict(self.model, x), y) for x, y in dataLoader])
        predLabels = torch.tensor(predLabels, dtype=torch.long)
        testLabels = torch.tensor(testLabels, dtype=torch.long)
        # Confusion matrix and normalized confusion matrix
        mconf = confusion_matrix(testLabels, predLabels)
        errors: float = 1 - 1.0 * mconf.diagonal().sum() / len(testDataset)
        logPrint("Error Rate: ", round(100.0 * errors, 3), "%")
        return errors

    # Function for computing predictions
    def predict(self, net: nn.Module, x):
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)
        return predicted.to(self.device)

    # Function to merge the models
    @staticmethod
    def _mergeModels(mOrig: nn.Module, mDest: nn.Module, alphaOrig: float, alphaDest: float) -> None:
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                weightedSum = alphaOrig * param1.data + alphaDest * dictParamsDest[name1].data
                dictParamsDest[name1].data.copy_(weightedSum)

    def handle_blocked(self, client: Client, round: int) -> None:
        logPrint("USER ", client.id, " BLOCKED!!!")
        client.p = 0
        client.blocked = True
        pair: IdRoundPair = (client.id, round)
        if client.byz or client.flip or client.free:
            if client.byz:
                self.faultyBlocked.append(pair)
            if client.flip:
                self.maliciousBlocked.append(pair)
            if client.free:
                self.freeRidersBlocked.append(pair)
        else:
            self.benignBlocked.append(pair)


    def handle_free_riders(self, models: List[nn.Module]):
        """Function to handle when we want to detect the presence of free-riders"""
        for id, model in enumerate(models):
            if self.clients[id].free:
                mean, std = FreeRider.free_grads(model, self.prev_model)
            else:
                mean, std = FreeRider.normal_grads(model)

            self.means[id][self.round] = mean.to(self.device)
            self.stds[id][self.round] = std.to(self.device)

        self.round += 1
        self.prev_model = copy.deepcopy(self.model)



def allAggregators() -> List[Aggregator]:
    return Aggregator.__subclasses__()
