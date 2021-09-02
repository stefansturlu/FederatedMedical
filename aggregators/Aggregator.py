from utils.typings import Errors, IdRoundPair
from experiment.AggregatorConfig import AggregatorConfig
from utils.FreeRider import FreeRider
from datasetLoaders.DatasetInterface import DatasetInterface
from torch import nn
from client import Client
import copy
from logger import logPrint
from threading import Thread
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional, Type
import torch
from random import uniform
from copy import deepcopy
import gc


class Aggregator:
    """
    Base Aggregator class that all aggregators should inherit from
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
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
        self.free_rider_util = FreeRider(self.device, self.config.freeRiderAttack)

        self.round = 0

        # List of malicious users blocked in tuple of client_id and iteration
        self.maliciousBlocked: List[IdRoundPair] = []
        # List of benign users blocked
        self.benignBlocked: List[IdRoundPair] = []
        # List of faulty users blocked
        self.faultyBlocked: List[IdRoundPair] = []
        # List of free-riding users blocked
        self.freeRidersBlocked: List[IdRoundPair] = []

        # Privacy amplification data
        self.chosen_indices = [i for i in range(len(self.clients))]

        # self.requiresData

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        """
        Sends the global model out each federated round for local clients to train on.

        Collects the local models from the clients and aggregates them specific to the aggregation strategy.
        """
        raise Exception(
            "Train method should be overridden by child class, "
            "specific to the aggregation strategy."
        )

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        """
        Performs the actual aggregation for the relevant aggregation strategy.
        """
        raise Exception(
            "Aggregation method should be overridden by child class, "
            "specific to the aggregation strategy."
        )

    def _shareModelAndTrainOnClients(
        self, models: Optional[List[nn.Module]] = None, labels: Optional[List[int]] = None
    ):
        """
        Method for sharing the relevant models to the relevant clients.
        By default, the global model will be shared but this can be changed depending on personalisation need.
        """
        # Default: global model
        if models == None and labels == None:
            models = [self.model]
            labels = [0] * len(self.clients)

        # Randomly choose clients to be sampled with for privacy amplification
        if self.config.privacyAmplification:
            self.chosen_indices = [
                i for i in range(len(self.clients)) if uniform(0, 1) <= self.config.amplificationP
            ]

        chosen_clients = [self.clients[i] for i in self.chosen_indices]

        # Actual sharing and training takes place here
        if self.useAsyncClients:
            threads: List[Thread] = []
            for client in chosen_clients:
                model = models[labels[client.id]]
                t = Thread(target=(lambda: self.__shareModelAndTrainOnClient(client, model)))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()
        else:
            for client in chosen_clients:
                gc.collect()
                torch.cuda.empty_cache()
                model = models[labels[client.id]]
                self.__shareModelAndTrainOnClient(client, model)

    def __shareModelAndTrainOnClient(self, client: Client, model: nn.Module) -> None:
        """
        Shares the given model to the given client and trains it.
        """
        broadcastModel = copy.deepcopy(model)
        client.updateModel(broadcastModel)
        error, pred = client.trainModel()

    def _retrieveClientModelsDict(self):
        """
        Retrieve the models from the clients if not blocked with the appropriate modifications, otherwise just use the clients model
        """
        models: List[nn.Module] = []
        chosen_clients = [self.clients[i] for i in self.chosen_indices]

        for client in chosen_clients:
            # If client blocked return an the unchanged version of the model
            if not client.blocked:
                models.append(client.retrieveModel())
            else:
                models.append(client.model)

        if self.detectFreeRiders:
            self.handle_free_riders(models, chosen_clients)
        return models

    def test(self, testDataset: DatasetInterface) -> float:
        """
        Tests the global model with the global test dataset.
        Bear in mind, this test dataset wouldn't be available in a true Federated setup.
        """
        dataLoader = DataLoader(testDataset, shuffle=False)
        with torch.no_grad():
            predLabels, testLabels = zip(*[(self.predict(self.model, x), y) for x, y in dataLoader])

            predLabels = torch.tensor(predLabels, dtype=torch.long)
            testLabels = torch.tensor(testLabels, dtype=torch.long)

            # Confusion matrix and normalized confusion matrix
            mconf = confusion_matrix(testLabels, predLabels)
            accPerClass = (mconf / (mconf.sum(axis=0) + 0.00001)[:, np.newaxis]).diagonal()
            logPrint(f"Accuracy per class:\n\t{accPerClass}")
            errors: float = 1 - 1.0 * mconf.diagonal().sum() / len(testDataset)
            logPrint("Error Rate: ", round(100.0 * errors, 3), "%")

            return errors

    def predict(self, net: nn.Module, x):
        """
        Returns the best indices (labels) associated with the model prediction
        """
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)

        return predicted.to(self.device)

    @staticmethod
    def _mergeModels(
        mOrig: nn.Module, mDest: nn.Module, alphaOrig: float, alphaDest: float
    ) -> None:
        """
        Merges 2 models together.
        Usually used in conjunction with one of the models being the future global model.
        """
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                weightedSum = alphaOrig * param1.data + alphaDest * dictParamsDest[name1].data
                dictParamsDest[name1].data.copy_(weightedSum)

    @staticmethod
    def _averageModel(models: List[nn.Module], clients: List[Client] = None):
        """
        Takes weighted average of models, using weights from clients.
        """
        if len(models) == 0:
            return None

        client_p = torch.ones(len(models)) / len(models)
        if clients:
            client_p = torch.tensor([c.p for c in clients])

        model = deepcopy(models[0])
        model_state_dict = model.state_dict()

        model_dicts = [m.state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1] for m in model_dicts])
            p_shape = torch.tensor(x.shape)
            p_shape[1:] = 1
            client_p = client_p.view(list(p_shape))

            x_mean = (x * client_p).sum(dim=0)
            model_state_dict[name1].data.copy_(x_mean)
        return model

    @staticmethod
    def _weightedAverageModel(models: List[nn.Module], weights: torch.Tensor = None):
        """
        Takes weighted average of models, using weights from clients.
        """
        if len(models) == 0:
            return None

        client_p = torch.ones(len(models)) / len(models)
        if weights is not None:
            client_p = weights

        model = deepcopy(models[0])
        model_state_dict = model.state_dict()

        model_dicts = [m.state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1] for m in model_dicts])
            p_shape = torch.tensor(x.shape)
            p_shape[1:] = 1
            client_p = client_p.view(list(p_shape))

            x_mean = (x * client_p).sum(dim=0)
            model_state_dict[name1].data.copy_(x_mean)
        return model

    @staticmethod
    def _medianModel(models: List[nn.Module]):
        """
        Takes element-wise median of models.
        """
        if len(models) == 0:
            return None
        model = deepcopy(models[0])
        model_state_dict = model.state_dict()

        model_dicts = [m.state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1] for m in model_dicts])
            x_median, _ = x.median(dim=0)
            model_state_dict[name1].data.copy_(x_median)
        return model

    def handle_blocked(self, client: Client, round: int) -> None:
        """
        Blocks the relevant client, sets its wighting to 0 and appends it to the relevant blocked lists.
        """
        logPrint("USER ", client.id, " BLOCKED!!!")
        client.p = 0
        client.blocked = True
        pair = IdRoundPair((client.id, round))
        if client.byz or client.flip or client.free:
            if client.byz:
                self.faultyBlocked.append(pair)
            if client.flip:
                self.maliciousBlocked.append(pair)
            if client.free:
                self.freeRidersBlocked.append(pair)
        else:
            self.benignBlocked.append(pair)

    def handle_free_riders(self, models: List[nn.Module], clients: List[Client]):
        """Function to handle when we want to detect the presence of free-riders"""
        for i, model in enumerate(models):
            client = clients[i]

            if not client.blocked:
                # Technically the aggregator wouldn't know this but we can't manually set grads so this will do
                if client.free:
                    mean, std = self.free_rider_util.free_grads(model, client.prev_model)
                else:
                    mean, std = self.free_rider_util.normal_grads(model)

                self.means[client.id][self.round] = mean.to(self.device)
                self.stds[client.id][self.round] = std.to(self.device)

        self.round += 1

    def renormalise_weights(self, clients: List[Client]):
        """
        Renormalises weights for:
            Privacy Amplification,
            Clustering Aggregation,
            FedPADRC Aggregation,
        """
        # Shouldn't change unless number of clients is less than len(self.clients)
        weight_total = sum([c.p for c in clients])
        for c in clients:
            c.p /= weight_total

    @staticmethod
    def requiresData():
        """
        Returns boolean value depending on whether the aggregation method requires server data or not.
        This should be overwritten in the subclasses which require data on the server side.
        """
        return False


def allAggregators() -> List[Type[Aggregator]]:
    return Aggregator.__subclasses__()
