from client import Client
import copy
from logger import logPrint
from threading import Thread
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from typing import List

import torch


class Aggregator:
    def __init__(self, clients, model, rounds, device, useAsyncClients=False):
        self.model = model.to(device)
        self.clients = clients
        self.rounds = rounds

        self.device = device
        self.useAsyncClients = useAsyncClients

        # List of malicious users blocked in tuple of client_id and iteration
        self.maliciousBlocked = []
        # List of benign users blocked
        self.benignBlocked = []
        # List of faulty users blocked
        self.faultyBlocked = []

    def trainAndTest(self, testDataset):
        raise Exception("Train method should be override by child class, " "specific to the aggregation strategy.")

    def _shareModelAndTrainOnClients(self):
        if self.useAsyncClients:
            threads = []
            for client in self.clients:
                t = Thread(target=(lambda: self.__shareModelAndTrainOnClient(client)))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()
        else:
            for client in self.clients:
                self.__shareModelAndTrainOnClient(client)

    def __shareModelAndTrainOnClient(self, client):
        broadcastModel = copy.deepcopy(self.model)
        client.updateModel(broadcastModel)
        error, pred = client.trainModel()

    def _retrieveClientModelsDict(self):
        models = dict()
        for client in self.clients:
            # If client blocked return an the unchanged version of the model
            if not client.blocked:
                models[client] = client.retrieveModel()
            else:
                models[client] = client.model
        return models

    def test(self, testDataset):
        dataLoader = DataLoader(testDataset, shuffle=False)
        with torch.no_grad():
            predLabels, testLabels = zip(*[(self.predict(self.model, x), y) for x, y in dataLoader])
        predLabels = torch.tensor(predLabels, dtype=torch.long)
        testLabels = torch.tensor(testLabels, dtype=torch.long)
        # Confusion matrix and normalized confusion matrix
        mconf = confusion_matrix(testLabels, predLabels)
        errors = 1 - 1.0 * mconf.diagonal().sum() / len(testDataset)
        logPrint("Error Rate: ", round(100.0 * errors, 3), "%")
        return errors

    # Function for computing predictions
    def predict(self, net, x):
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)
        return predicted.to(self.device)

    # Function to merge the models
    @staticmethod
    def _mergeModels(mOrig, mDest, alphaOrig, alphaDest):
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                weightedSum = alphaOrig * param1.data + alphaDest * dictParamsDest[name1].data
                dictParamsDest[name1].data.copy_(weightedSum)

    def handle_blocked(self, client: Client, round: int):
        logPrint("USER ", client.id, " BLOCKED!!!")
        client.p = 0
        client.blocked = True
        if client.byz or client.flip:
            if client.byz:
                self.faultyBlocked.append((client.id, round))
            if client.flip:
                self.maliciousBlocked.append((client.id, round))
        else:
            self.benignBlocked.append((client.id, round))


def allAggregators() -> List[Aggregator]:
    return Aggregator.__subclasses__()