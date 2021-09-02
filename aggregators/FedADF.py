from random import uniform
from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from client import Client
from logger import logPrint
from typing import List
import torch
import copy
from aggregators.Aggregator import Aggregator
from torch import nn, Tensor
from scipy.stats import beta
from datasetLoaders.DatasetInterface import DatasetInterface
from utils.KnowledgeDistiller import KnowledgeDistiller


class FedADFAggregator(Aggregator):
    """
    Adaptive Federated Averaging Aggregator
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        self.xi: float = config.xi
        self.deltaXi: float = config.deltaXi
        self.distillationData = None
        self.true_labels = None
        self.pseudolabelMethod = "medlogits"

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.rounds))

        for r in range(self.rounds):

            logPrint("Round... ", r)

            if self.config.privacyAmplification:
                self.chosen_indices = [
                    i
                    for i in range(len(self.clients))
                    if uniform(0, 1) <= self.config.amplificationP
                ]

            chosen_clients = [self.clients[i] for i in self.chosen_indices]

            for client in chosen_clients:
                broadcastModel = copy.deepcopy(self.model)
                client.updateModel(broadcastModel)
                if not client.blocked:
                    error, pred = client.trainModel()

            models = self._retrieveClientModelsDict()

            self.model = self.aggregate(chosen_clients, models)
            # print([c.n for c in chosen_clients])
            # print([c.pEpoch for c in chosen_clients])
            # print([c.p for c in chosen_clients])
            self.round = r

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __modelSimilarity(self, mOrig: nn.Module, mDest: nn.Module) -> Tensor:
        """
        Calculates model similarity based on the Cosine Similarity metric.
        Flattens the models into tensors before doing the comparison.
        """
        cos = nn.CosineSimilarity(0)

        d2 = torch.tensor([]).to(self.device)
        d1 = torch.tensor([]).to(self.device)

        paramsOrig = mOrig.named_parameters()
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)

        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                d1 = torch.cat((d1, dictParamsDest[name1].data.view(-1)))
                d2 = torch.cat((d2, param1.data.view(-1)))

        sim: Tensor = cos(d1, d2)

        return sim

    @staticmethod
    def checkBlockedUser(a: float, b: float, thr: float = 0.95) -> bool:
        """
        Checks if the user is blocked based on if the beta cdf distribution is greater than the threshold value.
        """
        return beta.cdf(0.5, a, b) > thr

    @staticmethod
    def updateUserScore(client: Client) -> None:
        """
        Updates client score based on its alpha and beta parameters.
        Updates either beta or alpha depending on if it has been classified as a bad update.
        """
        if client.badUpdate:
            client.beta += 1
        else:
            client.alpha += 1
        # This was alpha / beta. The AFA paper uses a probability of alpha/(alpha+beta).
        client.score = client.alpha / client.beta

    @staticmethod
    def notBlockedNorBadUpdate(client: Client) -> bool:
        """
        Returns True if the client isn't blocked or doesn't have a bad update.
        """
        return client.blocked == False | client.badUpdate == False

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        # We can't do aggregation if there are no models this round
        if len(models) == 0:
            return self.model

        empty_model = copy.deepcopy(self.model)
        self.renormalise_weights(clients)

        badCount: int = 2
        slack = self.xi
        while badCount != 0:
            pT_epoch = 0.0

            # Calculate the new weighting for each client as this epoch
            for client in clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.n * client.score
                    pT_epoch = pT_epoch + client.pEpoch

            # Normalise each weighting
            for client in clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.pEpoch / pT_epoch

            # Merge "good" models to form temporary global model
            comb = 0.0
            for i, client in enumerate(clients):
                if self.notBlockedNorBadUpdate(client):
                    self._mergeModels(
                        models[i].to(self.device),
                        empty_model.to(self.device),
                        client.pEpoch,
                        comb,
                    )
                    comb = 1.0

            # Calculate similarity between temporary global model and each "good" model
            # "Bad" models will receive worst score of 0 by default # NOTE: This is not true any more!
            sim = []  # torch.zeros(len(clients)).to(self.device)
            for i, client in enumerate(clients):
                if self.notBlockedNorBadUpdate(client):
                    client.sim = self.__modelSimilarity(empty_model, models[i])
                    # print(f"client {i} sim: {client.sim}")
                    # sim[i] = client.sim
                    sim.append(client.sim)
            sim = torch.tensor(sim)

            meanS = torch.mean(sim)
            medianS = torch.median(sim)
            desvS = torch.std(sim)

            if meanS < medianS:
                th = medianS - slack * desvS
            else:
                th = medianS + slack * desvS

            slack += self.deltaXi

            badCount = 0
            for client in clients:
                if not client.badUpdate:
                    # Malicious clients are below the threshold
                    if meanS < medianS:
                        if client.sim < th:
                            client.badUpdate = True
                            badCount += 1
                    # Malicious clients are above the threshold
                    else:
                        if client.sim > th:
                            client.badUpdate = True
                            badCount += 1

        # Block relevant clients based on their assigned scores from this round
        # Assign client's actual weighting based on updated score
        pT = 0.0
        for client in clients:
            if not client.blocked:
                self.updateUserScore(client)
                client.blocked = self.checkBlockedUser(client.alpha, client.beta)
                if client.blocked:
                    self.handle_blocked(client, self.round)
                else:
                    client.p = client.n * client.score
                    pT = pT + client.p

        # Normalise client's actual weighting
        for client in clients:
            client.p = client.p / pT

        # Update model's epoch weights with the updated scores
        pT_epoch = 0.0
        for client in clients:
            if self.notBlockedNorBadUpdate(client):
                client.pEpoch = client.n * client.score
                pT_epoch += client.pEpoch

        # Normalise epoch weights
        for client in clients:
            if self.notBlockedNorBadUpdate(client):
                client.pEpoch /= pT_epoch

        # Do actual aggregation of clients
        comb = 0.0
        for i, client in enumerate(clients):
            if self.notBlockedNorBadUpdate(client):
                self._mergeModels(
                    models[i].to(self.device),
                    empty_model.to(self.device),
                    client.pEpoch,
                    comb,
                )
                comb = 1.0

        # Distill knowledge using median pseudolabels
        notBlockedModels = [
            models[i] for i, c in enumerate(clients) if self.notBlockedNorBadUpdate(c)
        ]
        logPrint(
            f"FedADF: Distilling knowledge using median {len(notBlockedModels)} client model pseudolabels"
        )
        logPrint(
            f"FedADF: These were left out: {[i for i, c in enumerate(clients) if not self.notBlockedNorBadUpdate(c)]}"
        )
        kd = KnowledgeDistiller(self.distillationData, method=self.pseudolabelMethod)
        empty_model = kd.distillKnowledge(notBlockedModels, empty_model)

        # Reset badUpdate variable
        for client in clients:
            if not client.blocked:
                client.badUpdate = False

        return empty_model

    @staticmethod
    def requiresData():
        """
        Returns boolean value depending on whether the aggregation method requires server data or not.
        """
        return True
