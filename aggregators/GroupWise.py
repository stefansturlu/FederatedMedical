import os
from utils.typings import Errors, PersonalisationMethod
from experiment.AggregatorConfig import AggregatorConfig
from aggregators.FedMGDAPlus import FedMGDAPlusAggregator
from aggregators.AFA import AFAAggregator
from torch import nn, Tensor
from client import Client
from logger import logPrint
from typing import List, Tuple, Type
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import heapq
from utils.PCA import PCA

# Group-Wise Aggregator based on clustering
# Even though it itself does not do aggregation, it makes programatic sense to inherit attributes and functions
class GroupWiseAggregation(Aggregator):
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)

        self.config = config

        self.cluster_count = self.config.cluster_count
        self.cluster_centres: List[nn.Module] = [None] * self.cluster_count
        self.cluster_centres_len = torch.zeros(self.cluster_count)
        self.cluster_labels = [0] * len(self.clients)

        self.internalAggregator: Aggregator = None
        self.externalAggregator: Aggregator = None

        self.blocked_ps = []

        self.personalisation: PersonalisationMethod = config.personalisation

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.config.rounds))
        no_global_rounds_error = torch.zeros(self.config.rounds, self.cluster_count)
        for r in range(self.config.rounds):
            logPrint("Round... ", r)
            if r == 0:
                self._shareModelAndTrainOnClients()
            else:
                self._shareModelAndTrainOnClients(self.cluster_centres, self.cluster_labels)

            models = self._retrieveClientModelsDict()


            # Perform Clustering
            with torch.no_grad():
                self.generate_cluster_centres(models)


                if self.personalisation == PersonalisationMethod.NO_GLOBAL:
                    for i in range(len(self.cluster_centres)):
                        print(i)
                        self.model = self.cluster_centres[i]
                        err = self.test(testDataset)
                        no_global_rounds_error[r, i] = err
                    continue

                # Assume p value is based on size of cluster
                best_models, ps, indices = self._use_closest_clusters()
                conc_ps = [ps[i] for i in indices]
                conc_ps = [p / sum(conc_ps) for p in conc_ps]

                concentrated = self.externalAggregator.aggregate(
                    [FakeClient(p, i) for (i, p) in enumerate(conc_ps)], best_models
                )

                for i in range(len(self.cluster_centres)):
                    if i in indices:
                        self.cluster_centres[i] = concentrated

                if self.personalisation == PersonalisationMethod.SELECTIVE:
                    self.model = concentrated
                    roundsError[r] = self.test(testDataset)

                elif self.personalisation == PersonalisationMethod.GENERAL:
                    general = self.externalAggregator.aggregate(
                        [FakeClient(p, i) for (i, p) in enumerate(ps)], self.cluster_centres
                    )

                    for i in range(len(self.cluster_centres)):
                        if i not in indices:
                            self.cluster_centres[i] = general

                    self.model = general
                    roundsError[r] = self.test(testDataset)


        if self.personalisation == PersonalisationMethod.NO_GLOBAL:
            if not os.path.exists("personalisation_tests_4d/no_global"):
                os.makedirs("personalisation_tests_4d/no_global")

            plt.figure()
            plt.plot(range(self.rounds), no_global_rounds_error)

            plt.xlabel(f"Rounds")
            plt.ylabel("Error Rate (%)")
            plt.title(f"4D Personalisation Test: No Global \n {self.config.attackName}", loc="center", wrap=True)
            plt.ylim(0, 1.0)
            plt.savefig(f"personalisation_tests_4d/no_global/{self.config.attackName}.png", dpi=400)

        return roundsError

    def _init_aggregators(self, internal: Type[Aggregator], external: Type[Aggregator]) -> None:
        self.internalAggregator = self.__init_aggregator(internal)
        self.externalAggregator = self.__init_aggregator(external)

    def __init_aggregator(self, aggregator: Type[Aggregator]) -> Aggregator:
        agg = aggregator(self.clients, self.model, self.config)
        if isinstance(agg, AFAAggregator):
            agg.xi = self.config.xi
            agg.deltaXi = self.config.deltaXi
        elif isinstance(agg, FedMGDAPlusAggregator):
            agg.reinitialise(self.config.innerLR)

        return agg


    def _gen_cluster_centre(self, indices: List[int], models: List[nn.Module]) -> nn.Module:
        """ Takes the average of the clients assigned to each cluster to generate a new centre """
        # Here you should be using other robust aggregation algorithms to perform the centre calculation and blocking

        model = self.internalAggregator.aggregate([self.clients[i] for i in indices], [models[i] for i in indices])

        return model

    def _generate_weights(self, models: List[nn.Module]) -> List[Tensor]:
        X = []
        for model in models:
            X.append(self._generate_coords(model))

        return X

    def _generate_coords(self, model: nn.Module) -> Tensor:
        coords = torch.tensor([]).to(self.device)

        for param in model.parameters():
            coords = torch.cat((coords, param.data.view(-1)))

        return coords

    def generate_cluster_centres(self, models: List[nn.Module]) -> None:
        X = self._generate_weights(models)
        X = [model.tolist() for model in X]
        pca = PCA.pca(X, dim=4)
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit(pca)

        self.cluster_labels = kmeans.labels_
        indices: List[List[int]] = [[] for _ in range(self.cluster_count)]
        self.cluster_centres_len.zero_()

        for i, l in enumerate(self.cluster_labels):
            self.cluster_centres_len[l] += 1
            indices[l].append(i)

        print(self.cluster_labels)

        self.cluster_centres_len /= len(self.clients)
        for i, ins in enumerate(indices):
            self.cluster_centres[i] = self._gen_cluster_centre(ins, models)

    def _use_most_similar_clusters(self) -> Tuple[List[nn.Module], Tensor, List[int]]:
        num_to_take = math.floor(self.cluster_count / 2) + 1

        X = self._generate_weights(self.cluster_centres)

        sims = [[] for _ in range(self.cluster_count)]
        cos = nn.CosineSimilarity(0)

        for i, m1 in enumerate(X):
            for m2 in X:
                sim = cos(m1, m2)
                sims[i].append(sim)

        best_val = 0
        best_indices: List[int] = []
        besti = -1

        for i, s in enumerate(sims):
            indices = heapq.nlargest(num_to_take, range(len(s)), s.__getitem__)
            val = sum(s[i] for i in indices)
            if val > best_val:
                best_val = val
                best_indices = indices
                besti = i


        ps: Tensor = Tensor([p / sum(sims[besti]) for p in sims[besti]])
        best_models = [self.cluster_centres[i] for i in best_indices]

        if self.config.threshold:
            std = torch.std(ps[ps.nonzero()])
            mean = torch.mean(ps[ps.nonzero()])
            cutoff = mean - std
            ps[ps < cutoff] = 0

        ps = ps.mul(self.cluster_centres_len)
        ps /= ps.sum()

        return best_models, ps, best_indices


    def _use_closest_clusters(self) -> Tuple[List[nn.Module], Tensor, List[int]]:
        num_to_take = math.floor(self.cluster_count / 2) + 1

        X = self._generate_weights(self.cluster_centres)

        dists = [[] for _ in range(self.cluster_count)]

        for i, m1 in enumerate(X):
            for m2 in X:
                l2_dist = (m1 - m2).square().sum()
                dists[i].append(l2_dist)

        best_val = 100000000000
        best_indices: List[int] = []
        besti = -1

        for i, s in enumerate(dists):
            indices = heapq.nsmallest(num_to_take, range(len(s)), s.__getitem__)
            val = sum(s[i] for i in indices)
            if val < best_val:
                best_val = val
                best_indices = indices
                besti = i


        ps: Tensor = Tensor([p / sum(dists[besti]) for p in dists[besti]])
        ps = 1 - ps
        ps /= ps.sum()
        best_models = [self.cluster_centres[i] for i in best_indices]

        if self.config.threshold:
            std = torch.std(ps[ps.nonzero()])
            mean = torch.mean(ps[ps.nonzero()])
            cutoff = mean - std
            ps[ps < cutoff] = 0

        ps = ps.mul(self.cluster_centres_len)
        ps /= ps.sum()

        return best_models, ps, best_indices


class FakeClient:
    def __init__(self, p: float, id: int):
        self.p = p
        self.id = id
