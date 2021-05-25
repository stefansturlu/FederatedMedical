from utils.typings import Errors
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
        internal: Type[Aggregator],
        external: Type[Aggregator],
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)

        self.config = config

        self.cluster_count = 5
        self.cluster_centres: List[nn.Module] = [None] * self.cluster_count
        self.cluster_centres_len = torch.zeros(self.cluster_count)
        self.cluster_labels = [0] * len(self.clients)

        self.internalAggregator = self._init_aggregator(internal)
        self.externalAggregator = self._init_aggregator(external)

        self.blocked_ps = []

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.config.rounds))
        for r in range(self.config.rounds):
            logPrint("Round... ", r)
            if True:
                self._shareModelAndTrainOnClients()
            else:
                self._shareModelAndTrainOnClients(self.cluster_centres, self.cluster_labels)

            models = self._retrieveClientModelsDict()

            # Perform Clustering
            with torch.no_grad():
                self.generate_cluster_centres(models)

                if True:
                    # Assume p value is based on size of cluster
                    best_models, ps, indices = self._use_most_similar_clusters()
                    conc_ps = [ps[i] for i in indices]
                    conc_ps = [p / sum(conc_ps) for p in conc_ps]

                    general = self.externalAggregator.aggregate(
                        [FakeClient(p, i) for (i, p) in enumerate(ps)], self.cluster_centres
                    )

                    concentrated = self.externalAggregator.aggregate(
                        [FakeClient(p, i) for (i, p) in enumerate(conc_ps)], best_models
                    )

                    # give reference to customised fed learning
                    # for i in range(len(self.cluster_centres)):
                    #     # if i in indices:
                    #     # self.cluster_centres[i] = concentrated
                    #     # else:
                    #     self.cluster_centres[i] = general

                    print("Concentrated test")
                    self.model = concentrated
                    roundsError[r] = self.test(testDataset)

                    print("General test")
                    self.model = general
                    roundsError[r] = self.test(testDataset)

        return roundsError

    def _init_aggregator(self, aggregator: Type[Aggregator]) -> Aggregator:
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

        model = self.internalAggregator.aggregate([self.clients[i] for i in indices], models)

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
        pca = PCA.pca(X, dim=2)
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit(pca)
        y_kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit_predict(pca)

        # plt.figure()
        # plt.scatter(pca[y_kmeans==0, 0], pca[y_kmeans==0, 1], s=100, c='red', label ='Cluster 0')
        # plt.scatter(pca[y_kmeans==1, 0], pca[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 1')
        # plt.scatter(pca[y_kmeans==2, 0], pca[y_kmeans==2, 1], s=100, c='green', label ='Cluster 2')
        # plt.scatter(pca[y_kmeans==3, 0], pca[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 3')
        # plt.scatter(pca[y_kmeans==4, 0], pca[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 4')
        # #Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
        # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='yellow', label = 'Centroids')
        # plt.title('Clusters of Customers')
        # plt.xlabel('Annual Income(k$)')
        # plt.ylabel('Spending Score(1-100')
        # plt.legend()
        # plt.show()

        # PCA.pca1D(X, self.clients)
        # PCA.pca2D(X, self.clients)
        # PCA.pca3D(X, self.clients)

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
        Xl = [model.tolist() for model in X]
        kmeans = KMeans(n_clusters=num_to_take, random_state=0).fit(Xl)
        print(kmeans.labels_)

        sims = [[] for _ in range(self.cluster_count)]
        cos = nn.CosineSimilarity(0)

        for i, m1 in enumerate(X):
            for m2 in X:
                sim = cos(m1, m2)
                sims[i].append(sim)
        print("sims")
        print(sims)

        best_val = 0
        best_indices: List[int] = []
        besti = -1

        for i, s in enumerate(sims):
            indices = heapq.nlargest(num_to_take, range(len(s)), s.__getitem__)
            print(indices)
            val = sum(s[i] for i in indices)
            if val > best_val:
                best_val = val
                best_indices = indices
                besti = i

        print("best indices")
        print(best_indices)

        mean = 1 / self.cluster_count
        ps: Tensor = Tensor([p / sum(sims[besti]) for p in sims[besti]])
        std = torch.std(ps[ps.nonzero()])
        cutoff = mean - std
        print("cutoff")
        print(cutoff)

        best_models = [self.cluster_centres[i] for i in best_indices]
        ps[ps < cutoff] = 0
        ps = ps.mul(self.cluster_centres_len)
        ps /= ps.sum()
        print("ps")
        print(ps)

        return best_models, ps, best_indices


class FakeClient:
    def __init__(self, p: float, id: int):
        self.p = p
        self.id = id
