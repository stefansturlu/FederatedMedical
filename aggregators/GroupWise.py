from aggregators.FedMGDAPlus import FedMGDAPlusAggregator
from aggregators.AFA import AFAAggregator
from copy import deepcopy
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from torch import nn, device, Tensor
from client import Client
from logger import logPrint
from typing import Dict, List
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import heapq

# Group-Wise Aggregator based on clustering
# Even though it itself does not do aggregation, it makes programatic sense to inherit attributes and functions
class GroupWiseAggregation(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, rounds: int, device: device, detectFreeRiders:bool, config: DefaultExperimentConfiguration, useAsyncClients:bool=False):
        super().__init__(clients, model, rounds, device, detectFreeRiders, useAsyncClients)

        self.config = config

        self.cluster_count = 5
        self.cluster_centres: List[nn.Module] = [None]*self.cluster_count
        self.cluster_centres_p = [0]*self.cluster_count
        self.cluster_labels = [0]*len(self.clients)

        self.internalAggregator = self._init_aggregator(config.internalAggregator)
        self.externalAggregator = self._init_aggregator(config.externalAggregator)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.config.rounds)
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


                if r % 3 == 2:
                    # Assume p value is based on size of cluster
                    best_models, ps, indices = self._use_most_similar_clusters()
                    conc_ps = [ps[i] for i in indices]

                    class FakeClient:
                        def __init__(self, p:float, id:int):
                            self.p = p
                            self.id = id

                    general = self.externalAggregator.aggregate([FakeClient(p, i) for (i, p) in enumerate(ps)], self.cluster_centres)
                    concentrated = self.externalAggregator.aggregate([FakeClient(p, i) for (i, p) in enumerate(conc_ps)], best_models)

                    for i in range(len(self.cluster_centres)):
                        if i in indices:
                            self.cluster_centres[i] = concentrated
                        else:
                            self.cluster_centres[i] = general


                    print("Concentrated test")
                    self.model = concentrated
                    roundsError[r] = self.test(testDataset)

                    print("General test")
                    self.model = general
                    roundsError[r] = self.test(testDataset)

        return roundsError


    def _init_aggregator(self, aggregator: Aggregator) -> Aggregator:
        agg = aggregator(self.clients, self.model, self.rounds, self.device, self.detectFreeRiders)
        if isinstance(agg, AFAAggregator):
            agg.xi = self.config.xi
            agg.deltaXi = self.config.deltaXi
        elif isinstance(agg, FedMGDAPlusAggregator):
            agg.reinitialise(self.config.innerLR)

        return agg


    def _init_cluster_centres(self, models: List[nn.Module]) -> None:
        if self.cluster_centres == None:
            indices: Tensor = torch.randint(high=len(models), size=(self.cluster_count,))
            self.cluster_centres = [models[i] for i in indices]

        else:
            for choice in range(self.cluster_count):
                indices = [i for i,val in enumerate(self.cluster_labels) if val == choice]
                self.cluster_centres[choice] = self._gen_cluster_centre(indices, models)
                self.cluster_centres_p[choice] = len(indices)


    def _gen_cluster_centre(self, indices: Tensor, models: List[nn.Module]) -> nn.Module:
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
        pca = self.pca(X, dim=2)
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit(pca)
        y_kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit_predict(pca)

        plt.figure()
        plt.scatter(pca[y_kmeans==0, 0], pca[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
        plt.scatter(pca[y_kmeans==1, 0], pca[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
        plt.scatter(pca[y_kmeans==2, 0], pca[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
        plt.scatter(pca[y_kmeans==3, 0], pca[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
        plt.scatter(pca[y_kmeans==4, 0], pca[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
        #Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='yellow', label = 'Centroids')
        plt.title('Clusters of Customers')
        plt.xlabel('Annual Income(k$)')
        plt.ylabel('Spending Score(1-100')
        plt.show()


        # self.pca1D(X)
        # self.pca2D(X)
        # self.pca3D(X)


        self.cluster_labels = kmeans.labels_
        indices = [[] for _ in range(self.cluster_count)]

        for i, l in enumerate(self.cluster_labels):
            self.cluster_centres_p[l] += 1
            indices[l].append(i)

        print(self.cluster_labels)

        self.cluster_centres_p = [p/len(self.clients) for p in self.cluster_centres_p]
        for i, ins in enumerate(indices):
            self.cluster_centres[i] = self._gen_cluster_centre(ins, models)


    def _use_most_similar_clusters(self) -> None:
        num_to_take = math.floor(self.cluster_count/2) + 1

        X = self._generate_weights(self.cluster_centres)
        Xl = [model.tolist() for model in X]
        kmeans = KMeans(n_clusters=num_to_take, random_state=0).fit(Xl)
        print(kmeans.labels_)


        # dists = [[] for _ in range(self.cluster_count)]

        # for i, m1 in enumerate(X):
        #     for m2 in X:
        #         d = torch.square(m1 - m2).sum().sqrt()
        #         dists[i].append(d)
        # print("dists")
        # print(dists)

        # best_val = 100000000
        # best_indices = None

        # for i, d in enumerate(dists):
        #     indices = heapq.nsmallest(num_to_take, range(len(d)), d.__getitem__)
        #     print(indices)
        #     val = sum(d[i] for i in indices)
        #     if val < best_val:
        #         best_val = val
        #         best_indices = indices

        # print("best indices")
        # print(best_indices)

        # best_models = [self.cluster_centres[i] for i in best_indices]
        # ps = [self.cluster_centres_p[i] for i in best_indices]
        # print("ps")
        # print(ps)
        # ps = [p/sum(ps) for p in ps]
        # print("ps")
        # print(ps)

        # class FakeClient:
        #     def __init__(self, p:float, id:int):
        #         self.p = p
        #         self.id = id

        # self.model = self.externalAggregator.aggregate([FakeClient(p/len(ps), i) for (i, p) in enumerate(ps)], best_models)

        # for i in best_indices:
        #     self.cluster_centres[i] = self.model


        # return best_models, ps












        sims = [[] for _ in range(self.cluster_count)]
        cos = nn.CosineSimilarity(0)

        for i, m1 in enumerate(X):
            for m2 in X:
                sim = cos(m1, m2)
                sims[i].append(sim)
        print("sims")
        print(sims)

        best_val = 0
        best_indices = None
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

        best_models = [self.cluster_centres[i] for i in best_indices]
        ps = [s for s in sims[besti]]
        ps = [p/sum(ps) for p in ps]
        print("ps")
        print(ps)

        return best_models, ps, best_indices



