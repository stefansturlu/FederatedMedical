from abc import ABCMeta
from torch import nn
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as pca_func
from client import Client


class PCA(metaclass=ABCMeta):
    @staticmethod
    def pca3D(X, client_info: List[Client]):
        pca = pca_func(3).fit(X)
        pca_3d = pca.transform(X)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        c1, c2, c3, c4 = None, None, None, None
        for i in range(len(pca_3d)):
            if client_info[i].flip:
                c1 = ax.scatter(pca_3d[i, 0], pca_3d[i, 1], pca_3d[i, 2], c="r", marker="+")
            elif client_info[i].byz:
                c2 = ax.scatter(pca_3d[i, 0], pca_3d[i, 1], pca_3d[i, 2], c="g", marker="o")
            elif client_info[i].free:
                c3 = ax.scatter(pca_3d[i, 0], pca_3d[i, 1], pca_3d[i, 2], c="b", marker="*")
            else:
                c4 = ax.scatter(pca_3d[i, 0], pca_3d[i, 1], pca_3d[i, 2], c="y", marker=".")

        plt.legend([c1, c2, c3, c4], ["Byz", "Faulty", "Free", "Benign"])
        plt.title("Iris dataset with 3 clusters and known outcomes")
        plt.show()

    @staticmethod
    def pca2D(X, client_info: List[Client]):
        pca = pca_func(2).fit(X)
        pca_2d = pca.transform(X)

        plt.figure()
        c1, c2, c3, c4 = None, None, None, None
        for i in range(len(pca_2d)):
            if client_info[i].flip:
                c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="r", marker="+")
            elif client_info[i].byz:
                c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="g", marker="o")
            elif client_info[i].free:
                c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="b", marker="*")
            else:
                c4 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="y", marker=".")

        plt.legend([c1, c2, c3, c4], ["Byz", "Faulty", "Free", "Benign"])
        plt.title("Iris dataset with 3 clusters and known outcomes")
        plt.show()

    @staticmethod
    def pca1D(X, client_info: List[Client]):
        pca = pca_func(1).fit(X)
        pca_2d = pca.transform(X)

        plt.figure()
        c1, c2, c3, c4 = None, None, None, None
        for i in range(len(pca_2d)):
            if client_info[i].flip:
                c1 = plt.scatter(pca_2d[i], pca_2d[i], c="r", marker="+")
            elif client_info[i].byz:
                c2 = plt.scatter(pca_2d[i], pca_2d[i], c="g", marker="o")
            elif client_info[i].free:
                c3 = plt.scatter(pca_2d[i], pca_2d[i], c="b", marker="*")
            else:
                c4 = plt.scatter(pca_2d[i], pca_2d[i], c="y", marker=".")

        plt.legend([c1, c2, c3, c4], ["Byz", "Faulty", "Free", "Benign"])
        plt.title("Iris dataset with 3 clusters and known outcomes")
        plt.show()

    # dim must be between 0 and min(n_samples, n_features)
    # This is most likely len(client_info) (e.g. 30) unless you are working with a really small model
    @staticmethod
    def pca(flattened_models: nn.Module, dim=10) -> Tuple[Union[Tuple, float]]:
        return pca_func(dim).fit_transform(flattened_models)
