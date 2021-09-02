from typing import List, Optional, Tuple

import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
from pandas import DataFrame
from torch.tensor import Tensor
from torchvision import transforms, datasets
from logger import logPrint
import torch


class DatasetLoaderMNIST(DatasetLoader):
    def getDatasets(
        self,
        percUsers: Tensor,
        labels: Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=False,
        alpha=0.1,
        percServerData=0,
    ) -> Tuple[List[DatasetInterface], DatasetInterface]:
        logPrint("Loading MNIST...")
        self._setRandomSeeds()
        data = self.__loadMNISTData()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        serverDataset = []
        if percServerData > 0:
            # Knowledge distillation requires server data
            msk = np.random.rand(len(trainDataframe)) < percServerData
            serverDataframe, trainDataframe = trainDataframe[msk], trainDataframe[~msk]
            serverDataset = self.MNISTDataset(serverDataframe.reset_index(drop=True))
            logPrint(f"Lengths of server {len(serverDataframe)} and train {len(trainDataframe)}")
        else:
            logPrint(f"Lengths of server {0} and train {len(trainDataframe)}")
        clientDatasets = self._splitTrainDataIntoClientDatasets(
            percUsers, trainDataframe, self.MNISTDataset, nonIID, alpha
        )
        testDataset = self.MNISTDataset(testDataframe)
        return clientDatasets, testDataset, serverDataset

    @staticmethod
    def __loadMNISTData() -> Tuple[DataFrame, DataFrame]:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        # if not exist, download mnist dataset
        trainSet = datasets.MNIST("data", train=True, transform=trans, download=True)
        testSet = datasets.MNIST("data", train=False, transform=trans, download=True)

        # Scale pixel intensities to [-1, 1]
        xTrain: Tensor = trainSet.train_data
        xTrain = 2 * (xTrain.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        xTrain = xTrain.flatten(1, 2).numpy()
        yTrain = trainSet.train_labels.numpy()

        # Scale pixel intensities to [-1, 1]
        xtest: Tensor = testSet.test_data.clone().detach()
        xtest = 2 * (xtest.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        xTest: np.ndarray = xtest.flatten(1, 2).numpy()
        yTest: np.ndarray = testSet.test_labels.numpy()

        trainDataframe = DataFrame(zip(xTrain, yTrain))
        testDataframe = DataFrame(zip(xTest, yTest))
        trainDataframe.columns = testDataframe.columns = ["data", "labels"]

        return trainDataframe, testDataframe

    class MNISTDataset(DatasetInterface):
        def __init__(self, dataframe):
            self.data = torch.stack(
                [torch.from_numpy(data) for data in dataframe["data"].values], dim=0
            )
            super().__init__(dataframe["labels"].values)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def to(self, device):
            self.data = self.data.to(device)
            self.labels = self.labels.to(device)
