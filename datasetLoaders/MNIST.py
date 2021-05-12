from typing import Tuple
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
import pandas as pd
from torch.tensor import Tensor
from torchvision import transforms, datasets
from functools import reduce
from logger import logPrint
import torch



class DatasetLoaderMNIST(DatasetLoader):
    def getDatasets(self, percUsers: Tensor, labels: Tensor, size=None):
        logPrint("Loading MNIST...")
        self._setRandomSeeds()
        data = self.__loadMNISTData()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        clientDatasets = self._splitTrainDataIntoClientDatasets(
            percUsers, trainDataframe, self.MNISTDataset
        )
        testDataset = self.MNISTDataset(testDataframe)
        return clientDatasets, testDataset

    @staticmethod
    def __loadMNISTData() -> Tuple[pd.DataFrame, pd.DataFrame]:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        # if not exist, download mnist dataset
        trainSet = datasets.MNIST("data", train=True, transform=trans, download=True)
        testSet = datasets.MNIST("data", train=False, transform=trans, download=True)

        # Scale pixel intensities to [-1, 1]
        xTrain = trainSet.train_data
        xTrain = 2 * (xTrain.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        xTrain = xTrain.flatten(1, 2).numpy()
        yTrain = trainSet.train_labels.numpy()

        # Scale pixel intensities to [-1, 1]
        xTest = testSet.test_data.clone().detach()
        xTest = 2 * (xTest.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        xTest = xTest.flatten(1, 2).numpy()
        yTest = testSet.test_labels.numpy()

        trainDataframe = pd.DataFrame(zip(xTrain, yTrain))
        testDataframe = pd.DataFrame(zip(xTest, yTest))
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

