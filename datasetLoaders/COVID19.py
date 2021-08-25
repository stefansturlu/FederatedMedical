from typing import List, Optional, Tuple

import os
import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
import pandas as pd
from pandas import DataFrame
from torch.tensor import Tensor
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from logger import logPrint
import torch


class DatasetLoaderCOVID19(DatasetLoader):
    def __init__(self, dim=(224,224), ):
        # Data source: https://data.mendeley.com/datasets/jctsfj2sfn/1
        # Must be saved in ./data/
        self.dim = dim
        self.dataPath = "./data/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"
        # Check if this is correct...
        self.COVIDLabelsDict = {"pneumonia": 0, "normal": 1, "COVID-19": 2}
    
    def getDatasets(
        self, percUsers: Tensor, labels: Tensor, size: Optional[Tuple[int, int]] = None, nonIID = False, alpha = 0.1, percServerData = 0
    ) -> Tuple[List[DatasetInterface], DatasetInterface]:
        logPrint("Loading COVID19...")
        self._setRandomSeeds()
        data = self.__loadCOVID19Data()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        serverDataset = []
        if percServerData>0:
            # Knowledge distillation requires server data
            msk = np.random.rand(len(trainDataframe)) < percServerData
            serverDataframe, trainDataframe = trainDataframe[msk], trainDataframe[~msk]
            serverDataset = self.COVIDDataset(serverDataframe.reset_index(drop=True))
            logPrint(f"Lengths of server {len(serverDataframe)} and train {len(trainDataframe)}")
        else:
            logPrint(f"Lengths of server {0} and train {len(trainDataframe)}")
        clientDatasets = self._splitTrainDataIntoClientDatasets(
            percUsers, trainDataframe, self.COVIDDataset, nonIID, alpha
        )
        testDataset = self.COVIDDataset(testDataframe)
        return clientDatasets, testDataset, serverDataset

    def __loadCOVID19Data(self) -> Tuple[DataFrame, DataFrame]:

            
        if not os.path.exists(self.dataPath+"/CovidData_224x224.pkl"):
            self.__pickleData()
        
        logPrint("Loading dataframe")
        # if not exist, transform images and save as pickle
        dataFrame = pd.read_pickle(self.dataPath+"/CovidData_224x224.pkl")
        print("dataFrame:")
        print(dataFrame)

        # Split into train and test
        trainDataframe = dataFrame.sample(frac=0.8, random_state = 0)
        testDataframe = dataFrame.drop(trainDataframe.index)
        trainDataframe.columns = testDataframe.columns = ["data", "labels"]
        
        trainDataframe.reset_index(drop=True, inplace=True)
        testDataframe.reset_index(drop=True, inplace=True)

        return trainDataframe, testDataframe
    
    def __pickleData(self):
        if not os.path.exists(self.dataPath):
            print("The dataset must be downloaded and put in the data folder.")
        logPrint("Pickling Covid data...")
        transform = transforms.Compose([
            transforms.Resize(size=self.dim),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            root=self.dataPath,
            transform=transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )
        
        df = DataFrame(columns=['data','labels'])
        for batch_idx, (data, target) in enumerate(train_loader):
            logPrint(f"Batch {batch_idx} of {len(train_loader)}")
            dataFrame = DataFrame(zip(data.view(-1,3*224*224).numpy(), target.numpy()))
            dataFrame.columns = df.columns
            df = df.append(dataFrame, ignore_index=True)
        
        df.to_pickle(self.dataPath+"/CovidData_224x224.pkl")
        logPrint("Pickle saved.")


    class COVIDDataset(DatasetInterface):
        def __init__(self, dataframe):
            self.data = torch.stack(
                [torch.from_numpy(data).view(-1,224,224) for data in dataframe["data"].values], dim=0
            )
            super().__init__(dataframe["labels"].values.astype(int))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def to(self, device):
            self.data = self.data.to(device)
            self.labels = self.labels.to(device)


