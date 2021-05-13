import os
import sys
from typing import List, Tuple
import kaggle
import zipfile
import cv2
import numpy as np
from pandas import DataFrame
from torch.tensor import Tensor
from torchvision import transforms

# from cn.protect import Protect
# from cn.protect.privacy import KAnonymity

from logger import logPrint
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface

import cv2
import os


###############
######## GO HERE: https://www.kaggle.com/madz2000/pneumonia-detection-using-cnn-92-6-accuracy
###############


class DatasetLoaderPneumonia(DatasetLoader):
    def __init__(self, dim=(224, 224), assembleDatasets=True):
        self.assembleDatasets = assembleDatasets
        self.dim = dim
        self.dataPath = "./data/Pneumonia"
        self.testPath = self.dataPath + "/test"
        self.trainPath = self.dataPath + "/train"
        self.labels = {"PNEUMONIA": 0, "NORMAL": 1}
        self.img_size = 150

        self.__download_data()

    def getDatasets(self, percUsers, labels, size=None):
        logPrint("Loading Pneumonia Dataset...")
        self._setRandomSeeds()
        data = self.__loadPneumoniaData()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        logPrint("Splitting datasets over clients...")
        clientDatasets = self._splitTrainDataIntoClientDatasets(
            percUsers, trainDataframe, self.PneumoniaDataset
        )
        testDataset = self.PneumoniaDataset(testDataframe, isTestDataset=True)
        return clientDatasets, testDataset

    def __loadPneumoniaData(self) -> Tuple[DataFrame, DataFrame]:
        if self.__datasetNotFound():
            logPrint(
                "Can't find train|test split files or "
                "/train, /test files not populated accordingly."
            )
            sys.exit(0)


        logPrint("Loading training images from files...")
        trainDataframe = self.__readDataframe(self.trainPath)
        logPrint("Loading testing images from files...")
        testDataframe = self.__readDataframe(self.testPath)
        logPrint("Finished loading files")

        return trainDataframe, testDataframe

    def get_img_data(self, data_dir) -> np.ndarray:
        data = []
        for label, class_num in self.labels.items():
            path = os.path.join(data_dir, label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size)) # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data, dtype=object)

    def __datasetNotFound(self) -> bool:
        return (
            not os.path.exists(self.dataPath)
            or not os.path.exists(self.dataPath + "/test")
            or not os.path.exists(self.dataPath + "/train")
            or not len(os.listdir(self.dataPath + "/test"))
            or not len(os.listdir(self.dataPath + "/train"))
        )
        # Might also want to check the number of files or subfolders


    def __readDataframe(self, path: str) -> DataFrame:
        img = self.get_img_data(path)
        dataFrame = DataFrame(img, columns=["img", "labels"])

        return dataFrame


    def __download_data(self) -> None:
        if self.__datasetNotFound():
            try:
                logPrint("Need to download the KAGGLE Pneumonia Detection Dataset")
                os.makedirs(self.dataPath)

                kaggle.api.authenticate() # Get json file from kaggle account or just manually download
                kaggle.api.dataset_download_files("paultimothymooney/chest-xray-pneumonia" , self.dataPath)
                with zipfile.ZipFile(self.dataPath + "/chest-xray-pneumonia.zip", "r") as zip_ref:
                    zip_ref.extractall(self.dataPath)
            except:
                logPrint("Failed to get files.")
                logPrint(
                    "You need to unzip (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset to {}."
                    "".format(self.dataPath)
                )
                exit(0)

    class PneumoniaDataset(DatasetInterface):
        def __init__(self, dataframe: DataFrame, isTestDataset=False):
            self.root = "./data/Pneumonia/" + ("test/" if isTestDataset else "train/")
            self.imgs: List[np.ndarray] = dataframe["img"]
            super().__init__(dataframe["labels"].values.tolist())

        def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
            imageTensor = self.__load_image(self.imgs[index])
            labelTensor = self.labels[index]
            return imageTensor, labelTensor

        @staticmethod
        def __load_image(image: np.ndarray) -> Tensor:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            imageTensor = transform(image)
            return imageTensor

