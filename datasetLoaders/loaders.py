import os
import random
import re
import sys
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
from PIL import Image
from torch.tensor import Tensor

# import cn.protect.quality as quality
# from cn.protect.hierarchy import OrderHierarchy
from torch.utils.data import Dataset
from torchvision import transforms, datasets

# from cn.protect import Protect
# from cn.protect.privacy import KAnonymity
from functools import reduce

from logger import logPrint


class DatasetInterface(Dataset):
    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        raise Exception("Method should be implemented in subclass.")

    def getInputSize(self):
        raise Exception(
            "Method should be implemented by subclasses where "
            "models requires input size update (based on dataset)."
        )

    def zeroLabels(self):
        self.labels = torch.zeros(len(self.labels), dtype=torch.long)


class DatasetLoader:
    """Parent class used for specifying the data loading workflow """

    def getDatasets(self, percUsers, labels, size=(None, None)):
        raise Exception(
            "LoadData method should be override by child class, "
            "specific to the loaded dataset strategy."
        )

    @staticmethod
    def _filterDataByLabel(labels: Tensor[int], trainDataframe, testDataframe):
        trainDataframe = trainDataframe[trainDataframe["labels"].isin(labels)]
        testDataframe = testDataframe[testDataframe["labels"].isin(labels)]
        return trainDataframe, testDataframe

    @staticmethod
    def _splitTrainDataIntoClientDatasets(percUsers: Tensor[float], trainDataframe, DatasetType):
        DatasetLoader._setRandomSeeds()
        percUsers = percUsers / percUsers.sum()

        dataSplitCount = (percUsers * len(trainDataframe)).floor().numpy()
        _, *dataSplitIndex = [
            int(sum(dataSplitCount[range(i)])) for i in range(len(dataSplitCount))
        ]

        # Sample and reset_index shuffles the dataset in-place and resets the index
        trainDataframes = np.split(
            trainDataframe.sample(frac=1).reset_index(drop=True), indices_or_sections=dataSplitIndex
        )

        clientDatasets = [
            DatasetType(clientDataframe.reset_index(drop=True))
            for clientDataframe in trainDataframes
        ]
        return clientDatasets

    @staticmethod
    def _setRandomSeeds(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # When anonymizing the clients' datasets using  _anonymizeClientDatasets the function passed as
    #  parameter should take as parameter the cn.protect object and set ds specific generalisations
    # @staticmethod
    # def _anonymizeClientDatasets(
    #     clientDatasets, columns, k, quasiIds, setHierarchiesMethod
    # ):

    #     datasetClass = clientDatasets[0].__class__

    #     resultDataframes = []
    #     clientSyntacticMappings = []

    #     dataframes = [
    #         pd.DataFrame(list(ds.dataframe["data"]), columns=columns)
    #         for ds in clientDatasets
    #     ]
    #     for dataframe in dataframes:
    #         anonIndex = (
    #             dataframe.groupby(quasiIds)[dataframe.columns[0]].transform("size") >= k
    #         )

    #         anonDataframe = dataframe[anonIndex]
    #         needProtectDataframe = dataframe[~anonIndex]

    #         # Might want to ss those for the report:
    #         # print(anonDataframe)
    #         # print(needProtectDataframe)

    #         protect = Protect(needProtectDataframe, KAnonymity(k))
    #         protect.quality_model = quality.Loss()
    #         # protect.quality_model = quality.Classification()
    #         protect.suppression = 0

    #         for qid in quasiIds:
    #             protect.itypes[qid] = "quasi"

    #         setHierarchiesMethod(protect)

    #         protectedDataframe = protect.protect()
    #         mappings = protectedDataframe[quasiIds].drop_duplicates().to_dict("records")
    #         clientSyntacticMappings.append(mappings)
    #         protectedDataframe = pd.get_dummies(protectedDataframe)

    #         resultDataframe = (
    #             pd.concat([anonDataframe, protectedDataframe]).fillna(0).sort_index()
    #         )
    #         resultDataframes.append(resultDataframe)

    #     # All clients datasets should have same columns
    #     allColumns = set().union(*[df.columns.values for df in resultDataframes])
    #     for resultDataframe in resultDataframes:
    #         for col in allColumns - set(resultDataframe.columns.values):
    #             resultDataframe[col] = 0

    #     # Create new datasets by adding the labels to
    #     anonClientDatasets = []
    #     for resultDataframe, initialDataset in zip(resultDataframes, clientDatasets):
    #         labels = initialDataset.dataframe["labels"].values
    #         labeledDataframe = pd.DataFrame(zip(resultDataframe.values, labels))
    #         labeledDataframe.columns = ["data", "labels"]
    #         anonClientDatasets.append(datasetClass(labeledDataframe))

    #     return anonClientDatasets, clientSyntacticMappings, allColumns

    def _anonymizeTestDataset(
        self, testDataset, clientSyntacticMappings, columns, generalizedColumns
    ):

        datasetClass = testDataset.__class__
        dataframe = pd.DataFrame(list(testDataset.dataframe["data"]), columns=columns)

        domainsSize = dict()
        quasiIds = clientSyntacticMappings[0][0].keys()
        for quasiId in quasiIds:
            domainsSize[quasiId] = dataframe[quasiId].max() - dataframe[quasiId].min()

        generalisedDataframe = pd.DataFrame(dataframe)
        ungeneralisedIndex = []
        for i in range(len(dataframe)):
            legitMappings = []
            for clientMappings in clientSyntacticMappings:
                legitMappings += [
                    mapping
                    for mapping in clientMappings
                    if self.__legitMapping(dataframe.iloc[i], mapping)
                ]
            if legitMappings:
                # leastGeneralMapping = reduce(self.__leastGeneral, legitMappings)
                leastGeneralMapping = legitMappings[0]
                for legitMapping in legitMappings[1:]:
                    leastGeneralMapping = self.__leastGeneral(
                        leastGeneralMapping, legitMapping, domainsSize
                    )

                for col in leastGeneralMapping:
                    generalisedDataframe[col][i] = leastGeneralMapping[col]
            else:
                ungeneralisedIndex.append(i)
                generalisedDataframe = generalisedDataframe.drop(i)

        generalisedDataframe = pd.get_dummies(generalisedDataframe)
        ungeneralisedDataframe = dataframe.iloc[ungeneralisedIndex]

        resultDataframe = (
            pd.concat([ungeneralisedDataframe, generalisedDataframe]).fillna(0).sort_index()
        )
        for col in generalizedColumns - set(resultDataframe.columns.values):
            resultDataframe[col] = 0

        labels = testDataset.dataframe["labels"].values
        labeledDataframe = pd.DataFrame(zip(resultDataframe.values, labels))
        labeledDataframe.columns = ["data", "labels"]

        return datasetClass(labeledDataframe)

    @staticmethod
    def __leastGeneral(map1, map2, domainSize):
        map1Generality = map2Generality = 0
        for col in map1:
            if isinstance(map1[col], str):
                interval = np.array(re.findall(r"\d+.\d+", map1[col]), dtype=np.float)
                map1Generality += (interval[1] - interval[0]) / domainSize[col]

        for col in map2:
            if isinstance(map1[col], str):
                interval = np.array(re.findall(r"\d+.\d+", map2[col]), dtype=np.float)
                map2Generality += (interval[1] - interval[0]) / domainSize[col]

        return map1 if map1Generality <= map2Generality else map2

    @staticmethod
    def __legitMapping(entry, mapping):
        for col in mapping:
            if not isinstance(mapping[col], str):
                if entry[col] != mapping[col]:
                    return False
            else:
                interval = np.array(re.findall(r"\d+.\d+", mapping[col]), dtype=np.float)
                if interval[0] < entry[col] or entry[col] >= interval[1]:
                    return False
        return True


class DatasetLoaderMNIST(DatasetLoader):
    def getDatasets(self, percUsers: Tensor[float], labels: Tensor[int], size=None):
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
    def __loadMNISTData():
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


class DatasetLoaderCOVIDx(DatasetLoader):
    def __init__(self, dim=(224, 224), assembleDatasets=True):
        self.assembleDatasets = assembleDatasets
        self.dim = dim
        self.dataPath = "./data/COVIDx"
        self.testCSV = self.dataPath + "/test_split_v2.txt"
        self.trainCSV = self.dataPath + "/train_split_v2.txt"
        self.COVIDxLabelsDict = {"pneumonia": 0, "normal": 1, "COVID-19": 2}

    def getDatasets(self, percUsers, labels, size=None):
        logPrint("Loading COVIDx...")
        self._setRandomSeeds()
        data = self.__loadCOVIDxData(*size)
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        clientDatasets = self._splitTrainDataIntoClientDatasets(
            percUsers, trainDataframe, self.COVIDxDataset
        )
        testDataset = self.COVIDxDataset(testDataframe, isTestDataset=True)
        return clientDatasets, testDataset

    def __loadCOVIDxData(self, trainSize, testSize):
        if self.__datasetNotFound():
            logPrint(
                "Can't find train|test split .txt files or "
                "/train, /test files not populated accordingly."
            )
            if not self.assembleDatasets:
                sys.exit(0)

            logPrint("Proceeding to assemble dataset from downloaded resources.")
            self.__joinDatasets()

        trainDataframe = self.__readDataframe(self.trainCSV, trainSize)
        testDataframe = self.__readDataframe(self.testCSV, testSize)

        return trainDataframe, testDataframe

    def __datasetNotFound(self) -> bool:
        if (
            not os.path.exists(self.dataPath + "/test_split_v2.txt")
            or not os.path.exists(self.dataPath + "/train_split_v2.txt")
            or not os.path.exists(self.dataPath + "/test")
            or not os.path.exists(self.dataPath + "/train")
            or not len(os.listdir(self.dataPath + "/test"))
            or not len(os.listdir(self.dataPath + "/train"))
        ):
            # Might also want to check that files count of
            # /test, /train folder match .txt files
            return True
        return False

    def __readDataframe(self, file, size):
        dataFrame = pd.read_csv(
            file,
            names=["id", "fileNames", "labels"],
            sep=" ",
            header=None,
            usecols=[1, 2],
        )
        dataFrame["labels"] = dataFrame["labels"].map(lambda label: self.COVIDxLabelsDict[label])
        return dataFrame.head(size)

    def __joinDatasets(self):
        dataSources = [
            "/covid-chestxray-dataset",
            "/rsna-kaggle-dataset",
            "/Figure1-covid-chestxray-dataset",
        ]
        if not len(os.listdir(self.dataPath + dataSources[0])):
            logPrint(
                "You need to clone https://github.com/ieee8023/covid-chestxray-dataset to {}."
                "".format(self.dataPath + dataSources[0])
            )
            exit(0)
        if not len(os.listdir(self.dataPath + dataSources[1])):
            logPrint(
                "You need to unzip (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset to {}."
                "".format(self.dataPath + dataSources[1])
            )
            exit(0)

        COPY_FILE = True
        if COPY_FILE:
            if not os.path.exists(self.dataPath + "/train"):
                os.makedirs(self.dataPath + "/train")
            if not os.path.exists(self.dataPath + "/test"):
                os.makedirs(self.dataPath + "/test")

        # path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
        imgPath = self.dataPath + dataSources[0] + "/images"
        csvPath = self.dataPath + dataSources[0] + "/metadata.csv"

        # Path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
        kaggle_dataPath = self.dataPath + "/rsna-kaggle-dataset"
        kaggle_csvname = "stage_2_detailed_class_info.csv"  # get all the normal from here
        kaggle_csvname2 = (
            "stage_2_train_labels.csv"  # get all the 1s from here since 1 indicate pneumonia
        )
        kaggle_imgPath = "stage_2_train_images"

        # parameters for COVIDx dataset
        train = []
        test = []
        test_count = {"normal": 0, "pneumonia": 0, "COVID-19": 0}
        train_count = {"normal": 0, "pneumonia": 0, "COVID-19": 0}

        mapping = dict()
        mapping["COVID-19"] = "COVID-19"
        mapping["SARS"] = "pneumonia"
        mapping["MERS"] = "pneumonia"
        mapping["Streptococcus"] = "pneumonia"
        mapping["Normal"] = "normal"
        mapping["Lung Opacity"] = "pneumonia"
        mapping["1"] = "pneumonia"

        # train/test split
        split = 0.1

        # adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision./datasets.py#L814
        csv = pd.read_csv(csvPath, nrows=None)
        idx_pa = csv["view"] == "PA"  # Keep only the PA view
        csv = csv[idx_pa]

        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
        pathologies = [
            "Pneumonia",
            "Viral Pneumonia",
            "Bacterial Pneumonia",
            "No Finding",
        ] + pneumonias
        pathologies = sorted(pathologies)

        # get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset
        # stored as patient id, image filename and label
        filename_label = {"normal": [], "pneumonia": [], "COVID-19": []}
        count = {"normal": 0, "pneumonia": 0, "COVID-19": 0}
        print(csv.keys())
        for index, row in csv.iterrows():
            f = row["finding"]
            if f in mapping:
                count[mapping[f]] += 1
                entry = [int(row["patientid"]), row["filename"], mapping[f]]
                filename_label[mapping[f]].append(entry)

        print("Data distribution from covid-chestxray-dataset:")
        print(count)

        # add covid-chestxray-dataset into COVIDx dataset
        # since covid-chestxray-dataset doesn't have test dataset
        # split into train/test by patientid
        # for COVIDx:
        # patient 8 is used as non-COVID19 viral test
        # patient 31 is used as bacterial test
        # patients 19, 20, 36, 42, 86 are used as COVID-19 viral test

        for key in filename_label.keys():
            arr = np.array(filename_label[key])
            if arr.size == 0:
                continue
            # split by patients
            # num_diff_patients = len(np.unique(arr[:,0]))
            # num_test = max(1, round(split*num_diff_patients))
            # select num_test number of random patients
            if key == "pneumonia":
                test_patients = ["8", "31"]
            elif key == "COVID-19":
                test_patients = [
                    "19",
                    "20",
                    "36",
                    "42",
                    "86",
                ]  # random.sample(list(arr[:,0]), num_test)
            else:
                test_patients = []
            print("Key: ", key)
            print("Test patients: ", test_patients)
            # go through all the patients
            for patient in arr:
                if patient[0] in test_patients:
                    if COPY_FILE:
                        copyfile(
                            os.path.join(imgPath, patient[1]),
                            os.path.join(self.dataPath, "test", patient[1]),
                        )
                        test.append(patient)
                        test_count[patient[2]] += 1
                    else:
                        print("WARNING: passing copy file.")
                        break
                else:
                    if COPY_FILE:
                        copyfile(
                            os.path.join(imgPath, patient[1]),
                            os.path.join(self.dataPath, "train", patient[1]),
                        )
                        train.append(patient)
                        train_count[patient[2]] += 1

                    else:
                        print("WARNING: passing copy file.")
                        break

        print("test count: ", test_count)
        print("train count: ", train_count)

        # add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

        print(kaggle_dataPath)
        csv_normal = pd.read_csv(os.path.join(kaggle_dataPath, kaggle_csvname), nrows=None)
        csv_pneu = pd.read_csv(os.path.join(kaggle_dataPath, kaggle_csvname2), nrows=None)
        patients = {"normal": [], "pneumonia": []}

        for index, row in csv_normal.iterrows():
            if row["class"] == "Normal":
                patients["normal"].append(row["patientId"])

        for index, row in csv_pneu.iterrows():
            if int(row["Target"]) == 1:
                patients["pneumonia"].append(row["patientId"])

        for key in patients.keys():
            arr = np.array(patients[key])
            if arr.size == 0:
                continue
            # split by patients
            # num_diff_patients = len(np.unique(arr))
            # num_test = max(1, round(split*num_diff_patients))
            # '/content/COVID-Net/'
            test_patients = np.load(
                self.dataPath + "/COVID-Net/rsna_test_patients_{}.npy" "".format(key)
            )  # random.sample(list(arr), num_test)
            # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
            for patient in arr:
                ds = dicom.dcmread(os.path.join(kaggle_dataPath, kaggle_imgPath, patient + ".dcm"))
                pixel_array_numpy = ds.pixel_array
                imgname = patient + ".png"
                if patient in test_patients:
                    if COPY_FILE:
                        cv2.imwrite(
                            os.path.join(self.dataPath, "test", imgname),
                            pixel_array_numpy,
                        )
                        test.append([patient, imgname, key])
                        test_count[key] += 1
                    else:
                        print("WARNING: passing copy file.")
                        break
                else:
                    if COPY_FILE:
                        cv2.imwrite(
                            os.path.join(self.dataPath, "train", imgname),
                            pixel_array_numpy,
                        )
                        train.append([patient, imgname, key])
                        train_count[key] += 1
                    else:
                        print("WARNING: passing copy file.")
                        break
        print("test count: ", test_count)
        print("train count: ", train_count)

        # final stats
        print("Final stats")
        print("Train count: ", train_count)
        print("Test count: ", test_count)
        print("Total length of train: ", len(train))
        print("Total length of test: ", len(test))

        # export to train and test csv
        # format as patientid, filename, label - separated by a space
        train_file = open(self.dataPath + "/train_split_v2.txt", "w")
        for sample in train:
            info = str(sample[0]) + " " + sample[1] + " " + sample[2] + "\n"
            train_file.write(info)
        train_file.close()

        test_file = open(self.dataPath + "/test_split_v2.txt", "w")
        for sample in test:
            info = str(sample[0]) + " " + sample[1] + " " + sample[2] + "\n"
            test_file.write(info)
        test_file.close()

    class COVIDxDataset(DatasetInterface):
        def __init__(self, dataframe, isTestDataset=False):
            self.root = "./data/COVIDx/" + ("test/" if isTestDataset else "train/")
            self.paths = dataframe["fileNames"]
            super().__init__(dataframe["labels"].values)

        def __getitem__(self, index):
            imageTensor = self.__load_image(self.root + self.paths[index])
            labelTensor = self.labels[index]
            return imageTensor, labelTensor

        @staticmethod
        def __load_image(img_path: str) -> Image:
            if not os.path.exists(img_path):
                print("IMAGE DOES NOT EXIST {}".format(img_path))
            image = Image.open(img_path).convert("RGB")
            image = image.resize((224, 224)).convert("RGB")

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            # if(imageTensor.size(0)>1):
            #     #print(img_path," > 1 channels")
            #     imageTensor = imageTensor.mean(dim=0,keepdim=True)
            imageTensor = transform(image)
            return imageTensor


# class DatasetLoaderDiabetes(DatasetLoader):
#     def __init__(self, requiresAnonymization=False):
#         self.requireDatasetAnonymization = requiresAnonymization

#         # Parameters required by k-anonymity enforcement
#         self.k = 4
#         self.quasiIds = ["Pregnancies", "Age"]

#     def getDatasets(self, percUsers, labels, size=None):
#         logPrint("Loading Diabetes data...")
#         self._setRandomSeeds()
#         trainDataframe, testDataframe, columns = self.__loadDiabetesData()
#         trainDataframe, testDataframe = self._filterDataByLabel(
#             labels, trainDataframe, testDataframe
#         )
#         clientDatasets = self._splitTrainDataIntoClientDatasets(
#             percUsers, trainDataframe, self.DiabetesDataset
#         )
#         testDataset = self.DiabetesDataset(testDataframe)

#         if self.requireDatasetAnonymization:
#             clientAnonymizationResults = self._anonymizeClientDatasets(
#                 clientDatasets, columns, 4, self.quasiIds, self.__setHierarchies
#             )
#             (
#                 clientDatasets,
#                 syntacticMappings,
#                 generalizedColumns,
#             ) = clientAnonymizationResults
#             testDataset = self._anonymizeTestDataset(
#                 testDataset, syntacticMappings, columns, generalizedColumns
#             )

#         return clientDatasets, testDataset

#     @staticmethod
#     def __loadDiabetesData(dataBinning=False):
#         data = pd.read_csv("data/Diabetes/diabetes.csv")
#         # Shuffle
#         data = data.sample(frac=1).reset_index(drop=True)

#         # Handling Missing Data¶
#         data["BMI"] = data.BMI.mask(data.BMI == 0, (data["BMI"].mean(skipna=True)))
#         data["BloodPressure"] = data.BloodPressure.mask(
#             data.BloodPressure == 0, (data["BloodPressure"].mean(skipna=True))
#         )
#         data["Glucose"] = data.Glucose.mask(
#             data.Glucose == 0, (data["Glucose"].mean(skipna=True))
#         )

#         # data = data.drop(['Insulin'], axis=1)
#         # data = data.drop(['SkinThickness'], axis=1)
#         # data = data.drop(['DiabetesPedigreeFunction'], axis=1)

#         labels = data["Outcome"]
#         data = data.drop(["Outcome"], axis=1)

#         if dataBinning:
#             data["Age"] = data["Age"].astype(int)
#             data.loc[data["Age"] <= 16, "Age"] = 0
#             data.loc[(data["Age"] > 16) & (data["Age"] <= 32), "Age"] = 1
#             data.loc[(data["Age"] > 32) & (data["Age"] <= 48), "Age"] = 2
#             data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age"] = 3
#             data.loc[data["Age"] > 64, "Age"] = 4

#             data["Glucose"] = data["Glucose"].astype(int)
#             data.loc[data["Glucose"] <= 80, "Glucose"] = 0
#             data.loc[(data["Glucose"] > 80) & (data["Glucose"] <= 100), "Glucose"] = 1
#             data.loc[(data["Glucose"] > 100) & (data["Glucose"] <= 125), "Glucose"] = 2
#             data.loc[(data["Glucose"] > 125) & (data["Glucose"] <= 150), "Glucose"] = 3
#             data.loc[data["Glucose"] > 150, "Glucose"] = 4

#             data["BloodPressure"] = data["BloodPressure"].astype(int)
#             data.loc[data["BloodPressure"] <= 50, "BloodPressure"] = 0
#             data.loc[
#                 (data["BloodPressure"] > 50) & (data["BloodPressure"] <= 65),
#                 "BloodPressure",
#             ] = 1
#             data.loc[
#                 (data["BloodPressure"] > 65) & (data["BloodPressure"] <= 80),
#                 "BloodPressure",
#             ] = 2
#             data.loc[
#                 (data["BloodPressure"] > 80) & (data["BloodPressure"] <= 100),
#                 "BloodPressure",
#             ] = 3
#             data.loc[data["BloodPressure"] > 100, "BloodPressure"] = 4

#         xTrain = data.head(int(len(data) * 0.8)).values
#         xTest = data.tail(int(len(data) * 0.2)).values
#         yTrain = labels.head(int(len(data) * 0.8)).values
#         yTest = labels.tail(int(len(data) * 0.2)).values

#         trainDataframe = pd.DataFrame(zip(xTrain, yTrain))
#         testDataframe = pd.DataFrame(zip(xTest, yTest))
#         trainDataframe.columns = testDataframe.columns = ["data", "labels"]

#         return trainDataframe, testDataframe, data.columns

#     # @staticmethod
#     # def __setHierarchies(protect):
#     #     protect.hierarchies.Age = OrderHierarchy("interval", 1, 5, 2, 2, 2)
#     #     protect.hierarchies.Pregnancies = OrderHierarchy("interval", 1, 2, 2, 2, 2)

#     class DiabetesDataset(DatasetInterface):
#         def __init__(self, dataframe):
#             self.dataframe = dataframe
#             self.data = torch.stack(
#                 [torch.from_numpy(data) for data in dataframe["data"].values], dim=0
#             ).float()
#             super().__init__(dataframe["labels"].values)

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, index):
#             return self.data[index], self.labels[index]

#         def getInputSize(self):
#             return len(self.dataframe["data"][0])


# class DatasetLoaderHeartDisease(DatasetLoader):
#     def __init__(self, requiresAnonymization=False):
#         self.requireDatasetAnonymization = requiresAnonymization
#         # Parameters required by k-anonymity enforcement
#         self.k = 2
#         self.quasiIds = ["age", "sex"]

#     def getDatasets(self, percUsers, labels, size=None):
#         logPrint("Loading Heart Disease data...")
#         self._setRandomSeeds()
#         trainDataframe, testDataframe, columns = self.__loadHeartDiseaseData()
#         trainDataframe, testDataframe = self._filterDataByLabel(
#             labels, trainDataframe, testDataframe
#         )
#         clientDatasets = self._splitTrainDataIntoClientDatasets(
#             percUsers, trainDataframe, self.HeartDiseaseDataset
#         )
#         testDataset = self.HeartDiseaseDataset(testDataframe)

#         if self.requireDatasetAnonymization:
#             clientAnonymizationResults = self._anonymizeClientDatasets(
#                 clientDatasets, columns, 4, self.quasiIds, self.__setHierarchies
#             )
#             (
#                 clientDatasets,
#                 syntacticMappings,
#                 generalizedColumns,
#             ) = clientAnonymizationResults
#             testDataset = self._anonymizeTestDataset(
#                 testDataset, syntacticMappings, columns, generalizedColumns
#             )

#         return clientDatasets, testDataset

#     @staticmethod
#     def __loadHeartDiseaseData():
#         trainData = pd.read_csv("data/HeartDisease/train.csv")
#         testData = pd.read_csv("data/HeartDisease/test.csv")
#         # Shuffle train data
#         trainData = trainData.sample(frac=1).reset_index(drop=True)

#         trainLabels = (trainData["num"] != 0).astype(int)
#         trainData = trainData.drop(["num"], axis=1)
#         testLabels = (testData["num"] != 0).astype(int)
#         testData = testData.drop(["num"], axis=1)

#         trainDataframe = pd.DataFrame(zip(trainData.values, trainLabels))
#         testDataframe = pd.DataFrame(zip(testData.values, testLabels))
#         trainDataframe.columns = testDataframe.columns = ["data", "labels"]

#         return trainDataframe, testDataframe, trainData.columns

#     @staticmethod
#     def __setHierarchies(protect):
#         protect.hierarchies.age = OrderHierarchy("interval", 1, 2, 2, 2, 2)
#         protect.hierarchies.sex = OrderHierarchy("interval", 2, 2)

#     class HeartDiseaseDataset(DatasetInterface):
#         def __init__(self, dataframe):
#             self.dataframe = dataframe
#             self.data = torch.stack(
#                 [torch.from_numpy(data) for data in dataframe["data"].values], dim=0
#             ).float()
#             super().__init__(dataframe["labels"].values)

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, index):
#             return self.data[index], self.labels[index]

#         def getInputSize(self):
#             return len(self.dataframe["data"][0])
