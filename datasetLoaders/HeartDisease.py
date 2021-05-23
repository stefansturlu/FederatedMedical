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
