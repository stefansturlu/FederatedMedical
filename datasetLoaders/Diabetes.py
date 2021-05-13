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
