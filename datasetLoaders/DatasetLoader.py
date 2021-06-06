from datasetLoaders.DatasetInterface import DatasetInterface
import os
import random
import re
from typing import List, Tuple, Type
import numpy as np
import pandas as pd
import torch
from torch import Tensor, cuda
from pandas import DataFrame

# from cn.protect import Protect
# from cn.protect.privacy import KAnonymity


class DatasetLoader:
    """Parent class used for specifying the data loading workflow """

    def getDatasets(self, percUsers: Tensor, labels: Tensor, size=(None, None)):
        raise Exception(
            "LoadData method should be override by child class, "
            "specific to the loaded dataset strategy."
        )

    @staticmethod
    def _filterDataByLabel(
        labels: Tensor, trainDataframe: DataFrame, testDataframe: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        trainDataframe = trainDataframe[trainDataframe["labels"].isin(labels.tolist())]
        testDataframe = testDataframe[testDataframe["labels"].isin(labels.tolist())]
        return trainDataframe, testDataframe

    @staticmethod
    def _splitTrainDataIntoClientDatasets(
        percUsers: Tensor, trainDataframe: DataFrame, DatasetType: Type[DatasetInterface]
    ) -> List[DatasetInterface]:
        DatasetLoader._setRandomSeeds()
        percUsers = percUsers / percUsers.sum()

        dataSplitCount = (percUsers.cpu() * len(trainDataframe)).floor().numpy()
        _, *dataSplitIndex = [
            int(sum(dataSplitCount[range(i)])) for i in range(len(dataSplitCount))
        ]

        # Sample and reset_index shuffles the dataset in-place and resets the index
        trainDataframes: List[DataFrame] = np.split(
            trainDataframe.sample(frac=1).reset_index(drop=True), indices_or_sections=dataSplitIndex
        )

        clientDatasets: List[DatasetInterface] = [
            DatasetType(clientDataframe.reset_index(drop=True))
            for clientDataframe in trainDataframes
        ]
        return clientDatasets

    @staticmethod
    def _setRandomSeeds(seed=0) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cuda.manual_seed(seed)

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
    #         DataFrame(list(ds.dataframe["data"]), columns=columns)
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
    #         labeledDataframe = DataFrame(zip(resultDataframe.values, labels))
    #         labeledDataframe.columns = ["data", "labels"]
    #         anonClientDatasets.append(datasetClass(labeledDataframe))

    #     return anonClientDatasets, clientSyntacticMappings, allColumns

    def _anonymizeTestDataset(
        self, testDataset, clientSyntacticMappings, columns, generalizedColumns
    ):

        datasetClass = testDataset.__class__
        dataframe = DataFrame(list(testDataset.dataframe["data"]), columns=columns)

        domainsSize = dict()
        quasiIds = clientSyntacticMappings[0][0].keys()
        for quasiId in quasiIds:
            domainsSize[quasiId] = dataframe[quasiId].max() - dataframe[quasiId].min()

        generalisedDataframe = DataFrame(dataframe)
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
        labeledDataframe = DataFrame(zip(resultDataframe.values, labels))
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
    def __legitMapping(entry, mapping) -> bool:
        for col in mapping:
            if not isinstance(mapping[col], str):
                if entry[col] != mapping[col]:
                    return False
            else:
                interval = np.array(re.findall(r"\d+.\d+", mapping[col]), dtype=np.float)
                if interval[0] < entry[col] or entry[col] >= interval[1]:
                    return False
        return True
