import torch
from torch.utils.data import Dataset


class DatasetInterface(Dataset):
    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        raise Exception("Method should be implemented in subclass.")

    def getInputSize(self):
        raise Exception(
            "Method should be implemented by subclasses where "
            "models requires input size update (based on dataset)."
        )

    def zeroLabels(self) -> None:
        """
        Sets all labels to zero
        """
        self.labels = torch.zeros(len(self.labels), dtype=torch.long)

    def setLabels(self, value: int) -> None:
        """
        Sets all labels to the given value
        """
        self.labels = torch.zeros(len(self.labels), dtype=torch.long) + value
