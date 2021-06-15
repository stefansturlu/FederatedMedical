from torch import nn, Tensor
from torchvision import models


class Classifier(nn.Module):
    """
    Classified that is a standard ResNet model
    """
    def __init__(self, classes=3, model="resnet18"):
        super(Classifier, self).__init__()
        if model == "resnet18":
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.fc = nn.Linear(512, classes)

        elif model == "resnext50":
            self.cnn = models.resnext50_32x4d(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.cnn(x)
