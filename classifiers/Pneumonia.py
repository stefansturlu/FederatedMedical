from torch import Tensor
from torch.nn import Sequential, Module, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, Flatten, Linear, ReLU, Sigmoid, functional as F
from torch.nn.modules.padding import ZeroPad2d


class Classifier(Module):
   def __init__(self):
      super(Classifier, self).__init__()

      self.layer1 = Sequential(
         Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1),
         ReLU(),
         BatchNorm2d(32),
         ZeroPad2d([1,0,1,0]),
         MaxPool2d(kernel_size=(2,2), stride=2)
      )

      self.layer2 = Sequential(
         Conv2d(32, 64, (3,3), padding=1),
         ReLU(),
         Dropout2d(0.1),
         BatchNorm2d(64),
         ZeroPad2d([1,0,1,0]),
         MaxPool2d((2,2), stride=2)
      )

      self.layer3 = Sequential(
         Conv2d(64, 64, (3,3), padding=1),
         ReLU(),
         Dropout2d(0.2),
         BatchNorm2d(64),
         ZeroPad2d([1,0,1,0]),
         MaxPool2d((2,2), stride=2)
      )

      self.layer4 = Sequential(
         Conv2d(64, 128, (3,3), padding=1),
         ReLU(),
         Dropout2d(0.2),
         BatchNorm2d(128),
         ZeroPad2d([1,0,1,0]),
         MaxPool2d((2,2), stride=2)
      )

      self.layer5 = Sequential(
         Conv2d(128, 256, (3,3), padding=1),
         ReLU(),
         Dropout2d(0.2),
         BatchNorm2d(256),
         ZeroPad2d([1,0,1,0]),
         MaxPool2d((2,2), stride=2)
      )

      self.layer6 = Sequential(
         Flatten(),
         Linear(6400, 128),
         Dropout2d(0.2),
         Linear(128, 1),
         Sigmoid()
      )

   def forward(self, x: Tensor) -> Tensor:
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.layer5(x)
      x = self.layer6(x)

      return x
