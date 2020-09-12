import torch
import torch.nn as nn
import torchvision 
import torchvision.models as models

class ResNet18(nn.Module): 
    def __init__(self):
        super().__init__()
        self.name = "resnet18"
        self.epoch = 0
        self.build()

    def build(self):
        resnet18 = models.resnet18() 
        self.resnet18 = resnet18

    def forward(self, x):
        x = self.resnet18(x)
        return x
        

    def save(self):
        pass

    def load(self):
        pass


if __name__=="__main__":
    model = ResNet18() 
    x = torch.ones((1,3,256,256))
    x = model(x)
    breakpoint()



    