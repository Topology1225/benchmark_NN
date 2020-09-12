
import torch
import torch.nn as nn
class CNN28(nn.Module):
    def __init__(self):
        super().__init__()
        self.name  = "cnn28" 
        self.epoch = 0 
        self.build()


    def build(self):
        layers = list()
        for i in range(2):
            layer = nn.Sequential(
                nn.Conv2d(1, 2, 3, 1, 0),
                nn.ReLU()
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        pass

    def save(self):
        pass


    def load(self):
        pass


if __name__=="__main__":
    model = CNN28()
    
