import torch
import torch.nn as nn
from torchvision import models
from utils import initilize

from utils.logging import setup_logger

logger = setup_logger(__name__)


class ResNet(torch.nn.Module):
    def __init__(self, config, dset_config):
        super().__init__()
        self.config = config
        self.dset_config = dset_config
        self._constructor()

    def _constructor(self):
        assert not self.config.loss.name is None
        nc = self.config.loss.params.num_dim

        model = torch.hub.load(
            self.config.model.version,
            self.config.model.name,
            pretrained=self.config.model.pretrained,
        )
        fc = torch.nn.Linear(in_features=1000, out_features=nc)

        initilize(self.config, [fc])
        self.layers = torch.nn.ModuleList([model, torch.nn.ReLU(inplace=True), fc])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        return super().__repr__()
