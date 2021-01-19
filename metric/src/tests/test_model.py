import torch

from attrdict import AttrDict
from models import get_model
from utils import read_yaml

def test_resnet():
    config = read_yaml("./tests/config/triplet.yaml")
    dset_config = read_yaml("./tests/config/mnist.yaml")
    model = get_model(config, dset_config)
    x = torch.zeros((1,3,512,512)).type(torch.float)
    x = model(x)
    assert x.shape == (1, config.loss.params.num_dim)
    x = torch.zeros((1,3,256,256)).type(torch.float)
    x = model(x)
    assert x.shape == (1, config.loss.params.num_dim)
    
