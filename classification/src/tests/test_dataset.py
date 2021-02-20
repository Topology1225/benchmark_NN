from attrdict import AttrDict
from dataset import get_dataset
from utils import read_yaml

def test_MnistDset():
    config = read_yaml("./config/clf.yaml")
    dset_config = read_yaml("./config/mnist.yaml")
    
    dataset = get_dataset(
        config, dset_config, mode="train"
    )
    dataset = get_dataset(
        config, dset_config, mode="valid"
    )
    dataset = get_dataset(
        config, dset_config, mode="valid"
    )