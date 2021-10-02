from attrdict import AttrDict
from dataset import get_dataset
from utils import read_yaml


def test_TripletDset():
    config = read_yaml("./tests/config/triplet.yaml")
    dset_config = read_yaml("./tests/config/mnist.yaml")
    dataset = get_dataset(config, dset_config, mode="train")
    dataset = get_dataset(config, dset_config, mode="valid")
    dataset = get_dataset(config, dset_config, mode="valid")


def test_ClfDset():
    config = read_yaml("./tests/config/arcface.yaml")
    dset_config = read_yaml("./tests/config/mnist.yaml")

    dataset = get_dataset(config, dset_config, mode="train")
    dataset = get_dataset(config, dset_config, mode="valid")
    dataset = get_dataset(config, dset_config, mode="valid")
