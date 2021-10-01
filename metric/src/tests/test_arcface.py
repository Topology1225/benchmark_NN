from attrdict import AttrDict

from trainers import get_trainer
from dataset import get_dataset
from models import get_model
from utils import read_yaml


def test_build_arcface():
    config = read_yaml("./tests/config/arcface.yaml")
    dset_config = read_yaml("./tests/config/mnist.yaml")

    dset = get_dataset(config, dset_config, mode="train")
    valid_dset = get_dataset(config, dset_config, mode="valid")
    model = get_model(config, dset_config)

    trainer = get_trainer(config, dset_config)
    trainer.train(dataset=dset, valid_dataset=valid_dset, model=model)
