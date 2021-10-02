import torch
import torchvision
from torchvision import transforms

from dataset.mnist import TripletDset
from utils import setup_logger
from utils.dset_functions import EncodeOnehot

logger = setup_logger(__name__)


def get_dataset(config, dset_config, mode):
    dset_name = dset_config.name
    logger.debug(f"\n [DATASET]: {dset_name}")
    if dset_name == "mnist":
        return get_mnist(config, dset_config, mode)
    else:
        raise NotImplementedError


def get_mnist(config, dset_config, mode):
    logger.debug(f"\n [Dataset Type]: {config.train.dset_type}")
    if config.train.dset_type == "triplet":
        return TripletDset(
            root=dset_config.root,
            train=mode,
            transform=get_transforms(dset_config),
            target_transform=get_target_transforms(dset_config),
            download=dset_config.download,
            config=config,
        )
    elif config.train.dset_type == "clf":
        if mode == "train":
            train = True
        else:
            train = False
        return torchvision.datasets.MNIST(
            root=dset_config.root,
            train=train,
            transform=get_transforms(dset_config),
            target_transform=get_target_transforms(dset_config),
            download=dset_config.download,
        )
    else:
        raise NotImplementedError


def get_transforms(dset_config):
    _transforms = list()
    params = dset_config.transforms
    if hasattr(params, "resize"):
        resize = params.resize
        assert type(resize) is int or type(resize) is list
        _transforms.append(torchvision.transforms.Resize(size=resize))

    if hasattr(params, "RGB"):
        rgb_flag = params.RGB
        assert type(rgb_flag) is bool
        if rgb_flag:
            _transforms.append(transforms.Grayscale(num_output_channels=3))

    _transforms.append(transforms.ToTensor())
    logger.info(f"\n [Transforms] : {_transforms}")
    return transforms.Compose(_transforms)


def get_target_transforms(dset_config):
    if dset_config.target_transform is None:
        return None
    elif dset_config.target_transform == "onehot":
        func = EncodeOnehot(dset_config)
        return func
    else:
        assert NotImplementedError
