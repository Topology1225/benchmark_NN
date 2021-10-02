from models.resnet import ResNet


def get_model(config, dset_config):
    if "resnet" in config.model.name:
        return ResNet(config, dset_config)
    else:
        raise NotImplementedError
