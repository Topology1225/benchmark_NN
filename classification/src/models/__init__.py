from models.torchhub import TorchHub

def get_model(config, dset_config):
    return TorchHub(config, dset_config)

