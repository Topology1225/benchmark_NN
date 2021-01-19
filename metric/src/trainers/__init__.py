from trainers.triplet import TripletTrainer
from trainers.arcface import ArcfaceTrainer
from trainers.clf import ClfTrainer


def get_trainer(config, dset_config):
    if config.loss.name=="triplet":
        return TripletTrainer(config, dset_config)
    elif config.loss.name=="arcface":
        return ArcfaceTrainer(config, dset_config) 
    elif config.loss.name=="clf":
        return ClfTrainer(config, dset_config)
    else:
        raise NotImplementedError


