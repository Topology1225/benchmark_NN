from trainers.clf import ClfTrainer


def get_trainer(config, dset_config):
    if config.loss.name=="clf":
        return ClfTrainer(config, dset_config)
    else:
        raise NotImplementedError


