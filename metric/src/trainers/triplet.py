import torch

from trainers.BaseTrainer import BaseTrainer
from utils import setup_logger

logger = setup_logger(__name__)


class TripletLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        params = config.loss.params
        self.triplet = torch.nn.TripletMarginLoss(
            margin=params.margin,
            p=params.p,
            eps=params.eps,
            swap=params.swap,
            size_average=params.size_average,
            reduce=params.reduce,
            reduction=params.reduction,
        )

    def forward(self, output):
        n_batch = int(output.shape[0] / 3)
        anchor, pos, neg = torch.tensor_split(output, n_batch, dim=0)

        return self.triplet(anchor, pos, neg)

    def __repr__(self) -> str:
        return super().__repr__()()


class TripletTrainer(BaseTrainer):
    l_train = dict()
    l_valid = dict()

    def __init__(self, config, dset_config):
        super().__init__()
        self.config = config
        self.dset_config = dset_config

        self.criterion = TripletLoss(config)

    def _constructor(self):
        pass

    def _train(self, epoch, batch_size, dataset, valid_dataset, model):
        train_dloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
        )
        valid_dloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
        )

        for e in range(epoch):
            for x, targets in train_dloader:
                x = torch.cat(x, dim=1).to(self.device)
                breakpoint()
                output = model(x)
                loss = self.criterion(output)
                self._set_updater(loss)
                self.l_train[epoch] = dict(loss=loss)

            with torch.no_grad():
                for x, targets in valid_dloader:
                    x = torch.cat(x, dim=1).to(self.device)
                    output = model(x)
                    loss = self.criterion(output)
                    self.l_valid[epoch] = dict(loss=loss)

    def __repr__(self) -> str:
        return super().__repr__()
