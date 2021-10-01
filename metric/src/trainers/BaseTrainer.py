import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import plotly.graph_objects as go

from utils import get_optimizer, select_device, increment_path
from utils import setup_logger

logger = setup_logger(__name__)


class BaseTrainer(object):
    config = None
    dset_config = None

    def __init__(self):
        pass

    def train(self, dataset, valid_dataset, model):
        model = self._set_cuda(model)
        self.optimizer = get_optimizer(config=self.config, model=model)

        self.model = model
        self._train(
            epochs=self.config.train.epoch,
            batch_size=self.config.train.batch_size,
            dataset=dataset,
            valid_dataset=valid_dataset,
        )

    def save(self):
        self._describe()

    def _plot_loss(self, x_dict):
        assert set(x_dict.keys()).issubset(set(["train", "valid"]))
        data = list()
        for phase in x_dict.keys():
            for l_name, epoch2loss in x_dict[phase].items():
                loss = list(epoch2loss.values())
                data.append(
                    go.Scatter(
                        x=list(range(len(epoch2loss.values()))),
                        y=list(epoch2loss.values()),
                        name=f"{phase}-{l_name}",
                    )
                )
        fig = go.Figure(data=data)
        fig.write_html(os.path.join(self.config.result_dir, "loss.html"))

    def _set_updater(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _set_cuda(self, model):
        has_device = hasattr(self.config, "device")
        if has_device:
            device = self.config.device

        else:
            device = ""
        device = select_device(device, self.config.train.batch_size)
        model = torch.nn.DataParallel(model).to(device)
        self.device = device
        self.cuda = self.device.type != "cpu"
        setattr(self.config, "device", device)
        setattr(self.config, "cuda", self.cuda)
        return model

    def _train(self, *args, **kwars):
        raise NotImplementedError
