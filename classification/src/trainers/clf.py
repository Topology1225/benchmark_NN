import os
from collections import defaultdict
import math
from threading import Thread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn import Parameter
from tqdm import tqdm
import numpy as np

from trainers.BaseTrainer import BaseTrainer
from utils import setup_logger, save2json

logger = setup_logger(__name__)


class SoftMaxLoss(nn.Module): 
    def __init__(self, config, dset_config) -> None:
        super().__init__() 
        params = config.loss.params
        self.cel = nn.CrossEntropyLoss(
            weight=None,
            size_average=params.size_average,
            ignore_index=params.ignore_index,
            reduce=params.reduce,
            reduction=params.reduction
        )
    def forward(self, output, labels):
        loss = self.cel(output, labels) 
        return loss


class ClfTrainer(BaseTrainer):
    l_train = defaultdict(dict)
    l_valid = defaultdict(dict)

    def __init__(self, config, dset_config):
        super().__init__()
        self.config = config
        self.dset_config = dset_config 

        params = config.loss.params

        self.criterion = SoftMaxLoss(config, dset_config)

    def _constructor(self):
        pass

    def _train(self, epochs, batch_size, dataset, valid_dataset):

        train_dloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle = True, 
            num_workers = self.config.train.num_workers
        )
        valid_dloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle = True, 
            num_workers = self.config.train.num_workers
        )
        logger.info(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'loss')) 
        
        for e in range(epochs): 
            self.model.train()
            pbar = enumerate(train_dloader)
            pbar = tqdm(pbar, total=math.ceil(len(dataset)/batch_size))
            t_sum_loss = list()
            v_sum_loss = list()
            for i, (x, labels) in pbar:   
                with amp.autocast(enabled=self.cuda):  
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    output = self.model(x)
                    loss = self.criterion(output, labels) 
                    self._set_updater(loss)
                    t_sum_loss.append(loss.detach().cpu().item())

                ## string
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = \
                    ('%10s' + '%10.4s'*2)%(('%g/%g' % (e, epochs - 1)), mem, np.mean(t_sum_loss)) 
                pbar.set_description(s)

            self.l_train[e] = dict(
                        loss=np.mean(t_sum_loss) 
                )

            with torch.no_grad():
                self.model.eval()
                pbar = enumerate(valid_dloader)
                pbar = tqdm(pbar, total=int(len(valid_dataset)/batch_size))
                for i, (x, labels) in pbar:
                    with amp.autocast(enabled=self.cuda):  
                        x = x.to(self.device)
                        labels = labels.to(self.device)
                        output = self.model(x)
                        loss = self.criterion(output, labels) 
                        v_sum_loss.append(loss.detach().cpu().item()) 
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = \
                        ('%10s' + '%10.4s'*2)%(('%g/%g' % (e, epochs - 1)), mem, np.mean(v_sum_loss)) 
                    pbar.set_description(s)

            self.l_valid[e] = dict(
                loss=np.mean(v_sum_loss)
            )   
    
    def _describe(self):
        x_dict = dict(
            train=self.l_train,
            valid=self.l_valid
        )
        self._plot_loss(x_dict)

        # to json 
        json_file_name = os.path.join(self.config.result_dir, "result.json")
        save2json(x_dict, json_file_name)

        # save model
        self._save_model(epoch=self.config.train.epoch, name="last")

    def _save_model(self, epoch, name=None):
        if name is None:
            name = f"clf-{epoch}"
        ckpt = dict(
            epoch=epoch,
            model=self.model.cpu().state_dict(),
            criterion=self.criterion.cpu().state_dict(),
            optimizer=self.optimizer.state_dict()
        )
        pth_file = os.path.join(self.config.result_dir, "weights", f"{name}.pth") 
        torch.save(ckpt, pth_file)

    def __repr__(self) -> str:
        return super().__repr__()
