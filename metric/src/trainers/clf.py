import math
from threading import Thread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.nn import Parameter
from tqdm import tqdm

from trainers.BaseTrainer import BaseTrainer
from utils import setup_logger

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
        pred = F.softmax(output, dim=1)
        loss = self.cel(pred, labels) 
        return loss


class ClfTrainer(BaseTrainer):
    l_train = dict()
    l_valid = dict()
    
    def __init__(self, config, dset_config):
        super().__init__()
        self.config = config
        self.dset_config = dset_config 

        params = config.loss.params

        self.criterion = SoftMaxLoss(config, dset_config)

    def _constructor(self):
        pass

    def _train(self, epochs, batch_size, dataset, valid_dataset, model):

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
            model.train()
            self.criterion.train()

            pbar = enumerate(train_dloader)
            pbar = tqdm(pbar, total=int(len(dataset)/batch_size))
            t_sum_loss = 0
            v_sum_loss = 0
            for i, (x, labels) in pbar:   
                with amp.autocast(enabled=self.cuda):  
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    output = model(x)
                    loss = self.criterion(output, labels) 
                    self._set_updater(loss)
                    t_sum_loss += loss.detach().cpu().item()

                ## string
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = \
                    ('%10s' + '%10.4s'*2)%(('%g/%g' % (e, epochs - 1)), mem, t_sum_loss/(i+1)) 
                pbar.set_description(s)

            self.l_train[e] = dict(
                        loss=t_sum_loss/(i+1)
                )

            with torch.no_grad():
                model.eval()
                self.criterion.eval()
                pbar = enumerate(valid_dloader)
                pbar = tqdm(pbar, total=int(len(valid_dataset)/batch_size))
                for i, (x, labels) in pbar:
                    with amp.autocast(enabled=self.cuda):  
                        x = x.to(self.device)
                        labels = labels.to(self.device)
                        output = model(x)
                        loss = self.criterion(output, labels) 
                        v_sum_loss += loss.detach().cpu().item()
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = \
                        ('%10s' + '%10.4s'*2)%(('%g/%g' % (e, epochs - 1)), mem, v_sum_loss/(i+1)) 
                    pbar.set_description(s)

            self.l_valid[e] = dict(
                loss=v_sum_loss/(i+1)
            )   
    
    def _description(self): 
        # learning curveの描写plotly
        # train logのjsonへの書き出し
        # modelのsave
        pass

    def __repr__(self) -> str:
        return super().__repr__()
