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

"""Referebce
https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
"""

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each x sample
            out_features: size of each output sample
            s: norm of x feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, config):
        super().__init__() 
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.s = config.loss.params.s
        self.m = config.loss.params.m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features).type(torch.float16))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = config.loss.params.easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label): 
        # --------------------------- cos(theta) & phi(theta) --------------------------- 
        device = x.device
        cosine = F.linear(F.normalize(x), F.normalize(self.weight.to(device)))
        logger.debug(f"\n {self.weight}")
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = (cosine * self.cos_m - sine * self.sin_m).type(torch.float16)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class ArcfaceLoss(nn.Module) :
    def __init__(self, config, dset_config):
        super().__init__()
        self.config = config 
        self.arcface = ArcMarginProduct(
            config.loss.params.num_dim, 
            dset_config.classes,
            config
        )
        params = config.loss.params
        self.cel = nn.CrossEntropyLoss(
            weight=None,
            size_average=params.size_average,
            ignore_index=params.ignore_index,
            reduce=params.reduce,
            reduction=params.reduction
        )

    
    def forward(self, x, labels): 
        arc_x = self.arcface(x, labels) 
        # x = F.softmax(arc_x, dim=1)   
        loss = self.cel(x, labels)
        return loss
    
    def get_arg_x(self, x): 
        arc_x = self.arcface(x)
        x = F.softmax(arc_x, dim=1)
        return x, arc_x
        

class ArcfaceTrainer(BaseTrainer):
    l_train = defaultdict(dict)
    l_valid = defaultdict(dict)
    def __init__(self, config, dset_config):
        super().__init__()
        self.config = config
        self.dset_config = dset_config

        self.criterion = ArcfaceLoss(config, dset_config)

    def _constructor(self):
        pass

    def _train(self, epochs, batch_size, dataset, valid_dataset):
        # arcfaceのパラメタの追加
        self.optimizer.add_param_group(dict(params=self.criterion.parameters()))

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

                    logger.debug(f"\n {loss.detach().cpu().item()}")
                    t_sum_loss.append(loss.detach().cpu().item())

                ## string
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = \
                    ('%10s' + '%10.4s'*2)%(('%g/%g' % (e, epochs - 1)), mem, np.mean(t_sum_loss)) 
                pbar.set_description(s)

            self.l_train["loss"][e] = np.mean(t_sum_loss) 

            with torch.no_grad():
                self.model.eval()
                pbar = enumerate(valid_dloader)
                pbar = tqdm(pbar, total=math.ceil(len(valid_dataset)/batch_size))
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

            self.l_valid["loss"][e] = np.mean(v_sum_loss) 

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
            name = f"arcface-{epoch}"
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
