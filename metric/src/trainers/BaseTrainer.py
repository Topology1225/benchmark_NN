import torch 
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_optimizer , select_device


class BaseTrainer(object):
    config = None
    dset_config = None

    def __init__(self):
        pass

    def train(self, dataset, valid_dataset, model):
        model = self._set_cuda(model)
        self.optimizer = get_optimizer(
            config=self.config,
            model=model
        )

        self._train(
            epochs=self.config.train.epoch,
            batch_size=self.config.train.batch_size,
            dataset=dataset,
            valid_dataset=valid_dataset,
            model=model
        )
    
    def save(self):
        self.description()

    def _set_updater(self, loss): 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

    def _set_cuda(self, model):
        has_device = hasattr(self.config, "device") 
        if has_device:
            # device_ids = min(self.config.device_ids) 
            # self.device = torch.device(f"cuda:{device_ids}") 
            
            # # dataparallel
            # model = torch.nn.DataParallel(model, device_ids=device_ids) 
            # return model
            device = self.config.device

        else:
            device = ""
        device = select_device(device, self.config.train.batch_size)
        model = torch.nn.DataParallel(model).to(device)
        self.device = device 
        self.cuda = self.device.type != 'cpu' 
        setattr(self.config, "device", device) 
        setattr(self.config, "cuda", self.cuda)
        return model



    def _train(self, *args, **kwars):
        raise NotImplementedError