import torch

class EncodeOnehot(object):
    def __init__(self,  dset_config) -> None:
        super().__init__() 
        self.nc = dset_config.classes

    def __call__(self, x):
        assert x is not None
        return torch.eye(self.nc)[x].type(torch.long)
