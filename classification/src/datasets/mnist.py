
import torch
import torchvision
from torchvision import datasets, transforms

class MNIST(object): 
    def __init__(self, root="../storage"): 
        self.root = root 

    def constructor(self, transformer=None): 
        if transformer is None:
            self.transform = transforms.Compose(
            [transforms.ToTensor()])
        
        else:
            self.transform = transformer  

        self.trainset = datasets.MNIST(
            root=self.root, 
            train=True,
            download=True,
            transform=self.transform
        )

        self.testset = datasets.MNIST(
            root=self.root, 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
    def batch_iter(self, test=False, batch_size=100, shuffle=False):
        dataset = self.testset if test else self.trainset 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        for images, labels in dataloader:
            yield images, labels


if __name__=="__main__":
    dataset = MNIST()
    dataset.constructor()  
    for images, labels in dataset.batch_iter(
                    test=False,
                    batch_size=2,
                    shuffle=True 
                    ):
        breakpoint()