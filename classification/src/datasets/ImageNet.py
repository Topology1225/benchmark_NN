
import torch
import torchvision
from torchvision import datasets, transforms

class ImageNet(object): 
    def __init__(self, root="../storage"): 
        self.root = root 

    def constructor(self, transformer=None): 
        if transformer is None:
            self.transform = transforms.Compose(
            [transforms.ToTensor()])
        
        else:
            self.transform = transformer  

        self.trainset = datasets.ImageNet(
            root=self.root, 
            split="train",
            download=True,
            transform=self.transform
        )

        self.testset = datasets.ImageNet(
            root=self.root, 
            split="valid", 
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
    dataset = ImageNet()
    dataset.constructor()  
    for images, labels in dataset.batch_iter(
                    test=False,
                    batch_size=2,
                    shuffle=True 
                    ):
        breakpoint()
