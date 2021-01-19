import torch
import torchvision
from sklearn.model_selection import train_test_split

from utils import setup_logger

logger = setup_logger(__name__)

"""
train時は真にランダムにanchorに対してposとnegをreturn
valid, test時にはseed固定して同じanchor, pos, negの組み合わせが出力されるようにする.
"""

class TripletDset(torchvision.datasets.MNIST):
    test_size=0.2
    random_state=1
    def __init__(self, root, train, transform, target_transform, download, config):

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.config = config
        if download:
            self.download()

        self._construct()

    def _construct(self):
        tensors, labels = self._get_data()
        # self.tri_tensors, self.tri_labels = \
        #     self._create_triple(tensors, labels)
        logger.warning("triplet datasetは未実装")

        
    def _get_data(self):
        pt_train = \
            f"{self.root}/MNIST/processed/training.pt" 
        pt_test = \
            f"{self.root}/MNIST/processed/test.pt"
        lt_tensors, lt_labels = list(), list()
        for pt in [pt_train, pt_test]:
            tensor, label = torch.load(pt)
            lt_tensors.append(tensor)
            lt_labels.append(label)

        return torch.cat(lt_tensors), torch.cat(lt_labels)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        anchor, pos, neg, pos_labels, neg_labels \
            = self.tri_tensors[idx], self.tri_labels[idx]
        return anchor, pos, neg, pos_labels, neg_labels
    
    def _create_triple(self, tensors, labels):
        from collections import defaultdict
        class2where=dict()
        class2imgs= defaultdict(dict)
        for idx in range(10):
            bool_idx = torch.where(labels==idx)
            class2where[idx]=bool_idx
            img, label = tensors[bool_idx], labels[bool_idx]
            trainval_img, test_img, trainval_label, test_label =\
                train_test_split(img, label, 
                    test_size=self.test_size,
                    random_state=self.random_state
                )

            train_img, valid_img, train_label, valid_label =\
                train_test_split(trainval_img, trainval_label, 
                    test_size=self.test_size,
                    random_state=self.random_state
                )
            class2imgs[idx]["train"] = train_img
            class2imgs[idx]["valid"] = valid_img
            class2imgs[idx]["test"]  = test_img

        self.train = [((anc, pos, neg), label1), ((anc, pos, neg), label2)]
        breakpoint()
        



    