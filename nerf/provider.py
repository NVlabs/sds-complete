from scipy.spatial.transform import Slerp, Rotation

import torch.nn.functional as F
from torch.utils.data import DataLoader



class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W

        self.size = size

        self.training = self.type in ['train', 'all']
        


    def collate(self, index):

        B = len(index) # always 1



        data = {
            'H': self.H,
            'W': self.W,
        }

        return data


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        return loader