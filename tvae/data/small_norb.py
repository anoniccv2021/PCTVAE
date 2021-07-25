import os
import numpy as np
from sklearn.utils.extmath import cartesian
from torch.utils.data import DataLoader, Dataset
from urllib import request
import torch
from tvae.data.smallnorb.dataset import SmallNORBDataset
from scipy import ndimage

class SequenceDataset(Dataset):
    def __init__(self, seq_len=None, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.index_manager = None   # set in child class
        self.factor_sizes = None  # set in child class

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        obj_idx = idx % self.num_objects
        factor_idx = idx % self.num_factors 
        other_factor_idxs = self.factor_idxs.difference([factor_idx])

        start_idx = 0
        for ofi in other_factor_idxs:
            start_idx += self.strides[ofi] * torch.randint(0, self.factor_sizes[ofi], (1,)).item()

        x0 = self.data_by_objects[obj_idx][start_idx]

        img_seq = [ndimage.zoom(x0.image_lt / 255.0, 0.5)]
        feat_seq = [x0.pose]

        for t in range(1, self.seq_len):
            factor_t = t % self.factor_sizes[factor_idx]
            offset_idx = factor_t * self.strides[factor_idx]
            xt = self.data_by_objects[obj_idx][start_idx + offset_idx]
            img_seq.append(ndimage.zoom(xt.image_lt.copy() / 255.0, 0.5))
            feat_seq.append(xt.pose.copy())

        img_seq = torch.tensor(img_seq).unsqueeze(1)
        feat_seq = torch.tensor(feat_seq)

        return img_seq, feat_seq


class NORBDataset(SequenceDataset):
    """
    A PyTorch wrapper for the Small NORB dataset
    It contains images of 50 toys belonging to 5 generic categories: four-legged animals, human figures, airplanes, trucks, and cars.
    The objects were imaged by two cameras under 6 lighting conditions, 9 elevations (30 to 70 degrees every 5 degrees),
    and 18 azimuths (0 to 340 every 20 degrees). 
    
    * Each "-info" file stores 24,300 4-dimensional vectors, which contain additional information about the corresponding images:
        - 1. the instance in the category (0 to 9)
        - 2. the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
        - 3. the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
        - 4. the lighting condition (0 to 5)
    """
    def __init__(self, dir='', 
                 split='train',
                 seq_len=18,
                 max_n_objs=25,
                 **kwargs):
        super().__init__(seq_len, **kwargs)

        print("Loading NORB {} Dataset".format(split))
        self.path = dir
        self.dataset = SmallNORBDataset(self.path)
        self.split = split
        self.data_by_objects = self.dataset.group_dataset_by_category_and_instance(self.split)[:max_n_objs]

        lighting_relabel = [2, 3, 5, 1, 4, 0] # correct sequence [5, 3, 0, 1, 4, 2]

        print("Sorting Poses & Relabeling Lighting")
        for i in range(len(self.data_by_objects)):
            for e in range(len(self.data_by_objects[i])):
                self.data_by_objects[i][e].lighting = lighting_relabel[self.data_by_objects[i][e].lighting]
            self.data_by_objects[i] = sorted(self.data_by_objects[i], key=lambda x: (x.pose[0], x.pose[1], x.pose[2]))


        self.factor_sizes = [9, 18, 6]
        self.factor_idxs = set([0,1,2])
        self.strides = [self.factor_sizes[1] * self.factor_sizes[2],
                        self.factor_sizes[2],
                        1]

        self.num_factors = len(self.factor_sizes)     
        self.num_objects = len(self.data_by_objects)
   
        print("Done")

    def __len__(self):
        return len(self.dataset.data[self.split])


def get_dataloader(dir='', 
                   max_n_objs=25,
                   batch_size=20):
    train_data = NORBDataset(dir=dir, max_n_objs=max_n_objs, split='train')
    test_data = NORBDataset(dir=dir, max_n_objs=max_n_objs, split='test')
    
    train_loader = DataLoader(
		train_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=20,
		pin_memory=True,
		drop_last=True)
    
    test_loader = DataLoader(
		test_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=20,
		pin_memory=True,
		drop_last=True)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloader()

    for x, f in train_loader:
        print(x,f)
        break