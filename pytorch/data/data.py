import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

class ToTensor(object):
    def __call__(self, sample):
        entry = {}
        for k in sample:
            if k == 'rect':
                entry[k] = torch.IntTensor(sample[k])
            else:
                entry[k] = torch.FloatTensor(sample[k])
        return entry


class InpaintingDataset(Dataset):
    def __init__(self, info_list, root_dir='', im_size=(256, 256), transform=None):
        self.filenames = open(info_list, 'rt').read().splitlines()
        self.root_dir = root_dir
        self.transform = transform
        self.im_size = im_size
        np.random.seed(2018)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir, self.filenames[idx]))
        sample = {'gt': image}
        if self.transform:
            sample = self.transform(sample)
        return sample
