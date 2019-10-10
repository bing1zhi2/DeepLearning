# -*- coding:utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import collections
from skimage import io, transform
from PIL import Image
import numpy as np

data_dir = "D:\\dataset\\mtwi_2018_data\\mydata_train\\temp"
root_dir = "D:\\dataset\\mtwi_2018_data\\mydata_train\\temp\\images"
label_file = "D:\\dataset\\mtwi_2018_data\\mydata_train\\temp\\labels.txt"



class MTWIDataSet(Dataset):
    def __init__(self, label_file, root_dir, transform=None):
        sample = None
        self.img_names, self.img_label = self.load(label_file)
        self.transform = transform
        all_char = "73774-脸部身体均适用——6号SATFACTIONTEED吉林科学技术出版社45mm"

        counter_dict = collections.Counter()

        for s in all_char:
            counter_dict.update(s)

        idx_list = list(counter_dict.keys())
        keys_dict = {}
        for i, s in enumerate(idx_list):
            keys_dict[s] = i
        self.keys = keys_dict

    def load(self, label_file):
        img_names = []
        img_label = []
        with open(label_file, "r", encoding="utf-8") as f:
            for i in f:
                name, label = i.split(" ")
                print(name, label)
                img_names.append(name)
                img_label.append(label.strip())
        return img_names, img_label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        print("__getitem__")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.img_names[idx]
        label_str = self.img_label[idx]
        img_path = os.path.join(root_dir, name)
        image = io.imread(img_path)
        # image = Image.open(img_path)

        label_list = [self.keys[s] for s in label_str]

        sample = {'image': image, 'label': np.array(label_list)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class WidthSample(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


dataset = MTWIDataSet(label_file, root_dir)

a = [1, 5, 78, 9, 68]
# b = torch.utils.data.SequentialSampler(a)
# b = torch.utils.data.SequentialSampler(dataset)
b = WidthSample(dataset)
for x in b:
    print(x)


