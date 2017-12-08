
# coding: utf-8

# In[10]:


from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import torch.nn as nn

class FashionMNIST(data.Dataset):
    def __init__(self,root, transform = None, target_transform= None,train=True):
        self.root = root
        self.transform  = transform
        self.target_transform = target_transform
        self.train = train
        
        if train:
            self.train_labels = self.read_label_file(root+"/train-labels-idx1-ubyte")
            self.train_data = self.read_image_file(root+"/train-images-idx3-ubyte")
        else:
            self.test_labels = self.read_label_file(root+"/t10k-labels-idx1-ubyte")
            self.test_data = self.read_image_file(root+"/t10k-images-idx3-ubyte")
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def get_int(self,b):
        return int(codecs.encode(b, 'hex'), 16)


    def parse_byte(self,b):
        if isinstance(b, str):
            return ord(b)
        return b


    def read_label_file(self,path):
        with open(path, 'rb') as f:
                data = f.read()
                    
        length = self.get_int(data[4:8])
        labels = [self.parse_byte(b) for b in data[8:]]
    
        return torch.LongTensor(labels)


    def read_image_file(self,path):
        with open(path, 'rb') as f:
            data = f.read()
            
            length = self.get_int(data[4:8])
            num_rows = self.get_int(data[8:12])
            num_cols = self.get_int(data[12:16])
            images = []
            idx = 16
            for l in range(length):
                img = []
                images.append(img)
                for r in range(num_rows):
                    row = []
                    img.append(row)
                    for c in range(num_cols):
                        row.append(self.parse_byte(data[idx]))
                        idx += 1
            return torch.ByteTensor(images).view(-1, 28, 28)

