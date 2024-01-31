import torch
import torchvision.transforms as transform
import PIL.Image
import os
import json
import numpy as np
import cv2
import copy

class Dataset(torch.utils.data.Dataset):
    'Build a dataset for PyTorch'
    def __init__(self, datapath, labelpath, transform=None):
        'Initialization'
        self.datapath = datapath
        self.list_fnames = os.listdir(datapath) 
        self.fname2label = json.load(open(labelpath))
        self.preTransform = transform

    def __len__(self):
        'Count the total number of samples'
        return len(self.list_fnames)


    def loadImage(self,path):
        img = PIL.Image.open(path)
        img.load()
        img_array = np.array(img)
        img_array2 = copy.deepcopy(img_array)
        img_array3 = copy.deepcopy(img_array)

        img3d = cv2.merge((img_array, img_array2, img_array3))
        img_tensor = transform.ToTensor()(img3d)
        return img_tensor


    def __getitem__(self, index):
        'Retrieve one sample of data'
        fname = self.list_fnames[index] 
        X = self.loadImage(self.datapath + '/' + fname)
        if fname not in self.fname2label:
            label = 0
        else:
            label = self.fname2label[fname]
        label = int(label)

        return X, label, fname 
