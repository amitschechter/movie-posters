from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.transforms as T
import torchvision.datasets as dset

from torch.utils.data import DataLoader, sampler, Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class MOVIES(Dataset):

    def __init__(self, images_folder, labels_doc, transform=None, preload=False, train=True, genreIdx=7):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = images_folder
        self.transform = transform
        self.train = train
        self.genreIdx = genreIdx
            
        all_filenames = pd.read_csv(images_folder + '/' + labels_doc)        
        
        filenames = all_filenames['filename']
        
        # labels are read in this order:
        # 'Action'[0], 'Thriller'[1], 'Adventure'[2], 'Animation'[3], 'Western'[4], 'Comedy'[5], 
        # 'Crime'[6], 'Drama'[7], 'Horror'[8], 'Romance'[9], 'Science Fiction'[10], 'Fantasy'[11], 
        # 'Family'[12], 'Documentary'[13], 'History'[14], 'Music'[15], 'Mystery'[16], 'TV Movie'[17], 'War'[18]
        labels = all_filenames[all_filenames.columns[2:]]
        
        for idx, fn in enumerate(filenames):
            self.filenames.append((fn, list(labels.iloc[idx]))) # (filename, labels)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        
        image_fn, label = self.filenames[index]
        if self.genreIdx != -1:
            label = label[self.genreIdx] # [7] for drama label 

        image = Image.open('Data/posters_images/'+image_fn)

        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
            
        # return image and label
        return image, label
    
    def __len__(self):
        # Total number of samples in the dataset
        return self.len
    
def load_data(batchsize=1, genreIdx=7):
    image_folder = 'Data/poster_label_files'
    Train_images_labels = 'Dataset_Training_3345.csv'
    Val_images_labels = 'Dataset_Validation_955.csv'
    Test_images_labels = 'Dataset_Test_479.csv'

    BATCHSIZE = batchsize #cifar10 set to 64
    
    data_transforms = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize((0.46, 0.41, 0.38),(0.06, 0.28, 0.27))])    

    poster_train = MOVIES(image_folder, Train_images_labels, transform=data_transforms, train=True, genreIdx=genreIdx)
    poster_val = MOVIES(image_folder, Val_images_labels, transform=data_transforms, train=True, genreIdx=genreIdx)
    poster_test = MOVIES(image_folder, Test_images_labels, transform=data_transforms, train=True, genreIdx=genreIdx)

    dataloaders = {}
    dataloaders['train'] = DataLoader(poster_train, batch_size=BATCHSIZE)
    dataloaders['val'] = DataLoader(poster_val, batch_size=BATCHSIZE)
    dataloaders['test'] = DataLoader(poster_test, batch_size=BATCHSIZE)
    
    return dataloaders, poster_train, poster_val, poster_test
   
