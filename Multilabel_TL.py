# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.transforms as T
import torchvision.datasets as dset

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, sampler, Dataset
from pytorch_load_data import load_data


from sklearn.metrics import precision_recall_curve, average_precision_score, recall_score

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import copy
import pickle

plt.switch_backend('agg')


batchSize = 64
num_classes = 19

dataloaders, poster_train, poster_val, poster_test = load_data(batchSize, -1)
dataset_sizes = {}
dataset_sizes['train'] = len(dataloaders['train'])
dataset_sizes['val'] = len(dataloaders['val'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, optimizer, fileToWrite, num_epochs=25):
    since = time.time()
    n_classes= 19

    best_model_wts = copy.deepcopy(model.state_dict())
    best_prec = 0.0

    for epoch in range(num_epochs):
        fileToWrite.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        fileToWrite.write('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs-1))    
	
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_precision = []
            running_recall = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                labels_tensor = torch.zeros((labels[0].size()[0], num_classes))
                for idx,each in enumerate(labels):
                    labels_tensor[:,idx] = each

                inputs = inputs.to(device)
                labels = labels_tensor.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward -- track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    scores = model(inputs)
                    loss = multilabel_Loss(scores, labels)                     
                    print('Loss: %s' %(loss))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        fileToWrite.write('Train Loss: %s\n' %(loss))
                        train_losses.append(loss)
                    elif phase =='val':
                        val_losses.append(loss)
                        

                # Statistics --multilabel precision
                average_precision = average_precision_score(labels, scores.data, average="micro")
                fileToWrite.write("Average precision: %s\n" %(average_precision))
                running_precision.append(average_precision) 
                print("Average precision: %s" %(average_precision))                
                
                recall_batch=[]              
                for i in range(num_classes):
                    average_recall = recall_score(labels[:, i], scores.data[:, i].round(), average="micro")
                    recall_batch.append(average_recall)

                average_recall = np.mean(recall_batch)    
                fileToWrite.write("Average recall: %s\n" %(average_recall))
                running_recall.append(average_recall)
                print("Average recall: %s" %(average_recall))                                
                
                running_loss += loss.item() 
         

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_prec = np.mean(running_precision)
            epoch_recall = np.mean(running_recall)
            
            
            if phase == 'val':
                epoch_val_precison.append(epoch_prec)
                epoch_val_recall.append(epoch_recall)
            else:
                epoch_train_precision.append(epoch_prec)
                epoch_train_recall.append(epoch_recall)                

            fileToWrite.write('END OF EPOCH')
            fileToWrite.write('{} Loss: {:.4f} Prec: {:.4f}'.format(phase, epoch_loss, epoch_prec))            
            time_elapsed = time.time() - since
            fileToWrite.write('EPOCH complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Prec: {:.4f}'.format(phase, epoch_loss, epoch_prec))
            print('EPOCH complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#             # deep copy the model
            if phase == 'val' and epoch_prec > best_prec:
                  best_prec = epoch_prec
                  best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    fileToWrite.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))            
    fileToWrite.write('Best val Prec: {:4f}'.format(best_prec))            

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def multilabel_Loss(x, y):
    
    ## to calculate the log of the 
    shifted_logits = x - torch.max(x, dim=1, keepdim=True)[0]
    summed = torch.sum(torch.exp(shifted_logits), dim=1, keepdim=True)    
    log_probs = torch.log(torch.exp(shifted_logits) / summed)
    N, _ = x.size()
    
    loss = -1 * torch.sum(y * log_probs) / N    
    
    return loss

model_conv = torchvision.models.resnet18(pretrained=True)

# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)

model_conv = model_conv.to(device)

# learning_rates = [9e-4, 3e-3, 9e-3, 3e-2, 9e-2, 3e-1, 9e-1, 1]
learning_rates = [5e-6, 6e-6, 7e-6, 8e-6]#, 9e-3, 3e-2, 9e-2, 3e-1, 9e-1, 1]

for learn_rt in learning_rates:
    train_losses = []
    val_losses = []
    epoch_train_precision = []
    epoch_val_precision = []
    epoch_train_recall = []
    epoch_val_recall = []
    
    print('NOW ON LEARNING RATE: %s' %(learn_rt))
    lr_record_file= open("Results/multi_label_TL/multi_resnet18_ADAM_LR_%s.txt"%(learn_rt),"w+")
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=learn_rt)
    model_conv = train_model(model_conv, optimizer_conv, lr_record_file, num_epochs=10)
    
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(15,5))
    axes = axes.ravel()
    axes[0].plot(train_losses)
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Iteration')
    axes[1].plot(epoch_train_precision, '-o', label="Train precision")
    axes[1].plot(epoch_val_precision, '-o', label="Val precision")
    axes[1].plot(epoch_train_recall, '-s', label="Train recall")
    axes[1].plot(epoch_val_recall, '-s', label="Val recall")    
    axes[1].set_title('Precision/Recall')
    axes[1].set_xlabel('Epoch')    
    fig.savefig("Results/multi_label_TL/multi_resnet18_plots_ADAM_LR_%s.eps"%(learn_rt))    
    fig.savefig("Results/multi_label_TL/multi_resnet18_plots_ADAM_LR_%s.jpg"%(learn_rt))   
    
    with open("Results/multi_label_TL/PickleFile_%s.pickle"%(learn_rt), 'wb') as file:
        pickle.dump((train_losses, val_losses, epoch_train_precision, epoch_val_precision, 
                     epoch_train_recall, epoch_val_recall, model_conv), file,
                    protocol=pickle.HIGHEST_PROTOCOL)

