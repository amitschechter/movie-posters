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

from sklearn.metrics import precision_recall_fscore_support, average_precision_score

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

p_r_thresholds = [0.1382, 0.0357, 0.0421, 0.2912, 7.52e-05, 0.1815, 0.0304, 0.0233, 0.0959, 0.1143, 0.0971, 0.1628, 0.2196, 0.015177000000000001, 0.051547, 0.026259, 0.071176, 0.010339000000000001, 0.026667999999999997]

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
            running_probs_precision = []

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
                    probabilities, loss = multilabel_Loss(scores, labels)                     
#                     print('Loss: %s' %(loss))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        fileToWrite.write('Train Loss: %s\n' %(loss))
                        train_losses.append(loss)
                    elif phase =='val':
                        val_losses.append(loss)
                        
                # "OLD" way to calc precision using average_precision_score
                # calculating with the scores and probabilities
                average_precision = average_precision_score(labels, scores.data, average="micro")
                running_precision.append(average_precision) 

                average_probs_precision = average_precision_score(labels, probabilities.data, average="micro")
                running_probs_precision.append(average_probs_precision)             
                
                running_loss += loss.item() 
         
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_prec = np.mean(running_precision)
            epoch_probs_prec = np.mean(running_probs_precision)
                        
            precision_all_classes = []
            recall_all_classes = []           

            for i in range(num_classes):
                probabilities.data[:, i][probabilities.data[:, i] >= p_r_thresholds[i]] = 1
                probabilities.data[:, i][probabilities.data[:, i] < p_r_thresholds[i]] = 0
                p, r, f, s = precision_recall_fscore_support(labels[:, i], probabilities.data[:, i], average="binary")
                precision_all_classes.append(p)
                recall_all_classes.append(r)                    

            average_all_class_p = np.mean(precision_all_classes)
            average_all_class_r = np.mean(recall_all_classes)                

            if phase == 'val':
                epoch_val_precision_old.append(epoch_prec)
                epoch_val_probs_prec_old.append(epoch_probs_prec)
                epoch_val_precision_new.append(average_all_class_p)
                epoch_val_recall_new.append(average_all_class_r)
            else:
                epoch_train_precision_old.append(epoch_prec)
                epoch_train_probs_prec_old.append(epoch_probs_prec)            
                epoch_train_precision_new.append(average_all_class_p)
                epoch_train_recall_new.append(average_all_class_r)
                
#             fileToWrite.write('END OF EPOCH')
            fileToWrite.write('{} Loss: {:.4f} Prec: {:.4f}'.format(phase, epoch_loss, average_all_class_p, average_all_class_r))
            time_elapsed = time.time() - since
            fileToWrite.write('EPOCH complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))#, average_all_class_p, average_all_class_r))
            print("Average all class precision: %s" %(average_all_class_p))                
            print("Average all class recall: %s" %(average_all_class_r))  
            print("Average precision -- old way: %s" %(epoch_prec))                            
            print("Average probs precision -- old way: %s" %(epoch_probs_prec))                
            print('EPOCH complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#             # deep copy the model
            if phase == 'val' and average_all_class_p > best_prec:
                best_prec = epoch_prec
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    fileToWrite.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))            
    fileToWrite.write('Best val Prec: {:4f}'.format(best_prec))            

    print('LAST EPOCH LABELS AND SCORES')
    for i in range(num_classes):
        print(labels[:,i])
        print(probabilities[:,i])
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def multilabel_Loss(x, y):
    
    ## to calculate the log of the 
    shifted_logits = x - torch.max(x, dim=1, keepdim=True)[0]
    summed = torch.sum(torch.exp(shifted_logits), dim=1, keepdim=True)    
    probs = torch.exp(shifted_logits) / summed
    log_probs = torch.log(torch.exp(shifted_logits) / summed)
    N, _ = x.size()
    
    loss = -1 * torch.sum(y * log_probs) / N    
    
    return probs, loss

model_conv = torchvision.models.resnet18(pretrained=True)

# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)

model_conv = model_conv.to(device)

# learning_rates = [9e-4, 3e-3, 9e-3, 3e-2, 9e-2, 3e-1, 9e-1, 1]
learning_rates = [9e-6, 5e-4, 5e-3, 3e-2, 9e-2]

for learn_rt in learning_rates:
    train_losses = []
    val_losses = []

    epoch_train_precision_old = []
    epoch_train_probs_prec_old = []
    epoch_val_precision_old = []
    epoch_val_probs_prec_old = []

    epoch_train_precision_new = []
    epoch_train_recall_new = []
    epoch_val_precision_new = []
    epoch_val_recall_new = []
    
    print('NOW ON LEARNING RATE: %s' %(learn_rt))
#     lr_record_file= open("Results/multi_label_TL/test%s.txt" %(learn_rt),"w+")
    lr_record_file= open("Results/multi_label_TL/multi_resnet18_ADAM_LR_%s.txt"%(learn_rt),"w+")
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=learn_rt)
    model_conv = train_model(model_conv, optimizer_conv, lr_record_file, num_epochs=20)
    
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(15,5))
    axes = axes.ravel()
    axes[0].plot(train_losses, c='b', label='Train loss')
    axes[0].plot(val_losses, c='r', label='Val loss')    
    axes[0].set_title('Losses')
    axes[0].set_xlabel('Iteration')
    axes[0].legend()    
    
    axes[1].plot(epoch_train_precision_new, '-o', label="Train precision")
    axes[1].plot(epoch_val_precision_new, '-o', label="Val precision")
    axes[1].plot(epoch_train_recall_new, '-s', label="Train recall")
    axes[1].plot(epoch_val_recall_new, '-s', label="Val recall")    
    axes[1].set_title('Precision/Recall')
    axes[1].set_xlabel('Epoch')    
    axes[1].legend()
    
    axes[2].plot(epoch_train_precision_old, '-o', label="Train precision-old")
    axes[2].plot(epoch_val_precision_old, '-o', label="Val precision-old")  
    axes[2].plot(epoch_train_probs_prec_old, '-s', label="Train probs prec-old")
    axes[2].plot(epoch_val_probs_prec_old, '-s', label="Val probs prec-old")    
    axes[2].set_title('Old -- Precision')
    axes[2].set_xlabel('Epoch')    
    axes[2].legend()
    fig.savefig("Results/multi_label_TL/multi_resnet18_plots_ADAM_LR_%s.eps"%(learn_rt))    
    fig.savefig("Results/multi_label_TL/multi_resnet18_plots_ADAM_LR_%s.jpg"%(learn_rt))   
    
    with open("Results/multi_label_TL/PickleFile_%s.pickle"%(learn_rt), 'wb') as file:
        pickle.dump((train_losses, val_losses, epoch_train_precision_new, epoch_val_precision_new, 
                     epoch_train_recall_new, epoch_val_recall_new, model_conv), file,
                    protocol=pickle.HIGHEST_PROTOCOL)

