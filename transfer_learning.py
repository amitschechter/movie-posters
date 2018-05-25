# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import models, transforms
import torchvision.datasets as dset
import torchvision.transforms as T

from torch.utils.data import DataLoader 
from torch.utils.data import sampler
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import os
import copy
import glob
import os.path as osp
from PIL import Image
import pickle

from pytorch_load_data import load_data
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

batchsize=64
dataloaders, poster_train, poster_val, poster_test = load_data(batchsize=batchsize)
dataset_sizes = {}
dataset_sizes['train'] = len(dataloaders['train'])
dataset_sizes['val'] = len(dataloaders['val'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_positives_count = 1601.0
train_negatives_count = 1744.0
val_positives_count = 461.0
val_negatives_count = 493.0

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_epoch_accuracies, val_epoch_accuracies = [], []
        train_epoch_precisions, val_epoch_precisions = [], []
        train_epoch_recalls, val_epoch_recalls = [], []
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
#                 scheduler.step()    # Update learning rate (decay)
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_positives = 0
            running_pred_positives = 0

            # Iterate over batches.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_losses.append(loss)
                    elif phase == 'val':
                        val_losses.append(loss)

                # statistics
                running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
                batch_precision, batch_recall, batch_fscore, support = score(labels.data.cpu().numpy(), preds.data.cpu().numpy())
                batch_accuracy = accuracy_score(labels.data.cpu().numpy(), preds.data.cpu().numpy())
                if phase == 'train':
                    train_epoch_accuracies.append(batch_accuracy)
                    train_epoch_precisions.append(batch_precision)
                    train_epoch_recalls.append(batch_recall)
                elif phase == 'val':
                    val_epoch_accuracies.append(batch_accuracy)
                    val_epoch_precisions.append(batch_precision)
                    val_epoch_recalls.append(batch_recall)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            
    
            if phase == 'train':
                epoch_acc = np.mean(train_epoch_accuracies)
                epoch_precision, epoch_recall = np.mean(train_epoch_precisions), np.mean(train_epoch_recalls)
                train_accuracies.append(epoch_acc)
                train_precisions.append(epoch_precision)
                train_recalls.append(epoch_recall)
            elif phase == 'val':
                epoch_acc = np.mean(val_epoch_accuracies)
                epoch_precision, epoch_recall = np.mean(val_epoch_precisions), np.mean(val_epoch_recalls)
                val_accuracies.append(epoch_acc)
                val_precisions.append(epoch_precision)
                val_recalls.append(epoch_recall)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Precision: {:.4f}, Recall: {:.4f}'.format(
                epoch_precision, epoch_recall))
            time_elapsed = time.time() - since
            print('Time passes: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
model_conv = torchvision.models.resnet18(pretrained=True)

# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
# optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

learning_rates = [5e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
for learning_rate in learning_rates:
    print("learning_rate: {}".format(learning_rate))
    model_conv = torchvision.models.resnet18(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    exp_lr_scheduler = None
    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=learning_rate)
    model_conv, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=1)
    
    print("learning_rate: {}".format(learning_rate))
    
    plt.figure()
    plt.subplot(311)
    plt.title('Training loss')
    plt.plot(train_losses, 'o')
    plt.xlabel('Iteration')

    plt.subplot(312)
    plt.title('Validation loss')
    plt.plot(val_losses, 'o')
    plt.xlabel('Iteration')
    
    plt.subplot(313)
    plt.title('Accuracy')
    plt.plot(train_accuracies, '-o', label='Train Accuracy')
    plt.plot(val_accuracies, '-o', label='Val Accuracy')
    plt.xlabel('Iteration')
    plt.legend(loc='lower center')
    
    plt.savefig('Transfer learning'+str(learning_rate)+'.pdf')
    
    with open(str(learning_rate)+'.pickle', 'wb') as file:
        pickle.dump((train_losses, val_losses, train_accuracies, val_accuracies), file, protocol=pickle.HIGHEST_PROTOCOL)

