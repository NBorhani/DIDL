import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest,f_regression
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

def train_one_epoch(model, device, trainloader, optimizer, epoch):
    
    model.train()
    loss_func = nn.BCELoss() 
    predictions_tr = torch.Tensor()
    scheduler = MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.5)
    labels_tr = torch.Tensor()
    for count,(mir,prot, label) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(mir,prot)
        predictions_tr = torch.cat((predictions_tr, output.cpu()), 0)
        new_label = label.clone().detach()   #torch.tensor(label).view(-1,1).cpu()
        labels_tr = torch.cat((labels_tr, new_label), 0)
        loss = loss_func(output, new_label)
        loss.backward()
        optimizer.step()
    scheduler.step()
    #labels_tr = labels_tr.detach().numpy()
    #predictions_tr = predictions_tr.detach().numpy()
    ##auc_tr = auroc(labels_tr, predictions_tr)
    


def predict_one_epoch(model, device, loader):
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    with torch.no_grad():
      for count,(mir,prot, label) in enumerate(loader):
          output = model(mir,prot)
          #print("labelll-----", label)
          predictions = torch.cat((predictions, output.cpu()), 0)
          new_label =  label.clone().detach()# torch.tensor(label).view(-1,1).cpu()
          labels = torch.cat((labels,new_label), 0)
    labels = labels.numpy()
    predictions = predictions.numpy()
    return labels.flatten(), predictions.flatten()



import torch
import torch.nn.functional as F

def compute_bce_loss(y_true, y_pred):
    """
    Compute the Binary Cross-Entropy (BCE) loss between actual and predicted values.
    """
    # Ensure inputs are tensors on the same device
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Move to same device (useful if using GPU)
    y_true = y_true.to(y_pred.device)

    # Apply sigmoid and compute BCE loss
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

    return loss.item()