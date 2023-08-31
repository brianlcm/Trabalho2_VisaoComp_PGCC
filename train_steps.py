#%% LIBRARIES

import torch
import torch.nn.functional as F
import warnings

from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast
from tqdm import tqdm

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
#%% TRAINING STEPS LOOPS

def train_steps(epoch, net, train_loader, criterion, optimizer, scaler):
    num_samples, loss_train = 0, 0

    tgs_i = [] # Inputs targets
    preds_i = [] # Predictions

    net = net.train()
    for train in tqdm(train_loader, desc = 'Steps Epoch {}'.format(epoch+1)):
        inputs, labels = train
        inputs, labels = inputs.to(device), labels.to(device) # Put on GPU

        with autocast():
            outputs = net(inputs) # Forward
            loss = criterion(outputs, labels) # Loss function

        scaler.scale(loss).backward() # Backward

        # Optimizer
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Calculate performance measures
        num_samples += labels.size(0)
        loss_train += loss.item()

        tgs_i.extend(label.item() for label in labels)
        sms = F.softmax(outputs, dim = 1)
        preds = torch.argmax(sms, 1)
        preds_i.extend(pred.item() for pred in preds)

    acc_train = 100*(accuracy_score(tgs_i, preds_i))
    loss_train = loss_train/num_samples

    return acc_train, loss_train

#%% VALIDATION AND TEST STEPS LOOPS

def evaluate_steps(net, set_loader, criterion):
    num_samples, loss_set = 0, 0

    tgs_i = [] # Inputs targets
    preds_i = [] # Predictions

    with torch.no_grad(): # Disables gradient calculation
        net = net.eval()
        for set in set_loader:
            inputs, labels = set
            inputs, labels = inputs.to(device), labels.to(device) # Put on GPU

            outputs = net(inputs) # Forward
            loss = criterion(outputs, labels) # Loss function

            # Calculate performance metrics
            num_samples += labels.size(0)
            loss_set += loss.item()

            tgs_i.extend(label.item() for label in labels)
            sms = F.softmax(outputs, dim = 1)
            preds = torch.argmax(sms, 1)
            preds_i.extend(pred.item() for pred in preds)

    acc_set = 100*(accuracy_score(tgs_i, preds_i))
    loss_set = loss_set/num_samples

    return acc_set, loss_set