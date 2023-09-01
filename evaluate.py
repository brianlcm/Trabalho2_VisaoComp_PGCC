#%% LIBRARIES

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sb
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score

from save_plots import Logger
from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.VGG import VGG

device = torch.device('cpu')

def restore(net, logger, path):

    path = os.path.join(path, 'last-epoch.pt')

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)
            
            net.load_state_dict(checkpoint['params'])
            logger.restore_logs(checkpoint['logs'])
            
            print('Network Restored!')
        except Exception as e:
            print('Restore Failed!')
            print(e)
    else:
        print('Restore point unavailable.')

def evaluate(net, test_loader, dataset):
    
    checkpoint_save_dir = os.path.join(os.getcwd(), 'experiments', model, dataset,'experiment-{}'.format(experiment))

    num_samples = 0
    
    tgs_i = [] # Inputs targets
    preds_i = [] # Predictions

    with torch.no_grad(): # Disables gradient calculation
        net = net.eval()
        for test in test_loader:
            inputs, labels = test
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs) # Forward

            # Calculate performance metrics
            num_samples += labels.size(0)

            tgs_i.extend(label.item() for label in labels)
            confs = F.softmax(outputs, dim = 1)
            preds = torch.argmax(confs, 1)
            preds_i.extend(pred.item() for pred in preds)

    acc = 100*(accuracy_score(tgs_i, preds_i))
    cf_matrix_abs = confusion_matrix(tgs_i, preds_i)
    cf_matrix_rel = 100*cf_matrix_abs/cf_matrix_abs.sum(axis = 1)[:, np.newaxis]

    print('Accuracy: {:.8f}%'.format(acc))
    print('Confusion Matrix:\n', cf_matrix_rel, '\n')

    ploting_confusion_matrix(cf_matrix_abs, cf_matrix_rel, acc, checkpoint_save_dir, dataset)

    #%% MEASURES SAVE

    precision = 100*precision_score(tgs_i, preds_i, average = None)
    recall = 100*recall_score(tgs_i, preds_i, average = None)
    f1 = 100*f1_score(tgs_i, preds_i, average = None)
    f2 = 100*fbeta_score(tgs_i, preds_i, average = None, beta = 2)
    
    dict_metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2}
    df_metrics = pd.DataFrame(dict_metrics)
    df_metrics.to_csv('{}/measures_last-epoch.csv'.format(checkpoint_save_dir), index = False)

#%% CONFUSION MATRIX PLOT

def ploting_confusion_matrix(cf_matrix_abs, cf_matrix_rel, acc, checkpoint_save_dir, dataset):
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times']
    })

    if dataset == 'FashionMNIST':
        classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset == 'CIFAR10':
        classes_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    cf_matrix_path = os.path.join(checkpoint_save_dir, 'cf_matrix.pdf')
    
    group_counts = ['{0:.0f}'.format(value) for value in cf_matrix_abs.flatten()] # Confusion matrix with absolute values
    
    group_percentages = ['{0:.2f}'.format(value) for value in cf_matrix_rel.flatten()] # Confusion matrix with relative values
    
    # Confusion matrix with both absolute and relative values
    annot = [f'{v1}' + r'%' + f'\n{v2}' for v1, v2 in zip(group_percentages, group_counts)]
    annot = np.asarray(annot).reshape(len(classes_names) , len(classes_names))
    
    # Ploting
    _, ax = plt.subplots(1, figsize = (30, 30))
    
    sb.set(rc = {'text.usetex': False, 'font.family': 'serif', 'font.serif': ['Times']}, font_scale = 3)
    ax = sb.heatmap(cf_matrix_rel,
                    xticklabels = classes_names,
                    yticklabels = classes_names,
                    annot = annot,
                    fmt = '',
                    cmap = 'Blues',
                    cbar = False)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(length = 0)

    plt.xlabel('Acc.: {:.2f}'.format(acc) + r'%', labelpad = 30, fontsize = 30)

    plt.xticks(plt.xticks()[0], [label._text for label in plt.xticks()[1]], fontsize = 30)
    
    plt.yticks(plt.yticks()[0], [label._text for label in plt.yticks()[1]], fontsize = 30, rotation = 90)
    
    plt.savefig(cf_matrix_path, bbox_inches = 'tight', pad_inches = 0)

if __name__ == '__main__':
    # Build network (last epoch)

    model = 'LeNet' # LeNet, AlexNet, VGG
    dataset = 'FashionMNIST' # FashionMNIST, Cifar-10
    experiment = 8
    path = os.path.join(os.getcwd(), 'experiments', model, dataset,'experiment-{}'.format(experiment))

    logger = Logger()

    if model == 'LeNet':
        net = LeNet().to(device) # Put on GPU
    elif model == 'AlexNet':
        net = AlexNet().to(device) # Put on GPU
    elif model == 'VGG':
        net = VGG().to(device) # Put on GPU

    restore(net, logger, path)

    if dataset == 'FashionMNIST':
        test_dataset = datasets.FashionMNIST(root='./datasets/FashionMNIST/test', train = False, download = False, transform = ToTensor())
        test_loader = DataLoader(test_dataset)
    
    elif dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(root='./datasets/CIFAR10/test', train = False, download = False, transform = ToTensor())
        test_loader = DataLoader(test_dataset)

    evaluate(net, test_loader, dataset)