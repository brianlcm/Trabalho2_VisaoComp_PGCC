#%% LIBRARIES

import matplotlib as mpl
import numpy as np
import os

from decimal import Decimal
from matplotlib import pyplot as plt

#%% LOGGER CLASS
'''
Store and print all performance measures
'''

class Logger:
    def __init__(self):
        self.acc_train = []
        self.loss_train = []

        self.acc_test = []
        self.loss_test = []

    def get_logs(self):
        return self.acc_train, self.loss_train, self.acc_test, self.loss_test

    def save_plts(self):
        # Style
        mpl.style.use('seaborn')
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times']
         })
        
        # Hyper-parameters
        checkpoint_save_dir = os.path.join(os.getcwd(), 'checkpoints')

        num_epochs = len(self.acc_train)
 
        acc_path = os.path.join(checkpoint_save_dir, 'acc.pdf')
        loss_path = os.path.join(checkpoint_save_dir, 'loss.pdf')

        x_s = np.arange(1, num_epochs+1, 1) # Abscissa
        
        #%% ACCURACY PLOT

        _, ax = plt.subplots(1, figsize = (40, 40))
        plt.plot(x_s, self.acc_train, label = 'Training Acc.', color = 'red')
        plt.plot(x_s, self.acc_test, label = 'Test Acc.', color = 'blue')

        plt.grid(True)

        # x-axis configuration
        plt.xlabel('Epochs', labelpad = 72, fontsize = 72)
        xticks = np.arange(0, num_epochs+1, num_epochs//5)
        plt.xticks(xticks, ['{}'.format(x) for x in xticks], fontsize = 60)
        plt.xlim(1, ax.get_xticks()[-1])

        # y-axis configuration
        plt.ylabel('Accuracy', labelpad = 72, fontsize = 72)
        yticks = np.arange(0, 101, 20)
        plt.yticks(yticks,  ['{}'.format(y) + r'\%' for y in yticks], fontsize = 60)
        plt.ylim(0, 100)

        plt.legend(loc = 'best', fontsize = 72, frameon = True)

        plt.savefig(acc_path, bbox_inches = 'tight', pad_inches = 0)

        #%% LOSS PLOT

        _, ax = plt.subplots(1, figsize = (40, 40))
        plt.plot(x_s, self.loss_train, label = 'Training Loss', color = 'red')
        
        plt.grid(True, color = 'white')

        # x-axis configuration
        plt.xlabel('Epochs', labelpad = 72, fontsize = 72)
        xticks = np.arange(0, num_epochs+1, num_epochs//5)
        plt.xticks(xticks, ['{}'.format(x) for x in xticks], fontsize = 60)
        plt.xlim(1, ax.get_xticks()[-1])

        # y-axis configuration
        plt.ylabel('Loss', labelpad = 72, fontsize = 72)

        plt.yticks(yticks, ['{:.2E}'.format(Decimal(y)) for y in yticks], fontsize = 60)
        plt.ylim(0, ax.get_yticks()[-1])

        plt.savefig(loss_path, bbox_inches = 'tight', pad_inches = 0)