import time
import os
import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch.cuda.amp import GradScaler
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from save_plots import Logger
from models.LeNet import LeNet
from train_steps import train_steps, evaluate_steps

def save(net, logger):

    path = os.path.join(os.path.join(os.getcwd(), 'checkpoints', 'LeNet'), 'last-epoch.pt')

    # Logger and net dictionary
    checkpoint = {
        'logs': logger.get_logs(),
        'params': net.state_dict()
    }

    torch.save(checkpoint, path) # Save checkpoint

if __name__ == '__main__':
    # Hyper-parameters
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 256

    logger = Logger()
    net = LeNet().to(device) # Put on GPU

    train_dataset = mnist.MNIST(root='./datasets/MNIST/train', train=True, transform = ToTensor())
    test_dataset = mnist.MNIST(root='./datasets/MNIST/test', train=False, transform = ToTensor())
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
 
    # Optimzer and learning rate scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.9, nesterov = True, weight_decay = 1e-4)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()

    print('Training started...')
    for epoch in range(0, num_epochs):
        t_start = time.time()

        # Steps loops
        acc_train, loss_train = train_steps(epoch, net, train_loader, criterion, optimizer, scaler)
        acc_test, loss_test = evaluate_steps(net, test_loader, criterion)

        logger.acc_train.append(acc_train)
        logger.loss_train.append(loss_train)
        logger.acc_test.append(acc_test)
        logger.loss_test.append(loss_test)

        print('Epoch {} | Train Acc. {:.2f}% | Test Acc. {:.2f}%\n'.format(epoch+1, acc_train, acc_test))
        
        
    print('Finished training.\n')

    # Save last epoch net weights and all measures graphics
    save(net, logger) 
    logger.save_plts()