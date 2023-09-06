import time
import os
import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch.cuda.amp import GradScaler
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from save_plots import Logger
from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.VGG import VGG
from train_steps import train_steps, evaluate_steps

def save(net, logger, path):

    path = os.path.join(path, 'last-epoch.pt')

    # Logger and net dictionary
    checkpoint = {
        'logs': logger.get_logs(),
        'params': net.state_dict()
    }

    torch.save(checkpoint, path) # Save checkpoint

if __name__ == '__main__':
    # Hyper-parameters
    learning_rate = 1e-1
    num_epochs = 50
    batch_size = 512
    model = 'LeNet' # LeNet, AlexNet, VGG
    dataset = 'CIFAR10' # FashionMNIST, CIFAR10
    experiment = 10
    path = os.path.join(os.getcwd(), 'experiments', model, dataset,'experiment-{}'.format(experiment))

    if not os.path.exists(path):
            os.makedirs(path)

    logger = Logger()
    
    if model == 'LeNet':
        net = LeNet().to(device)
    elif model == 'AlexNet':
        net = AlexNet().to(device)
    elif model == 'VGG':
        net = VGG().to(device)

    print(net)

    
    # Define a transform to convert to images to tensor and normalize
    transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(degrees=(30,60)),
                                    transforms.Normalize((0.5,),(0.5,),)]) # mean and std have to be sequences (e.g., tuples), 
                                                                        # therefore we should add a comma after the values
    
    '''
    AlexTransform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    '''

    if dataset == 'FashionMNIST':
        if os.path.exists(os.path.join(os.getcwd(), 'datasets', 'FashionMNIST')):
            train_dataset = datasets.FashionMNIST(root='./datasets/FashionMNIST/train', train = True, download = False, transform = ToTensor())
            test_dataset = datasets.FashionMNIST(root='./datasets/FashionMNIST/test', train = False, download = False, transform = ToTensor())
            train_loader = DataLoader(train_dataset, batch_size = batch_size)
            test_loader = DataLoader(test_dataset, batch_size = batch_size)
        else:
            train_dataset = datasets.FashionMNIST(root='./datasets/FashionMNIST/train', train = True, download = True, transform = AlexTransform)
            test_dataset = datasets.FashionMNIST(root='./datasets/FashionMNIST/test', train = False, download = True, transform = AlexTransform)
            train_loader = DataLoader(train_dataset, batch_size = batch_size)
            test_loader = DataLoader(test_dataset, batch_size = batch_size)
    
    elif dataset == 'CIFAR10':
        if os.path.exists(os.path.join(os.getcwd(), 'datasets', 'CIFAR10')):
            train_dataset = datasets.CIFAR10(root='./datasets/CIFAR10/train', train = True, download = False, transform = transforms)
            test_dataset = datasets.CIFAR10(root='./datasets/CIFAR10/test', train = False, download = False, transform = transforms)
            train_loader = DataLoader(train_dataset, batch_size = batch_size)
            test_loader = DataLoader(test_dataset, batch_size = batch_size)
        else:
            train_dataset = datasets.CIFAR10(root='./datasets/CIFAR10/train', train = True, download = True, transform = ToTensor())
            test_dataset = datasets.CIFAR10(root='./datasets/CIFAR10/test', train = False, download = True, transform = ToTensor())
            train_loader = DataLoader(train_dataset, batch_size = batch_size)
            test_loader = DataLoader(test_dataset, batch_size = batch_size)
 
    # Optimzer and learning rate scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
    
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

        print('Epoch {} | Train Acc. {:.2f}% | Test Acc. {:.2f}% | Train Loss {:.5f}% | Test Loss {:.5f}%\n'.format(epoch+1, acc_train, acc_test, loss_train, loss_test))
        
        
    print('Finished training.\n')

    # Save last epoch net weights and all measures graphics
    save(net, logger, path) 
    logger.save_plts(path)