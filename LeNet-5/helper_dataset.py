import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def dataloader_mnist(batch_size, 
                     num_workers = 0,
                     train_transform = None, 
                     test_transform = None, 
                     valid_size = None):

    if train_transform is None:
        train_transform = transforms.ToTensor()
        
    if test_transform is None:
        test_transform = transforms.ToTensor()
    
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
    
    if valid_size is not None:
        # obtain training indices for creating a validation dataset
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_indices, valid_indices = indices[split:], indices[:split]
        
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
        # prepare train, test and validation data loaders
        train_loader = DataLoader(dataset = train_data, 
                                  batch_size = batch_size,
                                  sampler = train_sampler, 
                                  num_workers = num_workers)
        valid_loader = DataLoader(dataset = train_data, 
                                  batch_size = batch_size, 
                                  sampler = valid_sampler, 
                                  num_workers = num_workers,
                                  shuffle = False)
    else:
        
        train_loader = DataLoader(dataset = train_data,
                                  batch_size = batch_size,
                                  num_workers=num_workers,
                                  shuffle = True)
        
    test_loader = DataLoader(dataset = test_data,
                             batch_size = batch_size,
                             num_workers = num_workers,
                             shuffle = False)

    # image classes in the dataset
    classes = train_data.classes 
    
    if valid_size is None:
        return train_loader, test_loader, classes
    
    else:
        return train_loader, valid_loader, test_loader, classes 