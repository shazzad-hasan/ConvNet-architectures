import torch 
import numpy as np
import matplotlib.pyplot as plt 


def show_examples(train_loader):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()
    
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(str(labels[idx].item()))
    plt.show()
        
    

def show_sample_test_result(test_loader, model, classes, device):
    # get one batch of test images
    dataiter = iter(test_loader)
    inputs, targets = dataiter.next()
    inputs.numpy()
    
    inputs = inputs.to(device)
    
    # get sample outputs
    outputs = model(inputs)
    # convert output probabilities to predicted class
    _, predicted_labels = torch.max(outputs, 1)
    
    # plot the images in the batch along with predicted class and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        ax.imshow(inputs[idx])
        ax.set_title("{} ({})".format(classes[predicted_labels[idx]], classes[targets[idx]]),
                     color=("green" if predicted_labels[idx]==targets[idx].item() else "red"))  
    plt.show()