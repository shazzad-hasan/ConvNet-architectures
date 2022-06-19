import torch 
import numpy as np
import matplotlib.pyplot as plt 

def imshow(img):
  plt.imshow(np.squeeze(img), cmap='gray')

def show_examples(train_loader):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    inputs, targets = dataiter.next()
    inputs = inputs.numpy()
    
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(10, 4))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
        imshow(inputs[idx])
        ax.set_title(str(targets[idx].item()))
    plt.show()
        

# plot training and validation loss for each epoch

def plot_results(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, num_epochs):
    
    epochs = range(1, num_epochs+1)
    
    plt.plot(epochs, train_loss_list, 'bo', label='Training loss')
    plt.plot(epochs, valid_loss_list, 'b', label='Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, train_acc_list, 'bo', label='Training accuracy')
    plt.plot(epochs, valid_acc_list, 'b', label='Validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.show()
    

def show_sample_test_result(test_loader, model, classes, train_on_gpu, device):
    # obtain one batch of test images
    dataiter = iter(test_loader)
    inputs, targets = dataiter.next()
    inputs.numpy()
    inputs = inputs.to(device)
    
    # get sample outputs
    outputs = model(inputs)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(outputs, 1)
    predictions = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    
    # plot the images in the batch along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshow(inputs[idx] if not train_on_gpu else inputs[idx].cpu())
        ax.set_title("{} ({})".format(classes[predictions[idx]], classes[targets[idx]]),
                     color=("green" if predictions[idx]==targets[idx].item() else "red"))
    plt.show()