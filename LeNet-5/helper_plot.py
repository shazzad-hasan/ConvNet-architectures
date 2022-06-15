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
        ax.imshow(np.squeeze(inputs[idx]) if not train_on_gpu else np.squeeze(inputs[idx].cpu()), cmap="gray")
        ax.set_title("{} ({})".format(classes[predictions[idx]], classes[targets[idx]]),
                     color=("green" if predictions[idx]==targets[idx].item() else "red"))
    plt.show()