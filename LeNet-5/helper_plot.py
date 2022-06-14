import torch 
import numpy as np
import matplotlib.pyplot as plt 

# visualize a sample test results
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