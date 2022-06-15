import os
import random
import numpy as np
import torch

def set_all_seeds(seed):
    """ sets seed for pseudo-random number generators in: 
        pytorch, numpy, python.random In addition, 
        sets the following environment variables """
    
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_loss_accuracy(model, test_loader, criterion, num_classes, classes, device):
    # track test loss and accuracy
    test_loss = 0.0
    class_correct = [0 for i in range(num_classes)]
    class_total = [0 for i in range(num_classes)]
    
    model.eval()
    
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # forward pass
        outputs = model(inputs)
        # calculate the batch loss
        loss = criterion(outputs, targets)
        # update test loss
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, predicted_labels = torch.max(outputs, 1) 
        # collect the correct predictions for each class
        for target, prediction in zip(targets, predicted_labels):
            if target == prediction:
                class_correct[target] += 1
            class_total[target] += 1
    
    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print("Test loss (overall): {:6f}\n".format(test_loss))
    
    # print test accuracy for each classes
    for i in range(num_classes):
      if class_total[i] > 0:
        accuracy = (100 * class_correct[i]) / class_total[i]
        print(f'Test accuracy of {classes[i]:10s}: {accuracy:.1f} % ({np.sum(class_correct[i])}/{np.sum(class_total[i])})')
    
    # overall test accuracy
    test_acc = 100 * np.sum(class_correct) / np.sum(class_total)
    print("\nTest accuracy (overall): %2d%% (%2d/%2d)" % ( 
          test_acc, np.sum(class_correct), np.sum(class_total)))
    
    return test_loss, test_acc 
    
