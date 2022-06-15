import torch
import numpy as np

def compute_valid_loss(model, data_loader, criterion, device):
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
          # move tensor to the right device
          inputs, targets = inputs.to(device), targets.to(device)
          # forward pass
          outputs = model(inputs)
          # calculate the batch loss
          logits = criterion(outputs, targets)
          
    return logits


def compute_accuracy(model, data_loader, criterion, device):
    
    with torch.no_grad():
    
        correct_pred, num_examples = 0, 0
        
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)
    
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
            
    return correct_pred.float()/num_examples * 100



def train(model, num_epochs, train_loader, valid_loader,
          test_loader, optimizer, criterion, device, 
          lr_scheduler=None, lr_scheduler_on = "valid_loss"):
    
    # track minibatch losses and accuracies 
    train_loss_list, valid_loss_list = [], []
    train_acc_list, valid_acc_list = [], []
    # initialize tracker for min validation loss
    min_valid_loss = np.inf
    
    for epoch in range(num_epochs):
      running_train_loss = 0.0
      running_valid_loss = 0.0
    
      # --------- train the model -----------------
      # set model to training mode
      model.train()
    
      for batch_idx, data in enumerate(train_loader):
        # get the inputs, data is a list of [inputs, targets]
        inputs, targets = data
        # mode tensor to the right device
        inputs, targets = inputs.to(device), targets.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # calculate the batch loss
        loss = criterion(outputs, targets)
        # backward pass
        loss.backward()
        # update parameters
        optimizer.step()
        # update training loss
        running_train_loss += loss.item()
    
      # ---------- validate the model ------------
      # set the model to evaluation mode
      model.eval()
    
      # since we're not training, we don't need to calculate the gradients for out outputs
      with torch.no_grad():
          loss = compute_valid_loss(model, valid_loader, device)
          # update validation loss
          running_valid_loss += loss.item()
          
          train_acc = compute_accuracy(model, train_loader, criterion, device)
          valid_acc = compute_accuracy(model, valid_loader, criterion, device)
          
          train_acc_list.append(train_acc.item())
          valid_acc_list.append(valid_acc.item()) 
    
      # calculate average loss over an epoch
      running_train_loss = running_train_loss / len(train_loader)
      running_valid_loss = running_valid_loss / len(valid_loader)
    
      train_loss_list.append(running_train_loss)
      valid_loss_list.append(running_valid_loss)
      
      
      if lr_scheduler is not None:
          
          if lr_scheduler_on == "valid_loss":
              lr_scheduler.step(valid_loss_list[-1])
          elif lr_scheduler_on == "train_loss":
              lr_scheduler.step(train_loss_list[-1])
          else:
              raise ValueError("Invalid `lr_scheduler_on` choice.")
        
              
    
      print("Epoch: {} \tTraining loss: {:.6f} \tValidation loss: {:.6f}".format(epoch+1, running_train_loss, running_valid_loss))
    
      # save model if validation loss has decressed
      if running_valid_loss <= min_valid_loss:
        print("Validation loss decressed ({:.6f} --> {:.6f}). Saving model ...".format(min_valid_loss, running_valid_loss))
        torch.save(model.state_dict(), "model.pt")
        min_valid_loss = running_valid_loss
    
    print("Finished training!")
    
    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list