#Training code for the model
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

# Plot Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

#calculate training accuracy
def calculate_error(labels, predictions):
  error_count = 0
  for label,prediction in zip(labels, predictions):
    if label != prediction:
      error_count += 1

  # print("error_count was: ", error_count)
  return error_count

#Validation/testing code
def evaluate(net, loader, criterion):
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        if i == 20:
          break
        input, labels = data
        ############################################
        #To Enable GPU Usage
        # input = input.repeat((1,3,1,1))
        if torch.cuda.is_available():
          input = input.cuda()
          labels = labels.cuda()
        ############################################
        outputs = net(input)
        loss = criterion(outputs, labels)
        predictions = outputs.max(1)[1]
        total_err += calculate_error(labels,predictions)
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss
    
#Define a unique model name to save, like Alpha Go.
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:                            batch_size,
                       
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def train_net(net, train_data, val_data, batch_size=64, learning_rate=0.0005, num_epochs=10):

  ########################################################################
  # Fixed PyTorch random seed for reproducible result
  torch.manual_seed(1000)
  ########################################################################
  ########################################################################
  # Define the Loss function and optimizer
  # The loss function Cross_Entropy_Loss which already has an implementation
  # of softmax within.
  # Optimizer will be SGD with Momentum.
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)
  ########################################################################
  # Set up some numpy arrays to store the training/test loss/erruracy
  train_err = np.zeros(num_epochs)
  train_acc = np.zeros(num_epochs)
  train_loss = np.zeros(num_epochs)
  val_err = np.zeros(num_epochs)
  val_acc = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)
  files = []
  ########################################################################
  # Train the network
  # Loop over the data iterator and sample a new batch of training data
  # Get the output from the network, and optimize our loss function.
  start_time = time.time()
  for epoch in range(num_epochs):  # loop over the dataset multiple times
      total_train_loss = 0.0
      total_train_err = 0.0
      total_epoch = 0
      for i, data in enumerate(train_data, 0):
          # Get the inputs
          # if i == 50:
          #   break
          print(i)
          inputs, labels = data
          # inputs = inputs.repeat((1,3,1,1))
          #############################################
          #To Enable GPU Usage
          if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
          #############################################
          # Zero the parameter gradients
          optimizer.zero_grad()
          # Forward pass, backward pass, and optimize
          outputs = net(inputs)
          # print(outputs.shape)
          # print(outputs)
          predictions = outputs.max(1)[1]
          # print("Predictions:", predictions)
          # print("After Predictions")
          loss = criterion(outputs, labels)
          # print("After Loss")
          loss.backward()
          # print("After loss.backward")
          optimizer.step()
          #print("After optimizer.step")

          
          # Need implementation!!
          # Calculate the statistics
          # corr = (outputs > 0.0).squeeze().long() != labels
          # print("error:", calculate_error(labels,predictions))
          total_train_err += calculate_error(labels,predictions)
          total_train_loss += loss.item()
          total_epoch += len(labels)

      print("Finished {} loop".format(epoch+1))
      train_err[epoch] = float(total_train_err) / total_epoch
      train_acc[epoch] = 1 - train_err[epoch]
      train_loss[epoch] = float(total_train_loss) / (i+1)
      val_err[epoch], val_loss[epoch] = evaluate(net, val_data, criterion)
      val_acc[epoch] = 1-val_err[epoch]
      print(("Epoch {}: Train accuracy: {}, Train loss: {} |"+
              "Validation accuracy: {}, Validation loss: {}").format(
                  epoch + 1,
                  train_acc[epoch],
                  train_loss[epoch],
                  val_acc[epoch],
                  val_loss[epoch]))
      # Save the current model (checkpoint) to a file
  model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
  files.append(model_path)
  torch.save(net.state_dict(), model_path)
  print('Finished Training')
  end_time = time.time()
  elapsed_time = end_time - start_time
  print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
  # Write the train/test loss/err into CSV file for plotting later
  epochs = np.arange(1, num_epochs + 1)
  
  np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
  np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
  np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
  np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
  return files