from dataloader import parse_data
from model import NoteIdentifier
from train import train_net, get_model_name, plot_training_curve

import torch


if __name__ == "__main__":
  model_alex = NoteIdentifier()
  model_alex.name = "alex1"
  if torch.cuda.is_available():
    model_alex.cuda()
    print('CUDA is available!  Training on GPU ...')
  else:
    print('CUDA is not available.  Training on CPU ...')
  
  train_loader, val_loader, test_loader = parse_data()
  #proper model
  print(len(train_loader))
  train_net(model_alex, train_loader, train_loader, batch_size=256)
  path = get_model_name(model_alex.name, 256, 0.0005, 9)
  plot_training_curve(path)