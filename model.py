import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

class NoteIdentifier(nn.Module):
  def __init__(self):
    super(NoteIdentifier,self).__init__()
    self.name = "N1"
    alexnet = torchvision.models.alexnet(pretrained=True)
    transfer_net = alexnet.features
    self.transfer_net = transfer_net
    # for param in self.net.parameters():
    #   param.requires_grad = False
    self.fc1 = nn.Linear(256*6*6, 2048)
    self.fc2 = nn.Linear(2048, 128)
    

  def forward(self,notes):
    notes = self.transfer_net(notes)
    notes = notes.view(-1, 256 * 6* 6)
    notes = F.relu(self.fc1(notes))
    notes = self.fc2(notes)
    notes = notes.reshape(notes.shape[0],-1) # Flatten to [batch_size]
    return notes

class NoteIdentifier2(nn.Module):
  def __init__(self):
    super(NoteIdentifier2, self).__init__()
    self.name = "GI"
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 5, kernel_size = 5)#output size = ((224-5)/1)+1  X5 = 220x220x5
    self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) #output size = ((220-2)/2)+1 X5 = 110x110x5
    self.conv2 = nn.Conv2d(in_channels = 5, out_channels = 10, kernel_size = 5)#output size = (110-5)+1 X10 = 106X106X10
    #self.pool again so output size = ((106-2)/2)+1 X10 = 53x53x10
    self.fc1 = nn.Linear(10 * 53 * 53, 2048)
    self.fc2 = nn.Linear(2048, 88) #128 output one hot encoding

  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 10 * 53* 53)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    x = x.reshape(x.shape[0],-1) # Flatten to [batch_size]
    return x