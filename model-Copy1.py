import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class QNetworkCNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64 , dropout = 0.1 , augment_frames = 1 ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetworkCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(augment_frames * state_size[-1], 32, kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv1bnorm = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2,2), stride=1, padding=(1,1))
        self.conv2bnorm = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(2,2), stride=1, padding=(1,1))
        self.conv3bnorm = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.in_linear = 11*11*64
        
        self.fc1 = nn.Linear(self.in_linear, fc1_units)
        self.fc1bnorm = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units,  fc2_units)
        self.fc2bnorm = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv1(state)
        x = self.conv1bnorm(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv2bnorm(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv3bnorm(x)
        x = F.relu(x)
        x = self.pool(x)
        #print(x.size())
        x = x.view(-1,self.in_linear)
        x = self.fc1(x)
        x = self.fc1bnorm(x)
        x = F.relu(x)
#         x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc2bnorm(x)
        x = F.relu(x)
#         x = self.dropout(x)
        
        return self.fc3(x)
