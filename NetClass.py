import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, inputSize, numHiddenUnits, numClasses):
        super(Net, self).__init__()
        # self.numClasses = numClasses
        # self.LSTMlayer = nn.LSTM(input_size=inputSize, hidden_size=numHiddenUnits)
        self.fc1 = nn.Linear(in_features=inputSize, out_features=75)
        # self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=75, out_features=numClasses)

    def forward(self, x):
        # x, states = self.LSTMlayer(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=2)
        return x
