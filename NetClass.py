import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from v_fViterbi_file import v_fViterbi


class Net(nn.Module):
    def __init__(self, inputSize, numHiddenUnits, numClasses):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=inputSize, out_features=numHiddenUnits)
        self.fc2 = nn.Linear(in_features=numHiddenUnits, out_features=numClasses)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


    def TrainViterbiNet(self, X_train, Y_train, s_nConst, learnRate):
        """
        Train ViterbiNet conditional distribution network

        Syntax
        -------------------------------------------------------
        net = TrainViterbiNet(X_train,Y_train ,s_nConst, layers, learnRate)

        INPUT:
        -------------------------------------------------------
        X_train - training symobls corresponding to each channel output (memory x training size matrix)
        Y_train - training channel outputs (vector with training size entries)
        s_nConst - constellation size (positive integer)
        layers - neural network model to train / re-train
        learnRate - learning rate (poitive scalar, 0 for default of 0.01)


        OUTPUT:
        -------------------------------------------------------
        net - trained neural network model
        """

        # -----Training-----#
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learnRate)  # , weight_decay=1e-4
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
        epoch = 50
        miniBatchSize = 25

        # Combine each set of inputs as a single unique category
        s_nM = np.size(X_train, 0)
        combine_vec = s_nConst ** np.array([np.arange(s_nM)])  # np.array([[1, 2, 4, 8]])
        X_train = combine_vec.dot(X_train - 1)

        Y_train = np.reshape(Y_train, newshape=(np.size(Y_train, 1), 1))
        X_train = np.reshape(X_train, newshape=(np.size(X_train, 1), 1))
        X_train = to_categorical(X_train, 16)

        Y_train = torch.from_numpy(Y_train).float()
        X_train = torch.from_numpy(X_train).float()

        size_train_set = int(np.size(X_train, 0))
        train_set = TensorDataset(X_train[0:size_train_set], Y_train[0:size_train_set])
        loader = DataLoader(train_set, batch_size=miniBatchSize, pin_memory=True, shuffle=True)
        running_loss = np.zeros(epoch)

        for ii in range(epoch):
            for x, y in loader:
                self.zero_grad()
                optimizer.zero_grad()
                batch_outputs = self.forward(y).float()
                loss = criterion(batch_outputs, torch.max(x, 1)[1])

                loss.backward()
                optimizer.step()

                running_loss[ii] += loss.item()
            scheduler.step()
            # print
        print('-training-')


    def ApplyViterbiNet(self, Y_test, s_nConst, s_nMemSize):
        """
        # Apply ViterbiNet to observed channel outputs
        #
        # Syntax
        # -------------------------------------------------------
        # v_fXhat = ApplyViterbiNet(Y_test, net, GMModel, s_nConst)
        #
        # INPUT:
        # -------------------------------------------------------
        # Y_test - channel output vector
        # net - trained neural network model
        # GMModel - trained mixture model PDF estimate
        # s_nConst - constellation size (positive integer)
        # s_nMemSize - channel memory length
        #
        #
        # OUTPUT:
        # -------------------------------------------------------
        # v_fXhat - recovered symbols vector
        """
        s_nStates = s_nConst ** s_nMemSize
        # Use network to compute likelihood function
        Y_test = torch.from_numpy(np.reshape(Y_test, newshape=(np.size(Y_test, 0), np.size(Y_test, 1), 1)))
        m_fpS_Y = self.forward(Y_test.float())
        # Compute likelihoods
        m_fLikelihood = m_fpS_Y
        # Apply Viterbi output layer
        m_fLikelihood = np.reshape(m_fLikelihood.detach().numpy(), newshape=(50000, 16))
        v_fXhat = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize)

        return v_fXhat


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    x = np.zeros(shape=(np.size(y, 0), num_classes))
    for i in range(np.size(y, 0)):
        x[i, y[i, 0].__int__()] = 1
    return x
