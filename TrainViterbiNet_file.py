import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader



def TrainViterbiNet(m_fXtrain, v_fYtrain, s_nConst, layers, learnRate):
    """
    Train ViterbiNet conditional distribution network

    Syntax
    -------------------------------------------------------
    net = TrainViterbiNet(m_fXtrain,v_fYtrain ,s_nConst, layers, learnRate)

    INPUT:
    -------------------------------------------------------
    m_fXtrain - training symobls corresponding to each channel output (memory x training size matrix)
    v_fYtrain - training channel outputs (vector with training size entries)
    s_nConst - constellation size (positive integer)
    layers - neural network model to train / re-train
    learnRate - learning rate (poitive scalar, 0 for default of 0.01)


    OUTPUT:
    -------------------------------------------------------
    net - trained neural network model
    """

    # s_nM = np.size(m_fXtrain, 0)
    #
    # # Combine each set of inputs as a single unique category
    # v_fCombineVec = s_nConst**np.array([np.arange(s_nM)])

    # format training to comply with Matlab's deep learning toolbox settings

    # v_fXcat = np.transpose(v_fCombineVec.dot(m_fXtrain-1))
    # v_fXcat = torch.from_numpy(np.reshape(v_fXcat, newshape=(np.size(v_fXcat, 0), np.size(v_fXcat, 1), 1)))
    # v_fYcat = np.transpose(v_fYtrain)
    # v_fYcat = torch.from_numpy(np.reshape(v_fYcat, newshape=(np.size(v_fYcat, 0), np.size(v_fYcat, 1), 1)))



    #-----validation data-----#
    # Y_valid = v_fYcat[4500:5000, :, :]
    # X_valid = v_fXcat[4500:5000, :, :]
    # v_fYcat = v_fYcat[0:4500, :, :]
    # v_fXcat = v_fXcat[0:4500, :, :]



    # if learnRate == 0:
    #     learnRate = 0.00001
    #
    # # Network parameters
    # maxEpochs = 50
    # miniBatchSize = 25
    # optimizer = optim.Adam(layers.parameters(), lr=learnRate)  #weight_decay=1e-4)
    # scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.1)


    # l = np.zeros((maxEpochs, 1))
    # l_valid = np.zeros((maxEpochs, 1))

    # Train network
    # for epoch in range(maxEpochs):

        # -----shuffle-----#
        # v_fYcat = np.reshape(v_fYcat.numpy(), newshape=(np.size(v_fYcat, 0), np.size(v_fYcat, 1), 1))
        # v_fXcat = np.reshape(v_fXcat.numpy(), newshape=(np.size(v_fXcat, 0), np.size(v_fXcat, 1), 1))
        # shuffler = np.random.permutation(5000)
        # v_fYcat[:, 0, 0] = v_fYcat[shuffler, 0, 0]
        # v_fXcat[:, 0, 0] = v_fXcat[shuffler, 0, 0]
        # v_fYcat = torch.from_numpy(np.reshape(v_fYcat, newshape=(np.size(v_fYcat, 0), np.size(v_fYcat, 1), 1)))
        # v_fXcat = torch.from_numpy(np.reshape(v_fXcat, newshape=(np.size(v_fXcat, 0), np.size(v_fXcat, 1), 1)))
        #-----------------#

        # for idx in range(int((np.shape(v_fYcat)[0])/miniBatchSize)):#
        #     optimizer.zero_grad()
        #     output = layers.forward(v_fYcat[(idx*miniBatchSize):((idx+1)*miniBatchSize), :, :].float())#
        #     # nn_output_labels = torch.reshape(torch.argmax(output, 2), shape=(miniBatchSize, 1, 1))
        #     loss = F.cross_entropy(torch.reshape(output[:, 0, :], shape=(miniBatchSize, 16)).float(),
        #                            v_fXcat[(idx*miniBatchSize):((idx+1)*miniBatchSize), 0, 0].long())
        #     # l1 = 0
        #     # for p in layers.parameters():
        #     #     l1 = l1+p.abs().sum()
        #     # loss = loss+(1e-5)*l1
        #
        #     loss.backward()
        #     optimizer.step()
        # print(loss)
        #
        # if l[epoch-1] > l[epoch]:
        #     scheduler1.step()
        # else:
        #     scheduler2.step()

    #     out_valid = layers.forward(Y_valid.float())  #
    #     l_valid[epoch] = float(F.cross_entropy(torch.reshape(out_valid[:, 0, :], shape=(500, 16)).float(), X_valid[:, 0, 0].long()))
    #     l[epoch] = float(loss)
    #
    # plt.figure()
    # plt.plot(range(epoch+1), l[:, 0],
    #          range(epoch+1), l_valid[:, 0])
    # plt.legend(('Data training loss', 'Validation loss'))
    # plt.show()






    # -----addition-----#
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(layers.parameters(), lr=learnRate) #, weight_decay=1e-4
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    epoch = 50
    miniBatchSize = 25

    combine_vec = np.array([[1, 2, 4, 8]])
    m_fXtrain = combine_vec.dot(m_fXtrain - 1)

    v_fYtrain = np.reshape(v_fYtrain, newshape=(np.size(v_fYtrain, 1), 1))
    # m_fXtrain = v_fCombineVec.dot(m_fXtrain-1)
    m_fXtrain = np.reshape(m_fXtrain, newshape=(np.size(m_fXtrain, 1), 1))
    m_fXtrain = to_categorical(m_fXtrain, 16)

    v_fYtrain = torch.from_numpy(v_fYtrain).float()
    m_fXtrain = torch.from_numpy(m_fXtrain).float()

    size_train_set = int(np.size(m_fXtrain, 0))
    train_set = TensorDataset(m_fXtrain[0:size_train_set], v_fYtrain[0:size_train_set])
    loader = DataLoader(train_set, batch_size=miniBatchSize, pin_memory=True, shuffle=True)
    running_loss = np.zeros(epoch)

    for ii in range(epoch):
        for x, y in loader:
            layers.zero_grad()
            optimizer.zero_grad()
            batch_outputs = layers.forward(y).float()
            loss = criterion(batch_outputs, torch.max(x, 1)[1])

            loss.backward()
            optimizer.step()

            running_loss[ii] += loss.item()
        scheduler.step()
        # print
    print('-training-')


    return layers


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    x = np.zeros(shape=(np.size(y, 0), num_classes))
    for i in range(np.size(y, 0)):
        x[i, y[i, 0].__int__()] = 1
    return x
