import numpy as np
from m_fMyReshape_file import m_fMyReshape
from NetClass import Net
from TrainViterbiNet_file import TrainViterbiNet

def GetViterbiNet(v_fXtrain, v_fYtrain ,s_nConst, s_nMemSize):
    """
    Generate and train a new ViterbiNet conditional distribution network

    Syntax
    -------------------------------------------------------
    [net, GMModel] = GetViterbiNet(m_fXtrain,v_fYtrain ,s_nConst)

    INPUT:
    -------------------------------------------------------
    v_fXtrain - training symobls vector
    v_fYtrain - training channel outputs (vector with training size entries)
    s_nConst - constellation size (positive integer)
    s_nMemSize - channel memory length
    s_nMixtureSize - finite mixture size for PDF estimator (positive integer)


    OUTPUT:
    -------------------------------------------------------
    net - trained neural network model
    GMModel - trained mixture model PDF estimate

    Reshape input symbols into a matrix representation
    """

    m_fXtrain = m_fMyReshape(v_fXtrain, s_nMemSize)

    # Generate neural network
    inputSize = 1
    numHiddenUnits = 75
    numClasses = s_nConst ** s_nMemSize

    """Work around converting an LSTM, which is the supported first layer
    for seuquence proccessing networks in Matlab, into a perceptron with sigmoid activation"""

    # Generate network model
    net = Net(inputSize, numHiddenUnits, numClasses)


    # Train network with default learning rate
    # net = TrainViterbiNet(m_fXtrain, v_fYtrain, s_nConst, net, 0.00005)
    net.TrainViterbiNet(m_fXtrain, v_fYtrain, s_nConst, 0.00005)

    # # Compute output PDF using GMM fitting
    # GMModel = fitgmdist(v_fYtrain',s_nMixtureSize,'RegularizationValue',0.1)


    return  net
