# ViterbiNet Channel tracking example code - ISI channel with AWGN

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from GetViterbiNet_file import GetViterbiNet
# import TrainViterbiNet
from ApplyViterbiNet_file import ApplyViterbiNet
from v_fViterbi_file import v_fViterbi
from m_fMyReshape_file import m_fMyReshape
from SOVA_path_and_SER import *
import copy
import collections
from graph_plot_func import *


def Xhat_viterbinet(net1, net2, net3, net4, v_fYtest, s_nConst, s_nMemSize):
    likelihood1 = net1.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    likelihood2 = net2.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    likelihood3 = net3.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    likelihood4 = net4.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    ensamble_likelihood = (likelihood1 + likelihood2 + likelihood3 + likelihood4) / 4
    Xhat = v_fViterbi(ensamble_likelihood, s_nConst, s_nMemSize)
    return Xhat


np.random.seed(9001)

# ----------Parameters Setting----------#
s_nConst = 2  # Constellation size (2 = BPSK)
s_nMemSize = 4  # Number of taps
s_fTrainSize = 5000  # Training size
s_fTestSize = 50000  # Test data size
s_fTrainBlkSize = 2000  # Block size
s_fCompositeBlocks = 25  # Number of blocks
track_DataSize_per = 0.2  #  Datasize for re-training [%]
med_point = 0.4  #  median point


s_nStates = s_nConst ** s_nMemSize

v_fSigWdB = np.array([np.arange(-6, 11, 2)])    # Noise variance in dB
# v_fSigWdB = np.array([[4]])

s_fEstErrVar = 0.1  # Estimation error variance
# Frame size for generating noisy training
s_fFrameSize = 500
s_fNumFrames = s_fTrainSize / s_fFrameSize

v_nCurves = [  # Curves
    1,  # Deep Viterbi - perfect CSI
    1,  # Deep Viterbi - CSI uncertainty
    1  # Viterbi algorithm
]

s_nCurves = np.size(v_nCurves)

v_stProts = (
    'ViterbiNet, perfect CSI',
    'ViterbiNet, CSI uncertainty',
    'Viterbi algorithm')


# ----------Simulation Loop----------#
v_fExps = np.array([[0.5]])
m_fSERAvg = np.zeros((np.size(v_nCurves), np.size(v_fSigWdB)))
m_fSER = np.zeros((np.size(v_nCurves), np.size(v_fSigWdB), np.size(v_fExps)))

for eIdx in range(np.size(v_fExps)):
    # Exponentailly decaying channel
    v_fChannel = np.array([np.exp(-v_fExps[0, eIdx] * np.arange(0, s_nMemSize, 1))])

    # Generate trainin labels
    v_fXtrain = np.array([np.random.randint(1, 3, s_fTrainSize)])
    v_fStrain = 2 * (v_fXtrain - 0.5 * (s_nConst + 1))
    m_fStrain = m_fMyReshape(v_fStrain, s_nMemSize)

    # Training with perfect CSI
    v_Rtrain = np.dot(np.fliplr(v_fChannel), m_fStrain)
    # Training with noisy CSI
    v_Rtrain2 = np.array([np.zeros((np.size(v_Rtrain)))])
    for kk in range(int(s_fNumFrames)):
        Idxs = np.arange((kk * s_fFrameSize), (kk + 1) * s_fFrameSize)
        v_Rtrain2[0, Idxs] = np.fliplr(
            v_fChannel + np.sqrt(s_fEstErrVar) * np.dot(np.array([np.random.randn(np.size(v_fChannel))]),
                                                        np.diag(v_fChannel[0, :]))).dot(m_fStrain[:, Idxs])

    # Generate test labels
    v_fXtest = np.array([np.random.randint(1, 3, s_fTestSize)])
    v_fStest = 2 * (v_fXtest - 0.5 * (s_nConst + 1))
    m_fStest = m_fMyReshape(v_fStest, s_nMemSize)
    v_Rtest = np.dot(np.fliplr(v_fChannel), m_fStest)

    # Generate vectors to hold the success rates
    # perfect CSI
    d_path1 = np.zeros((np.size(v_fSigWdB), 4))
    d_symbol1 = np.zeros((np.size(v_fSigWdB), 4))
    # CSU uncertainty
    d_path2 = np.zeros((np.size(v_fSigWdB), 4))
    d_symbol2 = np.zeros((np.size(v_fSigWdB), 4))

    track_SER_avg = np.zeros((13, np.size(v_fSigWdB)))
    track_PER_avg = np.zeros((13, np.size(v_fSigWdB)))

    track_SER_perfectCSI = np.zeros((6, 25, np.size(v_fSigWdB)))
    track_SER_uncertaintyCSI = np.zeros((6, 25, np.size(v_fSigWdB)))
    track_SER_fullCSI = np.zeros((1, 25, np.size(v_fSigWdB)))
    track_PER_perfectCSI = np.zeros((6, 25, np.size(v_fSigWdB)))
    track_PER_uncertaintyCSI = np.zeros((6, 25, np.size(v_fSigWdB)))
    track_PER_fullCSI = np.zeros((1, 25, np.size(v_fSigWdB)))


    # Loop over number of SNR
    for mm in range(np.size(v_fSigWdB)):
        s_fSigmaW = 10 ** (-0.1 * (v_fSigWdB[0, mm]))
        # LTI AWGN channel
        v_fYtrain = v_Rtrain + np.sqrt(s_fSigmaW) * np.random.randn(np.size(v_Rtrain))
        v_fYtrain2 = v_Rtrain2 + np.sqrt(s_fSigmaW) * np.random.randn(np.size(v_Rtrain))
        v_fYtest = v_Rtest + np.sqrt(s_fSigmaW) * np.random.randn(np.size(v_Rtest))

        # Viterbi net - perfect CSI
        if v_nCurves[0] == 1:
            # Train network
            net1 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)
            # -----ensamble-----#
            # net1_1 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # net1_2 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # net1_3 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # v_fXhat1 = Xhat_viterbinet(net1, net1_1, net1_2, net1_3, v_fYtest, s_nConst, s_nMemSize)

            # Apply ViterbiNet detector
            v_fXhat1 = net1.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)

            # Evaluate error rate
            m_fSER[0, mm, eIdx] = np.mean(v_fXhat1[0, :] != v_fXtest)

            # -----soft output-----#
            datasize_and_err_path1, datasize_and_err_symbol1 = sova_corelation(v_fXtest, v_fXhat1)
            d_path1[mm, :] = get_d(datasize_and_err_path1)
            d_symbol1[mm, :] = get_d(datasize_and_err_symbol1)

        # Viterbi net - CSI uncertainty
        if v_nCurves[1] == 1:
            # Train network using training with uncertainty
            net2 = GetViterbiNet(v_fXtrain, v_fYtrain2, s_nConst, s_nMemSize)
            # -----ensamble-----#
            # net2_1 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # net2_2 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # net2_3 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # v_fXhat2 = Xhat_viterbinet(net2, net2_1, net2_2, net2_3, v_fYtest, s_nConst, s_nMemSize)

            # Apply ViterbiNet detector
            v_fXhat2 = net2.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)

            # Evaluate error rate
            m_fSER[1, mm, eIdx] = np.mean(v_fXhat2[0, :] != v_fXtest)

            # -----soft output-----#
            datasize_and_err_path2, datasize_and_err_symbol2 = sova_corelation(v_fXtest, v_fXhat2)
            d_path2[mm, :] = get_d(datasize_and_err_path2)
            d_symbol2[mm, :] = get_d(datasize_and_err_symbol2)

        # Model-based Viterbi algorithm
        if v_nCurves[2] == 1:
            m_fLikelihood = np.array(np.zeros((s_fTestSize, s_nStates)))
            # Compute conditional PDF for each state
            for ii in range(s_nStates):
                v_fX = np.zeros((s_nMemSize, 1))
                Idx = ii
                for ll in range(s_nMemSize):
                    v_fX[ll] = Idx % s_nConst + 1
                    Idx = np.floor(Idx / s_nConst)
                v_fS = 2 * (v_fX - 0.5 * (s_nConst + 1))
                m_fLikelihood[:, ii] = stats.norm.pdf(v_fYtest - np.fliplr(v_fChannel).dot(v_fS), 0, s_fSigmaW)
            # Apply Viterbi detection based on computed likelihoods
            v_fXhat3 = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize)
            # Evaluate error rate
            m_fSER[2, mm, eIdx] = np.mean(v_fXhat3[0, :] != v_fXtest)

        # Display SNR index
        print(mm)
        print(m_fSER[:, :, eIdx])



        #----------Channel Tracking ViterbiNet----------#
        v_fXtrain2 = np.array(np.random.randint(low=1, high=3, size=(1, s_fTrainBlkSize * s_fCompositeBlocks)))
        v_fStrain = 2 * (v_fXtrain2 - 0.5 * (s_nConst + 1))

        #exponential channel taps variation
        m_fChannel = np.zeros((200, 4))
        channel_finit = np.array([0.8, 0.45, 0.9, 0.75])
        m_fChannel[:, 0] = channel_finit[0] + (v_fChannel[0, 0] - channel_finit[0]) * np.exp(-np.array(range(200)) / 55)
        m_fChannel[:, 1] = channel_finit[1] + (v_fChannel[0, 1] - channel_finit[1]) * np.exp(-np.array(range(200)) / 70)
        m_fChannel[:, 2] = channel_finit[2] + (v_fChannel[0, 2] - channel_finit[2]) * np.exp(-np.array(range(200)) / 40)
        m_fChannel[:, 3] = channel_finit[3] + (v_fChannel[0, 3] - channel_finit[3]) * np.exp(-np.array(range(200)) / 60)

        # Networks deep copy
        m_fRtrain2 = np.zeros((s_fCompositeBlocks, s_fTrainBlkSize))
        path_test_B = path(v_fXtrain2 - 1)
        net1_train = copy.deepcopy(net1)
        net2_train = copy.deepcopy(net2)
        net1_NoTrain = copy.deepcopy(net1)
        net2_NoTrain = copy.deepcopy(net2)
        net1_BadTrain = copy.deepcopy(net1)
        net2_BadTrain = copy.deepcopy(net2)
        net1_MedTrain = copy.deepcopy(net1)
        net2_MedTrain = copy.deepcopy(net2)
        net1_jini = copy.deepcopy(net1)
        net2_jini = copy.deepcopy(net2)
        net1_jini_p = copy.deepcopy(net1)
        net2_jini_p = copy.deepcopy(net2)

        MSE_diversity_train = np.zeros((1, s_fCompositeBlocks))
        MSE_diversity_bad_train = np.zeros((1, s_fCompositeBlocks))
        MSE_diversity_med_train = np.zeros((1, s_fCompositeBlocks))

        NOISE_train = np.zeros((1, s_fCompositeBlocks))
        NOISE_bad_train = np.zeros((1, s_fCompositeBlocks))
        NOISE_med_train = np.zeros((1, s_fCompositeBlocks))

        ERR_train = np.zeros((1, s_fCompositeBlocks))
        ERR_bad_train = np.zeros((1, s_fCompositeBlocks))
        ERR_med_train = np.zeros((1, s_fCompositeBlocks))


        for kk in range(s_fCompositeBlocks):
            print(kk)
            m_fStrain2 = m_fMyReshape(v_fStrain[:, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))],
                                      s_nMemSize)
            m_fRtrain2[kk, :] = np.fliplr(np.reshape(m_fChannel[int(3 * kk), :], (1, 4))).dot(m_fStrain2)
            noise = np.sqrt(s_fSigmaW) * np.random.randn(np.size(m_fRtrain2[kk, :]))
            Y = np.reshape(m_fRtrain2[kk, :], (1, s_fTrainBlkSize)) + noise

            #-----ViterbiNet perfect CSI-----#
            if v_nCurves[0] == 1:

                #training - high score training
                v_fXhat_B = net1_train.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_perfectCSI[3, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_perfectCSI[3, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold = sort_price[int((1-track_DataSize_per) * s_fTrainBlkSize)]
                Xtrain_symbol = v_fXhat_B[
                    2, v_fXhat_B[1,
                       :] >= threshold]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, v_fXhat_B[1, :] >= threshold]

                net1_train.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                # MSE diversity
                path_temp = path_test_B[:, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]  #
                tem_train = path_temp[:, v_fXhat_B[1, :] >= threshold]  #
                dict_train = collections.Counter(tem_train[0, :])  #
                data = np.array(list(dict_train.items()))
                an_array = data[:, 1]
                an_optimal_array = np.fix(track_DataSize_per*s_fTrainBlkSize/16)*np.ones((16, 1))
                MSE_diversity_train[0, kk] = np.square(np.subtract(an_array, an_optimal_array)).mean()
                print('train diversity MSE - ', MSE_diversity_train[0, kk])

                # Noise power
                NOISE_train[0, kk] = np.sum(np.abs(noise[v_fXhat_B[1, :] >= threshold])**2)
                print('train noise power - ', NOISE_train[0, kk])

                # SER
                tem = path_test_B[:, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]
                ERR_train[0, kk] = np.mean(Xtrain_symbol != tem[:, v_fXhat_B[1, :] >= threshold])
                print('train error - ', ERR_train[0, kk])


                # no training
                v_fXhat_B = net1_NoTrain.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_perfectCSI[1, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_perfectCSI[1, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])


                #bad training - low score training
                v_fXhat_B = net1_BadTrain.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_perfectCSI[2, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_perfectCSI[2, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold = sort_price[int(track_DataSize_per * s_fTrainBlkSize)]
                Xtrain_symbol = v_fXhat_B[
                    2, v_fXhat_B[1,
                       :] < threshold]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, v_fXhat_B[1, :] < threshold]

                net1_BadTrain.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                # MSE diversity
                tem_bad_train = path_temp[:, v_fXhat_B[1, :] < threshold]  #
                dict_bad_train = collections.Counter(tem_bad_train[0, :])  #
                data = np.array(list(dict_bad_train.items()))
                an_array = data[:, 1]
                an_optimal_array = np.fix(track_DataSize_per * s_fTrainBlkSize / 16) * np.ones((16, 1))
                MSE_diversity_bad_train[0, kk] = np.square(np.subtract(an_array, an_optimal_array)).mean()
                print('bad train diversity MSE - ', MSE_diversity_bad_train[0, kk])

                # Noise power
                NOISE_bad_train[0, kk] = np.sum(np.abs(noise[v_fXhat_B[1, :] < threshold])**2)
                print('bad train noise power - ', NOISE_bad_train[0, kk])

                # SER
                tem = path_test_B[:, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]
                ERR_bad_train[0, kk] = np.mean(Xtrain_symbol != tem[:, v_fXhat_B[1, :] < threshold])
                print('bad train error - ', ERR_bad_train[0, kk])


                #Median training
                v_fXhat_B = net1_MedTrain.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_perfectCSI[0, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_perfectCSI[0, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold1 = sort_price[int((med_point - track_DataSize_per/2) * s_fTrainBlkSize)]
                threshold2 = sort_price[int((med_point + track_DataSize_per/2) * s_fTrainBlkSize)]
                aaa = v_fXhat_B

                Xtrain_symbol = v_fXhat_B[
                    2, (v_fXhat_B[1, :] >= threshold1) & (v_fXhat_B[1, :] < threshold2)]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, (v_fXhat_B[1, :] >= threshold1) & (v_fXhat_B[1, :] < threshold2)]

                net1_MedTrain.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                # MSE diversity
                path_temp = path_test_B[:, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]  #
                tem_train = path_temp[:, (v_fXhat_B[1, :] >= threshold1) & (v_fXhat_B[1, :] < threshold2)]  #
                dict_med_train = collections.Counter(tem_train[0, :])  #
                data = np.array(list(dict_med_train.items()))
                an_array = data[:, 1]
                an_optimal_array = np.fix(track_DataSize_per * s_fTrainBlkSize / 16) * np.ones((16, 1))
                MSE_diversity_med_train[0, kk] = np.square(np.subtract(an_array, an_optimal_array)).mean()
                print('median training diversity MSE - ', MSE_diversity_med_train[0, kk])

                # Noise power
                NOISE_med_train[0, kk] = np.sum(np.abs(noise[(v_fXhat_B[1, :] >= threshold1) & (v_fXhat_B[1, :] < threshold2)])**2)
                print('median training noise power - ', NOISE_med_train[0, kk])

                # SER
                tem = path_test_B[:, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]
                ERR_med_train[0, kk] = np.mean(Xtrain_symbol != tem[:, (v_fXhat_B[1, :] >= threshold1) & (v_fXhat_B[1, :] < threshold2)])
                print('train error - ', ERR_med_train[0, kk])


                #jini
                v_fXhat_B = net1_jini.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_perfectCSI[4, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_perfectCSI[4, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                Xtrain_symbol = path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y

                net1_jini.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                #jini_p%
                v_fXhat_B = net1_jini_p.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_perfectCSI[5, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_perfectCSI[5, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                pp = path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]###
                Xtrain_symbol = pp[(aaa[1, :] >= threshold1) & (aaa[1, :] < threshold2)]
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, (aaa[1, :] >= threshold1) & (aaa[1, :] < threshold2)]

                net1_jini_p.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)


            #-----ViterbiNet uncertainty CSI-----#
            if v_nCurves[1] == 1:

                #training - high score training
                v_fXhat_B = net2_train.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_uncertaintyCSI[3, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_uncertaintyCSI[3, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold = sort_price[int((1-track_DataSize_per) * s_fTrainBlkSize)]
                Xtrain_symbol = v_fXhat_B[
                    2, v_fXhat_B[1,
                       :] >= threshold]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, v_fXhat_B[1, :] >= threshold]

                net2_train.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                # no training
                v_fXhat_B = net2_NoTrain.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_uncertaintyCSI[1, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_uncertaintyCSI[1, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                #bad training - low score training
                v_fXhat_B = net2_BadTrain.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_uncertaintyCSI[2, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_uncertaintyCSI[2, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold = sort_price[int(track_DataSize_per * s_fTrainBlkSize)]
                Xtrain_symbol = v_fXhat_B[
                    2, v_fXhat_B[1,
                       :] < threshold]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, v_fXhat_B[1, :] < threshold]

                net2_BadTrain.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                # Median training
                v_fXhat_B = net2_MedTrain.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_uncertaintyCSI[0, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_uncertaintyCSI[0, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold1 = sort_price[int((med_point - track_DataSize_per / 2) * s_fTrainBlkSize)]
                threshold2 = sort_price[int((med_point + track_DataSize_per / 2) * s_fTrainBlkSize)]
                aaa = v_fXhat_B

                Xtrain_symbol = v_fXhat_B[
                    2, (v_fXhat_B[1, :] >= threshold1) & (v_fXhat_B[1,
                                                          :] < threshold2)]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, (v_fXhat_B[1, :] >= threshold1) & (v_fXhat_B[1, :] < threshold2)]

                net2_MedTrain.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                # jini
                v_fXhat_B = net2_jini.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_uncertaintyCSI[4, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_uncertaintyCSI[4, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                Xtrain_symbol = path_test_B[
                    0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y

                net2_jini.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

                # jini_p%
                v_fXhat_B = net2_jini_p.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_uncertaintyCSI[5, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_uncertaintyCSI[5, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[
                        0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                pp = path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))]  ###
                Xtrain_symbol = pp[(aaa[1, :] >= threshold1) & (aaa[1, :] < threshold2)]
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, (aaa[1, :] >= threshold1) & (aaa[1, :] < threshold2)]

                net2_jini_p.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)


            #-----full_CSI-----#
            if v_nCurves[2] == 1:
                m_fLikelihood = np.array(np.zeros((s_fTrainBlkSize, s_nStates)))
                # Compute conditional PDF for each state
                for ii in range(s_nStates):
                    v_fX = np.zeros((s_nMemSize, 1))
                    Idx = ii
                    for ll in range(s_nMemSize):
                        v_fX[ll] = Idx % s_nConst + 1
                        Idx = np.floor(Idx / s_nConst)
                    v_fS = 2 * (v_fX - 0.5 * (s_nConst + 1))
                    m_fLikelihood[:, ii] = stats.norm.pdf(
                        Y - np.fliplr(np.reshape(m_fChannel[int(3 * kk), :], (1, len(m_fChannel[int(3 * kk), :])))
                                      ).dot(v_fS), 0, s_fSigmaW)
                # Apply Viterbi detection based on computed likelihoods
                v_fXhat_B = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize)

                track_SER_fullCSI[0, kk, mm] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_fullCSI[0, kk, mm] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])


        track_SER_avg[0, mm] = np.mean(track_SER_perfectCSI[0, :, mm])
        track_SER_avg[1, mm] = np.mean(track_SER_perfectCSI[1, :, mm])
        track_SER_avg[2, mm] = np.mean(track_SER_perfectCSI[2, :, mm])
        track_SER_avg[3, mm] = np.mean(track_SER_perfectCSI[3, :, mm])
        track_SER_avg[4, mm] = np.mean(track_SER_perfectCSI[4, :, mm])
        track_SER_avg[5, mm] = np.mean(track_SER_perfectCSI[5, :, mm])
        track_SER_avg[6, mm] = np.mean(track_SER_uncertaintyCSI[0, :, mm])
        track_SER_avg[7, mm] = np.mean(track_SER_uncertaintyCSI[1, :, mm])
        track_SER_avg[8, mm] = np.mean(track_SER_uncertaintyCSI[2, :, mm])
        track_SER_avg[9, mm] = np.mean(track_SER_uncertaintyCSI[3, :, mm])
        track_SER_avg[10, mm] = np.mean(track_SER_uncertaintyCSI[4, :, mm])
        track_SER_avg[11, mm] = np.mean(track_SER_uncertaintyCSI[5, :, mm])
        track_SER_avg[12, mm] = np.mean(track_SER_fullCSI[0, :, mm])

        track_PER_avg[0, mm] = np.mean(track_PER_perfectCSI[0, :, mm])
        track_PER_avg[1, mm] = np.mean(track_PER_perfectCSI[1, :, mm])
        track_PER_avg[2, mm] = np.mean(track_PER_perfectCSI[2, :, mm])
        track_PER_avg[3, mm] = np.mean(track_PER_perfectCSI[3, :, mm])
        track_PER_avg[4, mm] = np.mean(track_PER_perfectCSI[4, :, mm])
        track_PER_avg[5, mm] = np.mean(track_PER_perfectCSI[5, :, mm])
        track_PER_avg[6, mm] = np.mean(track_PER_uncertaintyCSI[0, :, mm])
        track_PER_avg[7, mm] = np.mean(track_PER_uncertaintyCSI[1, :, mm])
        track_PER_avg[8, mm] = np.mean(track_PER_uncertaintyCSI[2, :, mm])
        track_PER_avg[9, mm] = np.mean(track_PER_uncertaintyCSI[3, :, mm])
        track_PER_avg[10, mm] = np.mean(track_PER_uncertaintyCSI[4, :, mm])
        track_PER_avg[11, mm] = np.mean(track_PER_uncertaintyCSI[5, :, mm])
        track_PER_avg[12, mm] = np.mean(track_PER_fullCSI[0, :, mm])
        ####################################



    m_fSERAvg = m_fSERAvg + m_fSER[:, :, eIdx]

    # Dispaly exponent index
    print(eIdx)

m_fSERAvg = m_fSERAvg / np.size(v_fExps)

print('')




#---------------Display Results---------------#

#success rate
# success_rate_all(d_symbol1, d_symbol2, d_path1, d_path2)
success_rate_plot(d_path1)


#ViterbiNet
ViterbiNet_plot(v_fSigWdB, m_fSERAvg)


#channel tracking simulation results (median point chosen to be optimal for SNR=6dB):
#channel coefficients
channel_taps_plot(m_fChannel)

#moving average of SER
dB_wanted = 6  #chosen SNR results [-6,-4,-2,0,2,4,6,8,10]
track_res_plot(track_SER_fullCSI, track_SER_perfectCSI, track_SER_uncertaintyCSI, v_fSigWdB, dB_wanted)

#AVG display - for each SNR
AVGtrack_plot(track_SER_avg, v_fSigWdB)


#diversity MSE
diversity_plot(dict_train, dict_bad_train, dict_med_train,
                   MSE_diversity_train, MSE_diversity_bad_train, MSE_diversity_med_train)
#noise power & error
NoisePower_error(NOISE_train, NOISE_bad_train, NOISE_med_train, ERR_train, ERR_bad_train, ERR_med_train)

#---------------------------------------------#
