# ViterbiNet example code - ISI channel with AWGN

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from GetViterbiNet_file import GetViterbiNet
# import TrainViterbiNet
from ApplyViterbiNet_file import ApplyViterbiNet
from v_fViterbi_file import v_fViterbi
from m_fMyReshape_file import m_fMyReshape
from SOVA_path_and_SER import *


# def grade_AVG(x1, x2, x3):
#     X = np.zeros((3, 50000))
#     X[0, :] = np.rint((x1[0, :]+x2[0, :]+x3[0, :])/3)
#     X[1, :] = (x1[1, :]+x2[1, :]+x3[1, :])/3
#     X[2, :] = path(np.reshape(X[0, :]-1, newshape=(1, 50000)))
#     return X

def Xhat_viterbinet(net1, net2, net3, net4, v_fYtest, s_nConst, s_nMemSize):
    likelihood1 = net1.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    likelihood2 = net2.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    likelihood3 = net3.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    likelihood4 = net4.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
    ensamble_likelihood = (likelihood1 + likelihood2 + likelihood3 + likelihood4) / 4
    Xhat = v_fViterbi(ensamble_likelihood, s_nConst, s_nMemSize)
    return Xhat


# np.random.seed(9001)

# ----------Parameters Setting----------#
s_nConst = 2  # Constellation size (2 = BPSK)
s_nMemSize = 4  # Number of taps
s_fTrainSize = 5000  # Training size
s_fTestSize = 50000  # Test data size

s_nStates = s_nConst ** s_nMemSize

v_fSigWdB = np.array([np.arange(-6, 13, 4)]) # np.array([[6]])    # Noise variance in dB

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

s_nMixtureSize = s_nStates

# ----------Simulation Loop----------#
# v_fExps = np.array([np.arange(1, 2, 1)])  #np.array([np.arange(0.1, 2, 0.1)])  #np.ones((1, 1))
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

    d_path1 = np.zeros((np.size(v_fSigWdB), 4))
    d_symbol1 = np.zeros((np.size(v_fSigWdB), 4))
    d_path2 = np.zeros((np.size(v_fSigWdB), 4))
    d_symbol2 = np.zeros((np.size(v_fSigWdB), 4))

    track_SER_avg = np.zeros((3, np.size(v_fSigWdB)))
    track_PER_avg = np.zeros((3, np.size(v_fSigWdB)))

    track_SER = np.zeros((3, 25, np.size(v_fSigWdB)))
    track_PER = np.zeros((3, 25, np.size(v_fSigWdB)))


    # Loop over number of SNR
    for mm in range(np.size(v_fSigWdB)):
        s_fSigmaW = 10 ** (-0.1 * (v_fSigWdB[0, mm]))
        # LTI AWGN channel
        v_fYtrain = v_Rtrain + np.sqrt(s_fSigmaW) * np.random.randn(np.size(v_Rtrain))
        v_fYtrain2 = v_Rtrain2 + np.sqrt(s_fSigmaW) * np.random.randn(np.size(v_Rtrain))
        v_fYtest = v_Rtest + np.sqrt(s_fSigmaW) * np.random.randn(np.size(v_Rtest))

        #############################
        # f = open('Data.txt', 'r')
        # for i in range(5000):
        #     v_fXtrain[0, i] = float(f.readline())
        # for i in range(50000):
        #     v_fXtest[0, i] = float(f.readline())
        # for i in range(5000):
        #     v_fYtrain[0, i] = float(f.readline())
        # for i in range(5000):
        #     v_fYtrain2[0, i] = float(f.readline())
        # for i in range(50000):
        #     v_fYtest[0, i] = float(f.readline())
        #############################

        # Viterbi net - perfect CSI
        if v_nCurves[0] == 1:
            # Train network
            net1 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)
            # -----ensamble-----#
            # net1_1 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # net1_2 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # net1_3 = GetViterbiNet(v_fXtrain, v_fYtrain, s_nConst, s_nMemSize)  # ensamble
            # v_fXhat1 = Xhat_viterbinet(net1, net1_1, net1_2, net1_3, v_fYtest, s_nConst, s_nMemSize)

            # Apply ViterbiNet detctor
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

            # Apply ViterbiNet detctor
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



        # ----------Channel Tracking ViterbiNet----------#
        s_fTrainBlkSize = 2000
        s_fCompositeBlocks = 25

        track_m_fSER_B = np.zeros((1, 1, s_fCompositeBlocks))
        track_m_fPER_B = np.zeros((1, 1, s_fCompositeBlocks))
        track_SER_perfectCSI = np.zeros((1, 1, s_fCompositeBlocks))
        track_PER_perfectCSI = np.zeros((1, 1, s_fCompositeBlocks))
        track_SER_uncertaintyCSI = np.zeros((1, 1, s_fCompositeBlocks))
        track_PER_uncertaintyCSI = np.zeros((1, 1, s_fCompositeBlocks))
        track_SER_fullCSI = np.zeros((1, 1, s_fCompositeBlocks))
        track_PER_fullCSI = np.zeros((1, 1, s_fCompositeBlocks))

        v_fXtrain2 = np.array(np.random.randint(low=1, high=3, size=(1, s_fTrainBlkSize * s_fCompositeBlocks)))
        v_fStrain = 2 * (v_fXtrain2 - 0.5 * (s_nConst + 1))

        v_nPers = 3 * np.array((17, 13, 11, 7))
        m_fTimeVar = 0.8 + 0.2 * np.transpose(
            np.cos(np.reshape((2 * np.pi / v_nPers), (4, 1)).dot(np.reshape(np.array(range(200)), (1, 200)))))
        m_fChannel = np.multiply(m_fTimeVar, v_fChannel)

        m_fRtrain2 = np.zeros((s_fCompositeBlocks, s_fTrainBlkSize))
        path_test_B = path(v_fXtrain2 - 1)

        for kk in range(s_fCompositeBlocks):
            print(kk)
            m_fStrain2 = m_fMyReshape(v_fStrain[:, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))],
                                      s_nMemSize)
            m_fRtrain2[kk, :] = np.fliplr(np.reshape(m_fChannel[int(3 * kk), :], (1, 4))).dot(m_fStrain2)
            Y = np.reshape(m_fRtrain2[kk, :], (1, s_fTrainBlkSize)) + np.sqrt(s_fSigmaW) * np.random.randn(
                np.size(m_fRtrain2[kk, :]))

            # -----ViterbiNet perfect CSI-----#
            if v_nCurves[0] == 1:
                v_fXhat_B = net1.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_perfectCSI[eIdx, 0, kk] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_perfectCSI[eIdx, 0, kk] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold = sort_price[int(0.8 * s_fTrainBlkSize)]
                Xtrain_symbol = v_fXhat_B[
                    2, v_fXhat_B[1,
                       :] >= threshold]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, v_fXhat_B[1, :] >= threshold]

                # net1.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

            # -----ViterbiNet uncertainty CSI-----#
            if v_nCurves[2] == 1:
                v_fXhat_B = net2.ApplyViterbiNet(np.reshape(Y, (1, s_fTrainBlkSize)), s_nConst, s_nMemSize)

                track_SER_uncertaintyCSI[eIdx, 0, kk] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_uncertaintyCSI[eIdx, 0, kk] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

                decode_symbol = np.reshape(v_fXhat_B[0, :], (1, len(v_fXhat_B[0, :])))
                sort_price = np.sort(v_fXhat_B[1, :])
                threshold = sort_price[int(0.8 * s_fTrainBlkSize)]
                Xtrain_symbol = v_fXhat_B[
                    2, v_fXhat_B[1,
                       :] >= threshold]  # m_fMyReshape(decode_symbol[:, v_fXhat_B[1, :] >= threshold], s_nMemSize)#
                Xtrain_symbol = np.reshape(Xtrain_symbol, newshape=(1, np.size(Xtrain_symbol)))
                Ytrain_symbol = Y[:, v_fXhat_B[1, :] >= threshold]

                # net2.TrainViterbiNet(Xtrain_symbol, Ytrain_symbol, s_nConst, 0.00005)

            # -----full_CSI-----#
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

                track_SER_fullCSI[eIdx, 0, kk] = np.mean(
                    v_fXhat_B[0, :] != v_fXtrain2[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])
                track_PER_fullCSI[eIdx, 0, kk] = np.mean(
                    v_fXhat_B[2, :] != path_test_B[0, range(int(kk * s_fTrainBlkSize), int(((kk + 1) * s_fTrainBlkSize)))])

        track_SER[0, :, mm] = track_SER_perfectCSI
        track_SER[1, :, mm] = track_SER_uncertaintyCSI
        track_SER[2, :, mm] = track_SER_fullCSI
        track_PER[0, :, mm] = track_PER_perfectCSI
        track_PER[1, :, mm] = track_PER_uncertaintyCSI
        track_PER[2, :, mm] = track_PER_fullCSI


        # track_SER_avg = np.zeros((3, np.size(v_fSigWdB)))
        # track_PER_avg = np.zeros((3, np.size(v_fSigWdB)))

        track_SER_avg[0, mm] = np.mean(track_SER_perfectCSI)
        track_SER_avg[1, mm] = np.mean(track_SER_uncertaintyCSI)
        track_SER_avg[2, mm] = np.mean(track_SER_fullCSI)

        track_PER_avg[0, mm] = np.mean(track_PER_perfectCSI)
        track_PER_avg[1, mm] = np.mean(track_PER_uncertaintyCSI)
        track_PER_avg[2, mm] = np.mean(track_PER_fullCSI)
        ####################################



    m_fSERAvg = m_fSERAvg + m_fSER[:, :, eIdx]

    # Dispaly exponent index
    print(eIdx)

m_fSERAvg = m_fSERAvg / np.size(v_fExps)





# ----------Display Results----------#
# d_symbol1_ = np.array([d_symbol1[3, :], d_symbol1[4, :], d_symbol1[5, :], d_symbol1[6, :], d_symbol1[7, :]])
# d_symbol2_ = np.array([d_symbol2[3, :], d_symbol2[4, :], d_symbol2[5, :], d_symbol2[6, :], d_symbol2[7, :]])
# d_path1_ = np.array([d_path1[3, :], d_path1[4, :], d_path1[5, :], d_path1[6, :], d_path1[7, :]])
# d_path2_ = np.array([d_path2[3, :], d_path2[4, :], d_path2[5, :], d_path2[6, :], d_path2[7, :]])

# diagram_plot(d_symbol1_, d_path1_)
# diagram_plot(d_symbol2_, d_path2_)

plt.figure()
plt.semilogy(np.transpose(v_fSigWdB), m_fSERAvg[0, :], 'ro--',
             np.transpose(v_fSigWdB), m_fSERAvg[1, :], 'go--',
             np.transpose(v_fSigWdB), m_fSERAvg[2, :], 'bo--')
plt.legend(('ViterbiNet - perfect CSI', 'ViterbiNet - CSI uncertainty', 'Viterbi '
                                                                        'algorithm'))
plt.title('SER - Symbol Error Rate\n(Learn Rate=0.00005, maxEpochs=50, miniBatchSize=25)\n(NN=1x75x16)')
plt.xlabel('SNR [dB]')
plt.ylabel('SER')
plt.grid(True, which="both", ls="-")

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.semilogy(np.arange(25), track_SER[0, :, 1], 'ro--',
             np.arange(25), track_SER[1, :, 1], 'go--',
             np.arange(25), track_SER[2, :, 1], 'bo--')
ax1.legend(('ViterbiNet - perfect CSI', 'ViterbiNet - CSI uncertainty', 'Viterbi algorithm'))
ax1.set_title('SER - Symbol Error Rate\nChannel Tracking')
ax1.set(xlabel='t [sec]', ylabel='SER')
# ax1.xlabel('t [sec]')
# ax1.ylabel('SER')
# ax1.grid(True, which="both", ls="-")
# ax1.show()

# plt.figure()
ax2.semilogy(np.arange(25), track_PER[0, :, 1], 'ro--',
             np.arange(25), track_PER[1, :, 1], 'go--',
             np.arange(25), track_PER[2, :, 1], 'bo--')
ax2.legend(('ViterbiNet - perfect CSI', 'ViterbiNet - CSI uncertainty', 'Viterbi algorithm'))
ax2.set_title('PER - Path Error Rate\nChannel Tracking')
ax2.set(xlabel='t [sec]', ylabel='PER')
# ax2.xlabel('t [sec]')
# ax2.ylabel('PER')
# ax2.grid(True, which="both", ls="-")
# ax2.show()


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(np.transpose(v_fSigWdB), track_SER_avg[0, :], 'ro--',
             np.transpose(v_fSigWdB), track_SER_avg[1, :], 'go--',
             np.transpose(v_fSigWdB), track_SER_avg[2, :], 'bo--')
ax1.legend(('ViterbiNet - perfect CSI', 'ViterbiNet - CSI uncertainty', 'Viterbi algorithm'))
ax1.set_title('SER - Symbol Error Rate\nChannel Tracking')
ax1.set(xlabel='SNR', ylabel='SER')
# ax1.xlabel('t [sec]')
# ax1.ylabel('SER')
# ax1.grid(True, which="both", ls="-")
ax1.grid()
# ax1.show()

# plt.figure()
ax2.plot(np.transpose(v_fSigWdB), track_PER_avg[0, :], 'ro--',
             np.transpose(v_fSigWdB), track_PER_avg[1, :], 'go--',
             np.transpose(v_fSigWdB), track_PER_avg[2, :], 'bo--')
ax2.legend(('ViterbiNet - perfect CSI', 'ViterbiNet - CSI uncertainty', 'Viterbi algorithm'))
ax2.set_title('PER - Path Error Rate\nChannel Tracking')
ax2.set(xlabel='SNR', ylabel='PER')
# ax2.xlabel('t [sec]')
# ax2.ylabel('PER')
# ax2.grid(True, which="both", ls="-")
ax2.grid()
# ax2.show()
print('')
