# ViterbiNet example code - ISI channel with AWGN

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from GetViterbiNet_file import GetViterbiNet
#import TrainViterbiNet
from ApplyViterbiNet_file import ApplyViterbiNet
from v_fViterbi_file import v_fViterbi
from m_fMyReshape_file import m_fMyReshape

np.random.seed(9001)

#----------Parameters Setting----------#
s_nConst = 2         # Constellation size (2 = BPSK)
s_nMemSize = 4       # Number of taps
s_fTrainSize = 5000  # Training size
s_fTestSize = 50000  # Test data size

s_nStates = s_nConst**s_nMemSize

v_fSigWdB = np.array([np.arange(-6, 11, 2)]) #  np.array([np.arange(-6, 11, 2)])    # Noise variance in dB

s_fEstErrVar = 0.1   # Estimation error variance
# Frame size for generating noisy training
s_fFrameSize = 500
s_fNumFrames = s_fTrainSize/s_fFrameSize

v_nCurves = [           # Curves
    1,                    # Deep Viterbi - perfect CSI
    1,                    # Deep Viterbi - CSI uncertainty
    1                     # Viterbi algorithm
    ]


s_nCurves = np.size(v_nCurves)

v_stProts = (
    'ViterbiNet, perfect CSI',
    'ViterbiNet, CSI uncertainty',
    'Viterbi algorithm')

s_nMixtureSize = s_nStates


#----------Simulation Loop----------#
# v_fExps = np.array([np.arange(1, 2, 1)])  #np.array([np.arange(0.1, 2, 0.1)])  #np.ones((1, 1))
v_fExps = np.array([[0.5, 1, 1.5]])
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
        Idxs = np.arange((kk * s_fFrameSize), (kk+1) * s_fFrameSize)
        v_Rtrain2[0, Idxs] = np.fliplr(v_fChannel + np.sqrt(s_fEstErrVar) * np.dot(np.array([np.random.randn(np.size(v_fChannel))]),
                                                                                   np.diag(v_fChannel[0, :]))).dot(m_fStrain[:, Idxs])

    # Generate test labels
    v_fXtest = np.array([np.random.randint(1, 3, s_fTestSize)])
    v_fStest = 2 * (v_fXtest - 0.5 * (s_nConst + 1))
    m_fStest = m_fMyReshape(v_fStest, s_nMemSize)
    v_Rtest = np.dot(np.fliplr(v_fChannel), m_fStest)

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
            # Apply ViterbiNet detctor
            # v_fXhat1 = ApplyViterbiNet(v_fYtest, net1, s_nConst, s_nMemSize)
            v_fXhat1 = net1.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
            # Evaluate error rate
            m_fSER[0, mm, eIdx] = np.mean(v_fXhat1 != v_fXtest)
            
            
        # Viterbi net - CSI uncertainty
        if v_nCurves[1] == 1:
            # Train network using training with uncertainty
            net2 = GetViterbiNet(v_fXtrain, v_fYtrain2, s_nConst, s_nMemSize)
            # Apply ViterbiNet detctor
            # v_fXhat2 = ApplyViterbiNet(v_fYtest, net2, s_nConst, s_nMemSize)
            v_fXhat2 = net2.ApplyViterbiNet(v_fYtest, s_nConst, s_nMemSize)
            # Evaluate error rate
            m_fSER[1, mm, eIdx] = np.mean(v_fXhat2 != v_fXtest)


        # Model-based Viterbi algorithm
        if v_nCurves[2] == 1:
            m_fLikelihood = np.array(np.zeros((s_fTestSize, s_nStates)))
            # Compute conditional PDF for each state
            for ii in range(s_nStates):
                v_fX = np.zeros((s_nMemSize, 1))
                Idx = ii
                for ll in range(s_nMemSize):
                    v_fX[ll] = Idx % s_nConst + 1
                    Idx = np.floor(Idx/s_nConst)
                v_fS = 2*(v_fX - 0.5*(s_nConst+1))
                m_fLikelihood[:, ii] = stats.norm.pdf(v_fYtest - np.fliplr(v_fChannel).dot(v_fS), 0, s_fSigmaW)
            # Apply Viterbi detection based on computed likelihoods
            v_fXhat3 = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize)
            # Evaluate error rate
            m_fSER[2, mm, eIdx] = np.mean(v_fXhat3 != v_fXtest)

        # Display SNR index
        print(mm)
        print(m_fSER[:, :, eIdx])

    m_fSERAvg = m_fSERAvg + m_fSER[:, :, eIdx]

    # Dispaly exponent index
    print(eIdx)

m_fSERAvg = m_fSERAvg / np.size(v_fExps)




#----------Display Results----------#
plt.figure()
plt.semilogy(np.transpose(v_fSigWdB), m_fSERAvg[0, :], 'ro--',
             np.transpose(v_fSigWdB), m_fSERAvg[1, :], 'go--',
             np.transpose(v_fSigWdB), m_fSERAvg[2, :], 'bo--')
plt.legend(('ViterbiNet - perfect CSI', 'ViterbiNet - CSI uncertainty', 'Viterbi algorithm'))
plt.title('SER - Symbol Error Rate\n(Learn Rate=0.00005, maxEpochs=50, miniBatchSize=25)\n(NN=1x75x16)')
plt.xlabel('SNR [dB]')
plt.ylabel('SER')
plt.grid(True, which="both", ls="-")
plt.show()
print('')
