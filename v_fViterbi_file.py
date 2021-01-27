import numpy as np

def v_fViterbi(m_fPriors, s_nConst, s_nMemSize):

    """
    Apply Viterbi detection from computed priors

    Syntax
    -------------------------------------------------------
    v_fXhat = v_fViterbi(m_fPriors, s_nConst, s_nMemSize)

    INPUT:
    -------------------------------------------------------
    m_fPriors - evaluated likelihoods for each state at each time instance
    s_nConst - constellation size (positive integer)
    s_nMemSize - channel memory length


    OUTPUT:
    -------------------------------------------------------
    v_fXhat - recovered symbols vector
    """


    s_nDataSize = np.size(m_fPriors, 0)
    s_nStates = s_nConst**s_nMemSize
    v_fXhat = np.zeros((1, s_nDataSize))

    # Generate trellis matrix
    m_fTrellis = np.zeros((s_nStates, s_nConst))
    for ii in range(s_nStates):
        Idx = ii % (s_nConst**(s_nMemSize-1))
        for ll in range(s_nConst):
            m_fTrellis[ii, ll] = s_nConst*Idx + ll + 1


    #Apply Viterbi
    m_fCost = -np.log(m_fPriors)


    #-----loop mode calculation-----#
    # v_fCtilde = np.zeros((s_nStates, 1))
    #
    # for kk in range(s_nDataSize):
    #     m_fCtildeNext = np.zeros((s_nStates, 1))
    #     for ii in range(s_nStates):
    #         v_fTemp = np.zeros((s_nConst, 1))
    #         for ll in range(s_nConst):
    #             v_fTemp[ll] = v_fCtilde[(int(m_fTrellis[ii, ll]))-1] + m_fCost[kk, ii]
    #         m_fCtildeNext[ii] = np.min(v_fTemp)
    #     v_fCtilde = m_fCtildeNext
    #     I = np.argmin(v_fCtilde)
    #     # return index of first symbol in current state
    #     v_fXhat[0, kk] = I % s_nConst + 1


    # -----less loop mode calculation-----#
    # v_fCtilde = np.zeros((s_nStates, 1))
    #
    # for kk in range(s_nDataSize): # ll
    #     m_fCtildeNext = np.zeros((s_nStates, 1))
    #     for ii in range(s_nStates):
    #         v_fTemp = np.zeros((s_nConst, 1))
    #     #    for ll in range(s_nConst):
    #         v_fTemp[0::s_nConst-1] = v_fCtilde[m_fTrellis[ii, 0::s_nConst-1].astype(int)-1] + m_fCost[kk, ii]
    #         m_fCtildeNext[ii] = np.min(v_fTemp)
    #     v_fCtilde = m_fCtildeNext
    #     I = np.argmin(v_fCtilde)
    #     # return index of first symbol in current state
    #     v_fXhat[0, kk] = I % s_nConst + 1


    # -----matrix mode calculation-----#
    v_fCtilde = np.zeros((s_nStates, 1))

    for kk in range(s_nDataSize):
        m_fCtildeNext = np.zeros((s_nStates, 1))
        #for ii in range(s_nStates):
        v_fTemp = np.array(np.zeros((s_nStates, s_nConst)))
        #    for ll in range(s_nConst):
        v_fTemp = np.transpose(np.vstack((np.transpose(np.vstack((v_fCtilde[0:s_nStates:s_nConst], v_fCtilde[0:s_nStates:s_nConst]))) + np.array([m_fCost[kk, 0:s_nStates]]),
                                          np.transpose(np.vstack((v_fCtilde[1:s_nStates:s_nConst], v_fCtilde[1:s_nStates:s_nConst]))) + np.array([m_fCost[kk, 0:s_nStates]]))))
        m_fCtildeNext = m_fCtildeNext = np.transpose(np.array([v_fTemp.min(axis=1)]))
        v_fCtilde = m_fCtildeNext
        I = np.argmin(v_fCtilde)
        # return index of first symbol in current state
        v_fXhat[0, kk] = I % s_nConst + 1

    print('--------------------viterbi algorithm done--------------------')

    return v_fXhat
