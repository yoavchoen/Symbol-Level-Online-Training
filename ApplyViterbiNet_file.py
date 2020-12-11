from v_fViterbi_file import v_fViterbi
import torch
import numpy as np

def ApplyViterbiNet(v_fY, net, s_nConst, s_nMemSize):
    """
    # Apply ViterbiNet to observed channel outputs
    #
    # Syntax
    # -------------------------------------------------------
    # v_fXhat = ApplyViterbiNet(v_fY, net, GMModel, s_nConst)
    #
    # INPUT:
    # -------------------------------------------------------
    # v_fY - channel output vector
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
    s_nStates = s_nConst**s_nMemSize
    # Use network to compute likelihood function
    # v_fYcat = np.transpose(v_fY)
    v_fYcat = v_fY
    v_fYcat = torch.from_numpy(np.reshape(v_fYcat, newshape=(np.size(v_fYcat, 0), np.size(v_fYcat, 1), 1)))
    m_fpS_Y = net.forward(v_fYcat.float())
    # Compute likelihoods
    m_fLikelihood = m_fpS_Y
    # Apply Viterbi output layer
    m_fLikelihood = np.reshape(m_fLikelihood.detach().numpy(), newshape=(50000, 16))
    v_fXhat = v_fViterbi(m_fLikelihood, s_nConst, s_nMemSize)

    return v_fXhat
