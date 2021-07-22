import numpy as np

def m_fMyReshape(v_fVec, s_nRows):

    """
    Reshape vector into matrix form with interleaved columns

    Syntax
    -------------------------------------------------------
    m_fMat = m_fMyReshape(v_fVec, s_nRows)

    INPUT:
    -------------------------------------------------------
    v_fVec - vector to reshape s_nRows - number of rows in matrix represetnation

    OUTPUT:
    -------------------------------------------------------
    m_fMat - matrix represetnation
    """


    s_nCols = np.size(v_fVec)

    m_fMat = np.ones((s_nRows, s_nCols))

    for kk in range(s_nRows):
        ll = s_nRows - kk - 1
        m_fMat[ll, 0:s_nCols-ll-1] = v_fVec[0, ll: s_nCols-1]

    return m_fMat
