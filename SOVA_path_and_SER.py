import numpy as np
import matplotlib.pyplot as plt

def path(x):
    # return the path (0-15) of binary sequence
    s = np.size(x)
    path = np.zeros((1, s))
    for i in range(0, s-3):
        path[0, i] = x[0, i] + x[0, i+1]*2 + x[0, i+2]*4 + x[0, i+3]*8
    return path


def sova_corelation(v_fXtest, v_fXhat):
    # return SER and PER for each datasize

    #-----PATH-----#
    p = path(v_fXtest - 1)
    eq = (p == v_fXhat[2, :])
    s = np.sum(eq)
    avg = s / 50000
    eq_grade = eq * v_fXhat[1, :]
    eq_grade_avg = np.sum(eq_grade) / s

    w = np.sum(v_fXhat[1, :] * eq) / np.sum(eq)
    l = np.sum(v_fXhat[1, :] * (1 - eq)) / np.sum(1 - eq)

    threshold = np.linspace(min(v_fXhat[1, :]), max(v_fXhat[1, :]), 100)
    datasize_and_err_path = np.zeros((2, 100))
    for i in range(0, 100):
        thresh = (v_fXhat[1, :] >= threshold[i])
        sum_thresh = np.sum(thresh)
        datasize_and_err_path[0, i] = sum_thresh / 50000

        a = thresh * p
        b = thresh * v_fXhat[2, :]
        datasize_and_err_path[1, i] = np.sum((a != b) * (a != 0)) / sum_thresh


    #-----symbol-----#
    w = np.sum(v_fXhat[1, :] * (v_fXhat[0, :] == v_fXtest)) / np.sum((v_fXhat[0, :] == v_fXtest))
    l = np.sum(v_fXhat[1, :] * (v_fXhat[0, :] != v_fXtest)) / np.sum((v_fXhat[0, :] != v_fXtest))

    threshold = np.linspace(min(v_fXhat[1, :]), max(v_fXhat[1, :]), 100)
    datasize_and_err_symbol = np.zeros((2, 100))
    for i in range(0, 100):
        thresh = (v_fXhat[1, :] >= threshold[i])
        sum_thresh = np.sum(thresh)
        datasize_and_err_symbol[0, i] = sum_thresh / 50000

        a = thresh * v_fXtest
        b = thresh * v_fXhat[0, :]
        datasize_and_err_symbol[1, i] = np.sum((a != b) * (a != 0)) / sum_thresh

    return datasize_and_err_path, datasize_and_err_symbol


def get_d(datasize_and_err):
    # get path and symbol error - 100,50,30,10 [%] of received data
    per = [1, 0.5, 0.3, 0.1]
    index = np.zeros((1, 4))
    for i in np.arange(0, 4):
        t = np.where(datasize_and_err[0, :] < per[i])
        temp = np.asarray(t)
        index[0, i] = temp[0, 0]
    d = np.array((datasize_and_err[1, int(index[0, 0] - 1)],
                               datasize_and_err[1, int(index[0, 1] - 1)],
                               datasize_and_err[1, int(index[0, 2] - 1)],
                               datasize_and_err[1, int(index[0, 3] - 1)]))
    return d


def diagram_plot(d_symbol, d_path):
    """
    plot columns diagram of SER scores and PATH scores
    for SNR = -4, 0, 4, 6, 8 [dB]
    and 10,30,50,100 [%] from data size
    (scores = right detection [%])
    """

    #-----symbol columns plot-----#
    n_groups = 5

    d_100 = 100 * (1 - d_symbol[:, 0])
    d_50 = 100 * (1 - d_symbol[:, 1])
    d_30 = 100 * (1 - d_symbol[:, 2])
    d_10 = 100 * (1 - d_symbol[:, 3])

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8

    rects1 = plt.bar(index, d_100, bar_width,
                     alpha=opacity,
                     color='b',
                     label='100% of received data')

    rects2 = plt.bar(index + bar_width, d_50, bar_width,
                     alpha=opacity,
                     color='g',
                     label='top 50% SOVA score')

    rects3 = plt.bar(index + 2*bar_width, d_30, bar_width,
                     alpha=opacity,
                     color='r',
                     label='top 30% SOVA score')

    rects4 = plt.bar(index + 3*bar_width, d_10, bar_width,
                     alpha=opacity,
                     color='y',
                     label='top 10% SOVA score')

    plt.xlabel('SNR')
    plt.ylabel('success rate')
    plt.title('success rate (symbol) by SNR')
    plt.xticks(index + 1.5*bar_width, ('0dB', '2dB', '4dB', '6dB', '8dB'))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()



    # -----PATH columns plot-----#
    n_groups = 5

    d_100 = 100*(1-d_path[:, 0])
    d_50 = 100*(1-d_path[:, 1])
    d_30 = 100*(1-d_path[:, 2])
    d_10 = 100*(1-d_path[:, 3])

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8

    rects1 = plt.bar(index, d_100, bar_width,
                     alpha=opacity,
                     color='b',
                     label='100% of received data')

    rects2 = plt.bar(index + bar_width, d_50, bar_width,
                     alpha=opacity,
                     color='g',
                     label='top 50% SOVA score')

    rects3 = plt.bar(index + 2 * bar_width, d_30, bar_width,
                     alpha=opacity,
                     color='r',
                     label='top 30% SOVA score')

    rects4 = plt.bar(index + 3 * bar_width, d_10, bar_width,
                     alpha=opacity,
                     color='y',
                     label='top 10% SOVA score')

    plt.xlabel('SNR')
    plt.ylabel('success rate')
    plt.title('success rate (path) by SNR')
    plt.xticks(index + 1.5*bar_width, ('0dB', '2dB', '4dB', '6dB', '8dB'))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
