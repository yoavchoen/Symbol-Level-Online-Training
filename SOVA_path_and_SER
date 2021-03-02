import numpy as np
import matplotlib.pyplot as plt

def path(x):
    # return the path (0-15) of binary sequence
    path = np.zeros((1, 50000))
    for i in range(0, 50000-3):
        path[0, i] = x[0, i] + x[0, i+1]*2 + x[0, i+2]*4 + x[0, i+3]*8
    return path

def sova_corelation(v_fXtest, v_fXhat):
    # get and plot path and symbol performance by SOVA
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

    # plt.figure()
    # plt.plot(threshold, datasize_and_err_path[0, :], marker='*')
    # plt.title('data size [%] - PATH')
    # plt.xlabel("Threshold")
    # plt.ylabel('% of symbols')
    #
    # plt.figure()
    # plt.plot(threshold, datasize_and_err_path[1, :], marker='*')
    # plt.title('PATH error')
    # plt.xlabel("Threshold")
    # plt.ylabel('PATH error')

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

    # plt.figure()
    # plt.plot(threshold, datasize_and_err_symbol[0, :], marker='*')
    # plt.title('data size [%] - symbol')
    # plt.xlabel("Threshold")
    # plt.ylabel('% of symbols')
    #
    # plt.figure()
    # plt.plot(threshold, datasize_and_err_symbol[1, :], marker='*')
    # plt.title('SER')
    # plt.xlabel("Threshold")
    # plt.ylabel('SER')

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
    # data to plot
    n_groups = 5
    # d_100 = (72.246, 82.238, 93.664, 97.794, 99.430)
    # d_50 = (76.624, 87.582, 98.393, 99.895, 100.00)
    # d_30 = (78.858, 90.240, 99.278, 99.980, 100.00)
    # d_10 = (83.404, 95.245, 99.778, 100.00, 100.00)

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
                     label='50% of received data')

    rects3 = plt.bar(index + 2*bar_width, d_30, bar_width,
                     alpha=opacity,
                     color='r',
                     label='30% of received data')

    rects4 = plt.bar(index + 3*bar_width, d_10, bar_width,
                     alpha=opacity,
                     color='y',
                     label='10% of received data')

    plt.xlabel('SNR')
    plt.ylabel('SER Scores')
    plt.title('SER Scores by SNR')
    plt.xticks(index + 1.5*bar_width, ('0dB', '2dB', '4dB', '6dB', '8dB'))
    plt.legend()

    plt.tight_layout()
    plt.show()



    # -----PATH columns plot-----#
    # data to plot
    n_groups = 5
    # d_100 = (34.610, 54.528, 81.428, 92.532, 97.954)
    # d_50 = (40.469, 66.768, 95.520, 99.649, 99.996)
    # d_30 = (45.403, 74.254, 98.213, 99.851, 100.00)
    # d_10 = (54.148, 85.431, 99.243, 99.943, 100.00)

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
                     label='50% of received data')

    rects3 = plt.bar(index + 2 * bar_width, d_30, bar_width,
                     alpha=opacity,
                     color='r',
                     label='30% of received data')

    rects4 = plt.bar(index + 3 * bar_width, d_10, bar_width,
                     alpha=opacity,
                     color='y',
                     label='10% of received data')

    plt.xlabel('SNR')
    plt.ylabel('PATH Scores')
    plt.title('PATH Scores by SNR')
    plt.xticks(index + 1.5*bar_width, ('0dB', '2dB', '4dB', '6dB', '8dB'))
    plt.legend()

    plt.tight_layout()
    plt.show()
