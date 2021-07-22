import numpy as np
import matplotlib.pyplot as plt
from SOVA_path_and_SER import diagram_plot

def success_rate_plot(d_path1):
    # -----PATH columns plot-----#
    d_path = np.array([d_path1[3, :], d_path1[4, :], d_path1[5, :], d_path1[6, :], d_path1[7, :]])

    n_groups = 5

    d_100 = 100 * (1 - d_path[:, 0])
    d_50 = 100 * (1 - d_path[:, 1])
    d_30 = 100 * (1 - d_path[:, 2])
    d_10 = 100 * (1 - d_path[:, 3])

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
    plt.xticks(index + 1.5 * bar_width, ('0dB', '2dB', '4dB', '6dB', '8dB'))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def success_rate_all(d_symbol1, d_symbol2, d_path1, d_path2):
    # success rate for 10,30,50,100% highest SOVA score
    # perfect CSI - symbol detection
    d_symbol1_ = np.array([d_symbol1[3, :], d_symbol1[4, :], d_symbol1[5, :], d_symbol1[6, :], d_symbol1[7, :]])
    # CSI uncertainty - symbol detection
    d_symbol2_ = np.array([d_symbol2[3, :], d_symbol2[4, :], d_symbol2[5, :], d_symbol2[6, :], d_symbol2[7, :]])
    # perfect CSI - path detection
    d_path1_ = np.array([d_path1[3, :], d_path1[4, :], d_path1[5, :], d_path1[6, :], d_path1[7, :]])
    # CSI uncertainty - path detection
    d_path2_ = np.array([d_path2[3, :], d_path2[4, :], d_path2[5, :], d_path2[6, :], d_path2[7, :]])

    diagram_plot(d_symbol1_, d_path1_)
    diagram_plot(d_symbol2_, d_path2_)

def ViterbiNet_plot(v_fSigWdB, m_fSERAvg):
    # ViterbiNet
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

def channel_taps_plot(m_fChannel):
    # channel coefficients
    plt.figure()
    plt.plot(np.arange(200), m_fChannel[:, 0],
             np.arange(200), m_fChannel[:, 1],
             np.arange(200), m_fChannel[:, 2],
             np.arange(200), m_fChannel[:, 3])
    plt.legend(('channel tap 1', 'channel tap 2', 'channel tap 3', 'channel tap 4'))
    plt.title('channel taps variation')
    plt.xlabel('Block')
    plt.ylabel('channel taps')

def track_res_plot(track_SER_fullCSI, track_SER_perfectCSI, track_SER_uncertaintyCSI, v_fSigWdB):
    # moving average of SER
    t_SER_fullCSI = np.zeros(track_SER_fullCSI.shape)
    t_SER_perfectCSI = np.zeros(track_SER_perfectCSI.shape)
    t_SER_uncertaintyCSI = np.zeros(track_SER_uncertaintyCSI.shape)

    t_SER_fullCSI[:, 0, :] = track_SER_fullCSI[:, 0, :]
    t_SER_perfectCSI[:, 0, :] = track_SER_perfectCSI[:, 0, :]
    t_SER_uncertaintyCSI[:, 0, :] = track_SER_uncertaintyCSI[:, 0, :]

    for index in range(24):
        index = index + 1
        t_SER_fullCSI[:, index, :] = (t_SER_fullCSI[:, index - 1, :] * (index) +
                                      track_SER_fullCSI[:, index, :]) / (index + 1)
        t_SER_perfectCSI[:, index, :] = (t_SER_perfectCSI[:, index - 1, :] * (index) +
                                         track_SER_perfectCSI[:, index, :]) / (index + 1)
        t_SER_uncertaintyCSI[:, index, :] = (t_SER_uncertaintyCSI[:, index - 1, :] * (index) +
                                             track_SER_uncertaintyCSI[:, index, :]) / (index + 1)

    dB_wanted = 6
    Idx_dB = np.where(v_fSigWdB == dB_wanted)
    Idx_dB = int(Idx_dB[1])

    plt.figure()
    plt.semilogy(np.arange(25), t_SER_perfectCSI[0, :, Idx_dB], 'ro--',
                 np.arange(25), t_SER_fullCSI[0, :, Idx_dB], 'bo--',
                 np.arange(25), t_SER_perfectCSI[1, :, Idx_dB],
                 np.arange(25), t_SER_perfectCSI[2, :, Idx_dB],
                 np.arange(25), t_SER_perfectCSI[3, :, Idx_dB],
                 np.arange(25), t_SER_perfectCSI[4, :, Idx_dB],
                 np.arange(25), t_SER_perfectCSI[5, :, Idx_dB])
    plt.legend(('ViterbiNet - median training', 'Viterbi algorithm', 'ViterbiNet - No Train',
                'ViterbiNet - low score training', 'ViterbiNet - high score training',
                'ViterbiNet - genie', 'ViterbiNet - genie 20% Data'))
    plt.title(f'SER - Symbol Error Rate\nChannel Tracking - {dB_wanted}dB')
    plt.xlabel('Block')
    plt.ylabel('SER')

    plt.figure()
    plt.semilogy(np.arange(25), t_SER_uncertaintyCSI[0, :, Idx_dB], 'go--',
                 np.arange(25), t_SER_fullCSI[0, :, Idx_dB], 'bo--',
                 np.arange(25), t_SER_uncertaintyCSI[1, :, Idx_dB],
                 np.arange(25), t_SER_uncertaintyCSI[2, :, Idx_dB],
                 np.arange(25), t_SER_uncertaintyCSI[3, :, Idx_dB],
                 np.arange(25), t_SER_uncertaintyCSI[4, :, Idx_dB],
                 np.arange(25), t_SER_uncertaintyCSI[5, :, Idx_dB])
    plt.legend(('ViterbiNet - median training', 'Viterbi algorithm', 'ViterbiNet - No Train',
                'ViterbiNet - low score training', 'ViterbiNet - high score training',
                'ViterbiNet - genie', 'ViterbiNet - genie 20% Data'))
    plt.title(f'SER - Symbol Error Rate\nChannel Tracking - {dB_wanted}dB')
    plt.xlabel('Block')
    plt.ylabel('SER')

def AVGtrack_plot(track_SER_avg, v_fSigWdB):
    # AVG display - for each SNR
    plt.figure()
    plt.semilogy(np.transpose(v_fSigWdB), track_SER_avg[0, :], 'ro--',
                 np.transpose(v_fSigWdB), track_SER_avg[12, :], 'bo--',
                 np.transpose(v_fSigWdB), track_SER_avg[1, :],
                 np.transpose(v_fSigWdB), track_SER_avg[2, :],
                 np.transpose(v_fSigWdB), track_SER_avg[3, :],
                 np.transpose(v_fSigWdB), track_SER_avg[5, :])
    plt.legend(('ViterbiNet - median training', 'Viterbi algorithm',
                'ViterbiNet - No Train',
                'ViterbiNet - low score training',
                'ViterbiNet - high score training',
                'ViterbiNet - genie 20% Data'))
    plt.title('Average SER (Symbol Error Rate)\nChannel Tracking')
    plt.xlabel('SNR')
    plt.ylabel('SER')
    plt.grid()

    plt.figure()
    plt.semilogy(np.transpose(v_fSigWdB), track_SER_avg[6, :], 'go--',
                 np.transpose(v_fSigWdB), track_SER_avg[12, :], 'bo--',
                 np.transpose(v_fSigWdB), track_SER_avg[7, :],
                 np.transpose(v_fSigWdB), track_SER_avg[8, :],
                 np.transpose(v_fSigWdB), track_SER_avg[9, :],
                 np.transpose(v_fSigWdB), track_SER_avg[11, :])
    plt.legend(('ViterbiNet - median training', 'Viterbi algorithm',
                'ViterbiNet - No Train',
                'ViterbiNet - low score training',
                'ViterbiNet - high score training',
                'ViterbiNet - genie 20% Data'))
    plt.title('Average SER (Symbol Error Rate)\nChannel Tracking')
    plt.xlabel('SNR')
    plt.ylabel('SER')
    plt.grid()

def diversity_plot(dict_train, dict_bad_train, dict_med_train,
                   MSE_diversity_train, MSE_diversity_bad_train, MSE_diversity_med_train):
    # diversity
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.bar(list(dict_train.keys()), dict_train.values(), color='g')
    ax1.set_title('high score training diversity MSE')
    ax1.set(xlabel='state number', ylabel='amount of data')

    ax2.bar(list(dict_med_train.keys()), dict_med_train.values(), color='g')
    ax2.set_title('median training diversity MSE')
    ax2.set(xlabel='state number', ylabel='amount of data')

    # ax2.bar(list(dict_bad_train.keys()), dict_bad_train.values(), color='g')
    # # ax2.set_title('bad train diversity MSE')
    # ax2.set(xlabel='state number', ylabel='amount of data')

    # diversity MSE
    plt.figure()
    plt.plot(range(25), MSE_diversity_train[0, :], range(25), MSE_diversity_bad_train[0, :], range(25),
             MSE_diversity_med_train[0, :])
    plt.legend(('high score training', 'low score training', 'median training'))
    plt.title('Training set diversity MSE')
    plt.xlabel('Block')
    plt.ylabel('diversity MSE')
    plt.show()

def NoisePower_error(NOISE_train, NOISE_bad_train, NOISE_med_train, ERR_train, ERR_bad_train, ERR_med_train):
    # noise power
    plt.figure()
    plt.plot(range(25), NOISE_train[0, :], range(25), NOISE_bad_train[0, :], range(25), NOISE_med_train[0, :])
    plt.legend(('high score training', 'low score training', 'median training'))
    plt.title('Training set additive noise power')
    plt.xlabel('Block')
    plt.ylabel('noise sum')
    plt.show()

    # error
    plt.figure()
    plt.plot(range(25), ERR_train[0, :], range(25), ERR_bad_train[0, :], range(25), ERR_med_train[0, :])
    plt.legend(('high score training', 'low score training', 'median training'))
    plt.title('Training set error (SER)')
    plt.xlabel('Block')
    plt.ylabel('error')
    plt.show()
