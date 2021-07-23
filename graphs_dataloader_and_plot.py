import pickle
from graph_plot_func import *
# import numpy as np
# import matplotlib.pyplot as plt

"""
1. Saving results simulation DATA (variables) to binary files.
2. Loading the DATA and plot the results.
"""

#---success rate---#
# pickle_success_rate = open("pickle_success_rate.pickle", "wb")
# pickle.dump(d_path1, pickle_success_rate)
# pickle_success_rate.close()
pickle_success_rate_in = open("pickle_success_rate.pickle", "rb")
d_path1 = pickle.load(pickle_success_rate_in)
pickle_success_rate_in.close()
success_rate_plot(d_path1)


#---ViterbiNet---#
# import pickle
# pickle_ViterbiNet = open("pickle_ViterbiNet.pickle", "wb")
# pickle.dump([m_fSERAvg, v_fSigWdB], pickle_ViterbiNet)
# pickle_ViterbiNet.close()
pickle_ViterbiNet_in = open("pickle_ViterbiNet.pickle", "rb")
[m_fSERAvg, v_fSigWdB] = pickle.load(pickle_ViterbiNet_in)
pickle_ViterbiNet_in.close()
ViterbiNet_plot(v_fSigWdB, m_fSERAvg)


#---channel tracking simulation results (median point chosen to be optimal for SNR=6dB)---:
#---channel coefficients---#
# import pickle
# pickle_channel_taps = open("pickle_channel_taps.pickle", "wb")
# pickle.dump(m_fChannel, pickle_channel_taps)
# pickle_channel_taps.close()
pickle_channel_taps_in = open("pickle_channel_taps.pickle", "rb")
m_fChannel = pickle.load(pickle_channel_taps_in)
pickle_channel_taps_in.close()
channel_taps_plot(m_fChannel)


#---moving average of SER---#
# import pickle
# pickle_track_res = open("pickle_track_res.pickle", "wb")
# pickle.dump([track_SER_fullCSI, track_SER_perfectCSI, track_SER_uncertaintyCSI, v_fSigWdB], pickle_track_res)
# pickle_track_res.close()
pickle_track_res_in = open("pickle_track_res.pickle", "rb")
[track_SER_fullCSI, track_SER_perfectCSI, track_SER_uncertaintyCSI, v_fSigWdB] = pickle.load(pickle_track_res_in)
pickle_track_res_in.close()
dB_wanted = 6  #chosen SNR results [-6,-4,-2,0,2,4,6,8,10]
track_res_plot(track_SER_fullCSI, track_SER_perfectCSI, track_SER_uncertaintyCSI, v_fSigWdB, dB_wanted)


#---AVG display - for each SNR---#
# import pickle
# pickle_AVGtrack_res = open("pickle_AVGtrack_res.pickle", "wb")
# pickle.dump([track_SER_avg, v_fSigWdB], pickle_AVGtrack_res)
# pickle_AVGtrack_res.close()
pickle_AVGtrack_res_in = open("pickle_AVGtrack_res.pickle", "rb")
[track_SER_avg, v_fSigWdB] = pickle.load(pickle_AVGtrack_res_in)
pickle_AVGtrack_res_in.close()
AVGtrack_plot(track_SER_avg, v_fSigWdB)


#---diversity MSE---#
# import pickle
# pickle_diversity = open("pickle_diversity.pickle", "wb")
# pickle.dump([dict_train, dict_bad_train, dict_med_train,
#              MSE_diversity_train, MSE_diversity_bad_train, MSE_diversity_med_train], pickle_diversity)
# pickle_diversity.close()
pickle_diversity_in = open("pickle_diversity.pickle", "rb")
[dict_train, dict_bad_train, dict_med_train,
             MSE_diversity_train, MSE_diversity_bad_train, MSE_diversity_med_train] = pickle.load(pickle_diversity_in)
pickle_diversity_in.close()
diversity_plot(dict_train, dict_bad_train, dict_med_train,
             MSE_diversity_train, MSE_diversity_bad_train, MSE_diversity_med_train)


#---noise power & error---#
# import pickle
# pickle_NoisePower_error = open("pickle_NoisePower_error.pickle", "wb")
# pickle.dump([NOISE_train, NOISE_bad_train, NOISE_med_train,
#              ERR_train, ERR_bad_train, ERR_med_train], pickle_NoisePower_error)
# pickle_NoisePower_error.close()
pickle_NoisePower_error_in = open("pickle_NoisePower_error.pickle", "rb")
[NOISE_train, NOISE_bad_train, NOISE_med_train,
             ERR_train, ERR_bad_train, ERR_med_train] = pickle.load(pickle_NoisePower_error_in)
pickle_NoisePower_error_in.close()
NoisePower_error(NOISE_train, NOISE_bad_train, NOISE_med_train, ERR_train, ERR_bad_train, ERR_med_train)

plt.show(block=True)
print('Simulation results')
print('')
