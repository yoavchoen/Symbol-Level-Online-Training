# Symbol-Level-Online-Training
**Symbol-Level Online Channel Tracking for DeepLearning-Aided Receivers.**

based on the paper:

R.A.Finish, Y.Cohen, T.Raviv, and N.Shlezinger

**"Symbol-Level Online Channel Tracking for Deep Learning-Aided Receivers"**


## Repository content ##
The implementation of the system includes several parts -
- NetClass - contain network definition and its methods:
  - generate and train ViterbiNet detector.
  - use trained model to detect symbols.

- v_fViterbi_file - Viterbi algorithm implementation with SOVA.

- graph_data_loader_and_plot - Load the results data from the '.pickle' files and display them.

A code example for evaluating symbol-level online channel tracking for ViterbiNet
can be found in the script Test_ChannelTrack_ViterbiNet.py.
