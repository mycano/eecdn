<!--
 * @Author: myyao
 * @Date: 2022-02-02 16:48:18
 * @Description: 
-->

# Summary

This folder contains the code about network change. Then, the `pred/throughput.csv` file is from paper [1], which are continuing measurement and keep adding new trances to NYU-METS Dataset.
Meanwhile, `pred/throughput.csv` is one of data from NYU-METS. 

We use `pred/throughput.csv` to simulate the network change, and all of experiment are from them. We use LSTM model to capture the network change. By default, the seq_len is set as 5, which is the input size of LSTM model. Other parameters can be found in the `pred/main.py` file.

# How to use
the `throughput.csv` file is obtained from [1].
Then, the `pred/main.py` file is used to train the model to fix the change of network.

To run the main.py file, the following packages need to be installed.
* matplotlib
* pandas
* numpy
* tqdm
* torch
* torchvision

After running the `pred/main.py` file, the output file `pred.pth` is the weight of the trained model, which is used for predicting network before our algorithm.

[1] Mei L, Hu R, Cao H, et al. Realtime mobile bandwidth prediction using LSTM neural network and Bayesian fusion[J]. Computer Networks, 2020, 182: 107515.