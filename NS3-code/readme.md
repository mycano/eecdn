This folder only contains the NS3 code.

We use the yans-wifi-channel to simulation the experiment. And four computing modes are described as 1 (local computing), 2 (Dtx-Edge), 3 (Dtx-Relay), 4 (CTx-Edge). 

The topology graph should be the input txt file, which will be read by  `EXPER.cc`. Besides, the result of NS3 will be written by `EXPER.cc`. For the simulation result, the NS3 only contains the transmission phase, while the execution phase are measured by different inference task.