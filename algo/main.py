import math
import time
from tkinter.messagebox import NO
from unittest import result

import algo
import utils
import pandas as pd
import numpy as np
import random
from pred import LSTM
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd



def exper(NUM=True, IMG=True, delay=0.1, band=2, seed=60, pred=True, acutal=False):
    random.seed(seed)
    mobile_num = 20
    relay_num = 20
    if NUM:
        task = utils.Task(0.3, 0.5, 15, 20)  # (0.5, 0.7, 10, 15)
    else:
        if IMG:
            # image task
            task_datasize = [
                12802, 6452, 21819, 18429, 7921,
                25134, 16229, 6020, 14176, 15321,
                9608, 5154, 5663, 8193, 19251,
                10246, 7565, 6628, 14077, 14950,
                5031, 7864, 11640, 9071, 20310
            ]
            task_workloadsize = [
                11476839, 11644901, 12078055, 11823981, 11453533,
                12027002, 12469431, 11857990, 12305889, 11824219,
                12041192, 12771307, 14233837, 11715217, 11567053,
                12171281, 11838805, 12619577, 11814627, 11471528,
                11337951, 11767141, 11629442, 11380283, 11690008
            ]
        else:
            # video task
            task_datasize = [
                561880, 346738, 214848, 274926, 232376,
                686714, 298388, 718614, 294406, 315268,
                212234, 134574, 662794, 311352, 258264,
                287096, 297472, 226668, 250574, 318458
            ]
            task_workloadsize = [
                55169357, 58432996, 55436194, 52628303, 54136970,
                53585776, 54520339, 55023413, 52164469, 52350209,
                52756013, 52941911, 54228928, 53958920, 53574044,
                52907982, 52587952, 55393307, 54480306, 52320719
            ]
        min_datasize = min(task_datasize)
        max_datasize = max(task_datasize)
        min_workloadsize = min(task_workloadsize)
        max_workloadsize = max(task_workloadsize)
        task = utils.Task(min_datasize, max_datasize, min_workloadsize, max_workloadsize)
    bandwidth = band * utils.bandwidth_unit
    time_constraint = delay
    env = utils.Environment(bandwidth=bandwidth)
    device = utils.Device()
    task.init_env(mobile_num, relay_num)
    ALGO = None
    use_LSTM = None
    if pred:
        ALGO = [True, True, True, False,
                 True, True, True, True,
                 False] 
        use_LSTM = [True, True, True, False, False, False, False, False, False]
    else:
        ALGO = [True, True, False, False,
                 False, False, False, False,
                 False] 
        use_LSTM = [False, False, True, False, False, False, False, False, False]
    ALGO_NAME = ['OTCA', 'GAF', 'DQN', 'GARR', 'LOCAL', 'CC', 'Direct', 'DF', 'RAND']
    # ALGO label
    ALGO_LABEL = []
    for i in range(len(ALGO)):
        if ALGO[i]:
            ALGO_LABEL.append(ALGO_NAME[i])
    print(ALGO_LABEL)
    # dynamic network throught put
    throught_put = pd.read_csv('throughput.csv')
    network_through = np.array(throught_put.iloc[:, 0])
    max_rate = np.max(network_through)
    # lstm = LSTM()
    seq_len = 5
    lstm = torch.load(utils.fading_predict_model_path)
    # topology
    data_size = []
    workload_size = []
    mobile_loc = []
    relay_loc = []
    for t in range(network_through.shape[0]):
        data_size.append(task.data_size)
        workload_size.append(task.workload_size)
        mobile_loc.append(task.mobile_topology)
        relay_loc.append(task.relay_topology)
        task.gen_new_env(time_constraint)
    RECORD_TOTAL = []
    RECORD_MAX = []
    RECORD_TIME = []
    for i in range(len(ALGO)):
        if not ALGO[i]:
            continue
        rest_datasize = [0 for _ in range(task.mobile_num)]
        rest_workloadsize = [0 for _ in range(task.mobile_num)]
        TOTAL_ENERGY_RECORD = []
        MAX_ENERGY_RECORD = []
        running_time = 0
        for t in tqdm(range(263, network_through.shape[0]), desc=ALGO_NAME[i]):  # the true iter: network_through.shape[0]
            # update env
            d = [rest_datasize[k] + data_size[t][k] for k in range(task.mobile_num)]
            w = [rest_workloadsize[k] + workload_size[t][k] for k in range(task.mobile_num)]
            m = mobile_loc[t].copy()
            r = relay_loc[t].copy()
            task.update_env_by_data(d, w, m, r, time_constraint=time_constraint)
            # predict network
            running_time = running_time - time.time()
            if use_LSTM[i]:
                the_front_network = network_through[t - seq_len:t]
                the_front_network = the_front_network.reshape((1, -1, seq_len))
                the_front_network = torch.DoubleTensor(the_front_network)
                rate = lstm(Variable(the_front_network))
                fading = math.pow(2, rate/max_rate) - 1
            else:
                # fading = algo.cal_pred_through(network_through, t, front=seq_len)
                fading = 0.01
            # update network
            # algo
            if acutal:
                fading = math.pow(2, network_through[t]/max_rate) - 1
            if i == 0:
                mode, match, computing_info = algo.otca(task, env, device, fading)
            elif i == 1:
                mode, match, computing_info = algo.oecaa(task, env, device, fading)
            elif i == 2:
                # dqn total
                mode, match, computing_info = algo.dqn(task, env, device, fading)
            elif i == 3:
                # comp, not relay
                mode, match, computing_info = algo.comp(task, env, device, fading)
            elif i == 4:
                # local
                mode, match, computing_info = algo.local(task, env, device, fading)
            elif i == 5:
                # cc
                mode, match, computing_info = algo.cc(task, env, device, fading)
            elif i == 6:
                mode, match, computing_info = algo.dec(task, env, device, fading)
            elif i == 7:
                mode, match, computing_info = algo.df(task, env, device, fading)
            else:
                # rand
                mode, match, computing_info = algo.rand(task, env, device, fading)
            weight = algo.cal_weight(computing_info, task.mobile_num, task=task)
            running_time = running_time + time.time()
            if not NUM:
                # 仿真实验
                with open("topo.txt", "w") as f:
                    for m in range(task.mobile_num):
                        f.write("{}\t".format(task.data_size[m]))
                    f.write("\n")
                    for m in range(task.mobile_num):
                        if mode[m] == 1:
                            w = 0
                        elif mode[m] == 2:
                            w = computing_info[1].mobile_power[m]
                        elif mode[m] == 3 or mode[m] == 4:
                            w = computing_info[mode[m]-1].mobile_power[m][match[m]]
                        else:
                            w = device.mobile_min_p
                        db = utils.w2dBm(w)
                        f.write("{}\t".format(db))
                    f.write("\n")
                    for m in range(task.mobile_num):
                        if mode[m] == 4:
                            w = computing_info[3].relay_power[m][match[m]]
                        else:
                            w = 0
                        db = utils.w2dBm(w)
                        f.write("{}\t".format(db))
                    f.write("\n")
                    for m in range(task.mobile_num):
                        f.write("{:.2f}\t{:.2f}\t".format(task.mobile_topology[m][0], task.mobile_topology[m][1]))
                    f.write("\n")
                    for m in range(task.relay_num):
                        f.write("{:.2f}\t{:.2f}\t".format(task.relay_topology[m][0], task.relay_topology[m][1]))
                    f.write("\n")
                    for m in range(task.mobile_num):
                        f.write("{}\t".format(mode[m]))
                    f.write("\n")
                    for m in range(task.mobile_num):
                        f.write("{}\t".format(match[m]))
                    f.write("\n")
                tx_time = [0 for _ in range(task.mobile_num)]
                with open("time.txt") as f:
                    t = f.readline()
                    t = t.split()
                    for m in range(task.mobile_num):
                        tx_time[m] = float(t[m])
                total_energy, max_energy, _rest_datasize, _rest_workloadsize = algo.ns3_process(tx_time, mode, match, task, env, device, weight)
            else:
                # record algo
                total_energy, max_energy, _rest_datasize, _rest_workloadsize = algo.cal_result_predict(mode, match, computing_info, weight, network_through[t], task, env, device)
            # update the rest bit
            for k in range(task.mobile_num):
                rest_datasize[i] = _rest_datasize[i]
                rest_workloadsize[i] = _rest_workloadsize[i]
            # record detail
            TOTAL_ENERGY_RECORD.append(total_energy)
            MAX_ENERGY_RECORD.append(max_energy)
        RECORD_TOTAL.append(TOTAL_ENERGY_RECORD)
        RECORD_MAX.append(MAX_ENERGY_RECORD)
        RECORD_TIME.append(running_time / (network_through.shape[0] - 263))
    return [sum(RECORD_TOTAL[i]) for i in range(len(RECORD_TOTAL))], [sum(RECORD_MAX[i]) for i in range(len(RECORD_MAX))], RECORD_TIME


if __name__ == '__main__':
    exper(NUM=True, delay=0.2, band=5, pred=True, acutal=False)
