import math
import random
import numpy as np
import sys
import algo
import torch


Mb = math.pow(10, 6)
Gb = math.pow(10, 9)
bandwidth_unit = math.pow(10, 6)
max_energy = 1
FLOAT_CONFIG = 1.00001
bandwidth_csv = 'throughput.csv'

# model path
fading_predict_model_path = 'pred_model.pth'
dqn_min_max_model_pth = 'model_min_max.pth'
dqn_model_pth = 'model.pth'


def dBm2w(dBm):
    w = math.pow(10, dBm / 10 - 3)
    return w


def w2dBm(w):
    if w==0:
        return 0
    dbm = 10 * math.log10(w) + 30
    return dbm


class Task:
    def __init__(self, min_datasize, max_datasize, min_workloadsize, max_workloadsize):
        self.min_datasize = min_datasize * Mb
        self.max_datasize = max_datasize * Mb
        self.min_workloadsize = min_workloadsize * Mb
        self.max_workloadsize = max_workloadsize * Mb
        self.mobile_num = None
        self.relay_num = None
        self.data_size = None
        self.workload_size = None
        self.mobile_topology = None
        self.relay_topology = None
        self.edge_topology = [100, 100]
        self.time_constraint = None

    def init_env(self, mobile_num, relay_num, time_constraint=0.1):
        self._gen_info(mobile_num, relay_num, time_constraint=time_constraint)
        self._gen_topology()

    def _gen_info(self, mobile_num, relay_num, time_constraint=0.1):
        self.mobile_num = mobile_num
        self.relay_num = relay_num
        self.data_size = [int(random.uniform(self.min_datasize, self.max_datasize)) for _ in
                          range(self.mobile_num)]
        self.workload_size = [int(random.uniform(self.min_workloadsize, self.max_workloadsize))
                              for _ in range(self.mobile_num)]
        self.time_constraint = []
        for _ in range(self.mobile_num):
            self.time_constraint.append(time_constraint)

    def _gen_topology(self):
        self.mobile_topology = []
        for _ in range(self.mobile_num):
            self.mobile_topology.append([float(random.uniform(0, 200)) for _ in range(2)])
        self.relay_topology = []
        for _ in range(self.relay_num):
            self.relay_topology.append([float(random.uniform(0, 200)) for _ in range(2)])
        self.mobile_move_rate = []
        self.relay_move_rate = []
        for i in range(self.mobile_num):
            self.mobile_move_rate.append([random.uniform(0, 1), random.uniform(0, 1)])
        for i in range(self.relay_num):
            if i%2 == 0:
                self.relay_move_rate.append([0, 0])
            self.relay_move_rate.append([random.uniform(0, 2), random.uniform(0, 2)])

    def set_time_constraint(self, min_time, max_time=None):
        if max_time is None:
            self.time_constraint = [min_time for _ in range(self.mobile_num)]
        else:
            self.time_constraint = [random.uniform(min_time, max_time) for _ in range(self.mobile_num)]

    def gen_new_env(self, min_time, max_time=None):
        """
        new datasize, new workloadsize, the update topology (by rate and the location)
        :param min_time:
        :param max_time:
        :return:
        """
        self._gen_info(self.mobile_num, self.relay_num)
        self.set_time_constraint(min_time, max_time)
        # update topology
        for i in range(self.mobile_num):
            x_move = 1 if random.uniform(0, 1) < 0.8 else 0
            y_move = 1 if random.uniform(0, 1) < 0.8 else 0
            self.mobile_topology[i][0] = self.mobile_topology[i][0] + x_move * self.mobile_move_rate[i][0]
            self.mobile_topology[i][1] = self.mobile_topology[i][1] + y_move * self.mobile_move_rate[i][1]
            if self.mobile_topology[i][0] > 100 or self.mobile_topology[i][0] < 0:
                self.mobile_move_rate[i] = [-self.mobile_move_rate[i][0], self.mobile_move_rate[i][1]]
                if self.mobile_topology[i][0] > 100:
                    self.mobile_topology[i] = [100, self.mobile_topology[i][1]]
                elif self.mobile_topology[i][0] < 0:
                    self.mobile_topology[i] = [0, self.mobile_topology[i][1]]
            if self.mobile_topology[i][1] > 100 or self.mobile_topology[i][1] < 0:
                self.mobile_move_rate[i] = [self.mobile_move_rate[i][0], -self.mobile_move_rate[i][1]]
                if self.mobile_topology[i][1] > 100:
                    self.mobile_topology[i] = [self.mobile_topology[i][0], 100]
                elif self.mobile_topology[i][1] < 0:
                    self.mobile_topology[i] = [self.mobile_topology[i][0], 0]
        for j in range(self.relay_num):
            x_move = 1 if random.uniform(0, 1) < 0.8 else 0
            y_move = 1 if random.uniform(0, 1) < 0.8 else 0
            self.relay_topology[j] = [
                self.relay_topology[j][0] + x_move * self.relay_move_rate[j][0],
                self.relay_topology[j][1] + y_move * self.relay_move_rate[j][1]
            ]
            if self.relay_topology[j][0] > 100 or self.relay_topology[j][0] < 0:
                self.relay_move_rate[j] = [-self.relay_move_rate[j][0], self.relay_move_rate[j][1]]
                if self.relay_topology[j][0] > 100:
                    self.relay_topology[j] = [100, self.relay_topology[j][1]]
                elif self.relay_topology[j][0] < 0:
                    self.relay_topology[j] = [0, self.relay_topology[j][1]]
            if self.relay_topology[j][1] > 100 or self.relay_topology[j][1] < 0:
                self.relay_move_rate[j] = [self.relay_move_rate[j][0], -self.relay_move_rate[j][1]]
                if self.relay_topology[j][1] > 100:
                    self.relay_topology[j] = [self.relay_topology[j][0], 100]
                elif self.relay_topology[j][1] < 0:
                    self.relay_topology[j] = [self.relay_topology[j][0], 0]

    def update_env_by_data(self, datasize, workloadsize, mobile_topology, relay_topology, time_constraint=0.1):
        self.data_size = datasize
        self.workload_size = workloadsize
        self.mobile_topology = mobile_topology
        self.relay_topology = relay_topology
        self.set_time_constraint(time_constraint)


class Environment:
    def __init__(self, bandwidth=2, relay_awgn=-70, edge_awgn=-40):
        self.bandwidth = int(bandwidth)
        # self.pathloss = pathloss
        self.relay_awgn = dBm2w(relay_awgn)
        self.edge_awgn = dBm2w(edge_awgn)


class Device:
    def __init__(self):
        self.mobile_min_f = 100 * math.pow(10, 6)
        self.mobile_max_f = 600 * math.pow(10, 6)
        self.relay_min_f = 0 * math.pow(10, 6)
        self.relay_max_f = 1000 * math.pow(10, 6)
        self.edge_f = 3000 * math.pow(10, 6)
        self.mobile_k = math.pow(10, -27)
        self.relay_k = 0.3 * math.pow(10, -27)
        self.mobile_min_p = dBm2w(15)
        self.mobile_max_p = dBm2w(30)
        self.relay_min_p = dBm2w(10)
        self.relay_max_p = dBm2w(30)

class Min_max():
    def __init__(self):
        pass

    def start(self, matrix, Priority):
        self.row, self.col = matrix.shape
        self.max = np.max(np.max(matrix))
        priority = Priority.copy()
        # random select
        self.right_match = np.zeros([self.col, 1], dtype=np.int8) - 1
        self.match = np.zeros([self.row, 1], dtype=np.int8) - 1
        self.matrix = matrix.copy()
        self.static_matrix = matrix.copy()
        for _ in range(min(self.row, self.col)):
            tmp = np.max(priority)
            i = np.argwhere(priority == tmp)[0][0]
            r = np.random.randint(0, self.col)
            for _ in range(self.col):
                if (self.right_match[r] == -1) and (not np.isinf(self.matrix[i][r])):
                    self.right_match[r] = i
                    self.match[i] = r
                    priority[i] = 0
                    break
                r = (r + 1) % self.col
        self.vector = np.zeros([self.row, 1], dtype=np.float64)
        for i in range(self.row):
            if self.match[i] == -1:
                continue
            self.vector[i] = self.matrix[i][self.match[i]]
        while True:
            breakLabel = True
            tmp_max = np.max(self.vector)
            E_max = np.max(self.vector)
            i = np.argwhere(E_max == self.vector)[0][0]
            E_min = np.min(matrix[i])
            r = np.argwhere(matrix[i] == E_min)[0][0]
            self.label_right = [False] * self.col
            tmp = self.right_match[self.match[i]]
            self.right_match[self.match[i]] = -1
            if self.check_avai(i, r, E_max):
                self.match[i] = r
                self.right_match[r] = i
                self.vector[i] = self.static_matrix[i][r]
                self.matrix[i][r] = self.max
                breakLabel = False
            else:
                self.right_match[r] = tmp
                self.matrix[i][r] = self.max
            if breakLabel:
                break

            if tmp_max == np.max(self.vector):
                break

        return self.match

    def check_avai(self, s, r, E_max):
        if self.right_match[r] == -1:
            return True
        if self.matrix[s][r] == E_max:
            return True
        self.label_right[r] = True
        find_s = self.right_match[r]
        if self.find_for(find_s, E_max):
            return True
        return False

    def find_for(self, s, E_max):
        s = s[0]
        t = self.static_matrix[s].copy()
        for _ in range(self.col):
            e_min = np.min(t)
            r = np.argwhere(e_min == t)[0][0]
            if self.label_right[r]:
                t[r] = self.max
                continue
            if e_min <= E_max:
                tmp = (self.right_match[self.match[s]])[0][0]
                self.right_match[self.match[s]] = -1
                if self.check_avai(s, r, E_max):
                    self.match[s] = r
                    self.right_match[r] = s
                    self.vector[s] = self.static_matrix[s][r]
                    self.matrix[s][r] = self.max
                    return True
                else:
                    self.right_match[self.match[s]] = tmp
                    t[r] = self.max

        return False


def process_action_value(action, mobile_num, relay_num):
    action = action[0] if len(action.shape) > 1 else action
    policy_value = action[:4 * mobile_num]
    match_value = action[4 * mobile_num:]
    policy_value = policy_value.reshape((mobile_num, 4))
    match_value = match_value.reshape((mobile_num, relay_num))
    process_policy = [0] * mobile_num
    process_match = [-1] * mobile_num
    relay_used = [False] * relay_num
    for i in range(mobile_num):
        t = torch.argmax(policy_value[i])
        process_policy[i] = int(t) + 1
        if process_policy[i] <= 2:
            continue
        _, index = match_value[i].sort(descending=True)
        _, order = index.sort()
        for j_index in order:
            if not relay_used[int(j_index)]:
                process_match[i] = int(j_index)
                relay_used[j_index] = True
                break
        if process_match[i] > -1:
            # match
            continue
        else:
            t = torch.argmax(policy_value[i][:2])
            process_policy[i] = int(t) + 1
    return process_policy, process_match


def get_reward(action, task, env, device, fading, transform=False):
    if transform:
        policy, match = process_action_value(action, task.mobile_num, task.relay_num)
    else:
        policy = action[:task.mobile_num]
        match = action[task.mobile_num:]
    computing_info = algo.get_computing_info(task, env, device, fading)
    weigtht = algo.cal_weight(computing_info, task.mobile_num, task)
    total_energy, max_energy = algo.cal_object_value(policy, match, computing_info, weigtht)
    return 1-total_energy, 1-max_energy


def get_state_by_index(i, task, env, fading):
    import A2C_
    datasize = task.data_size
    workloadsize = task.workload_size
    mobile_topo = task.mobile_topology
    mobile_num = task.mobile_num
    time_constraint = task.time_constraint
    task.data_size = [datasize[i]]
    task.workload_size = [workloadsize[i]]
    task.mobile_topology = [mobile_topo[i]]
    task.mobile_num = 1
    task.time_constraint = [time_constraint[i]]
    state = A2C_.DQN_A2C.get_state(task, env, fading)
    task.data_size = datasize
    task.workload_size = workloadsize
    task.mobile_topology = mobile_topo
    task.mobile_num = mobile_num
    task.time_constraint = time_constraint
    return state


def store2csv(data, label, filename):
    import pandas as pd
    data_T = [list(row) for row in zip(*data)]
    d = pd.DataFrame(data=data_T, columns=label)
    d.to_csv(filename)
