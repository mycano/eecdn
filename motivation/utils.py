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
        self.edge_topology = [50, 50]
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
            self.mobile_topology.append([float(random.uniform(0, 100)) for _ in range(2)])
        self.relay_topology = []
        for _ in range(self.relay_num):
            self.relay_topology.append([float(random.uniform(0, 100)) for _ in range(2)])
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
    def __init__(self, bandwidth=10, pathloss=1, relay_awgn=-174, edge_awgn=-70):
        self.bandwidth = int(bandwidth)
        self.pathloss = pathloss
        self.relay_awgn = dBm2w(relay_awgn)
        self.edge_awgn = dBm2w(edge_awgn)


class Device:
    def __init__(self):
        self.mobile_min_f = 100 * math.pow(10, 6)
        self.mobile_max_f = 600 * math.pow(10, 6)
        self.relay_min_f = 100 * math.pow(10, 6)
        self.relay_max_f = 1000 * math.pow(10, 6)
        self.edge_f = 3000 * math.pow(10, 6)
        self.mobile_k = math.pow(10, -27)
        self.relay_k = 0.5 * math.pow(10, -27)
        self.mobile_min_p = dBm2w(20)
        self.mobile_max_p = dBm2w(30)
        self.relay_min_p = dBm2w(10)
        self.relay_max_p = dBm2w(30)
