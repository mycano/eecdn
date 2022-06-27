import sys

import utils
import math
import numpy as np
import random

# due to the metric of FLOAT, thus we use utils.FLOAT_CONFIG to cope with.


class CalInfo:
    def __init__(self):
        self.f = []
        self.energy = []
        self.time = []
        self.mobile_power = []
        self.relay_power = []

    def print_info(self):
        if len(self.f) != 0:
            print("self.f: ", self.f)
        print("self.energy: ", self.energy)
        print("self.time: ", self.time)
        if len(self.mobile_power) != 0:
            print("self.mobile_power: ", self.mobile_power)
        if len(self.relay_power) != 0:
            print("self.relay_power:", self.relay_power)


def get_fix_value(fix_value, min_value, max_value):
    fix_value = max([fix_value, min_value])
    fix_value = min([fix_value, max_value])
    return fix_value


def cal_f(time_constraint, workload, min_f=None, max_f=None):
    fix_f = int(math.ceil(workload * utils.FLOAT_CONFIG / time_constraint))
    if min_f is not None and max_f is not None:
        fix_f = get_fix_value(fix_f, min_f, max_f)
    return fix_f


def cal_comp_energy(k, f, workload):
    energy = k * workload * math.pow(f, 2)
    return energy


def cal_comp_time(workload, f):
    try:
        t = workload / f
    except:
        t = 100
    return t


def cal_trans_rate(b, awgn, power, distance, fading):
    channel_gain = cal_channel_gain(distance, fading)
    rate = b * math.log2(1 + power * channel_gain / math.pow(awgn, 2))
    return rate


def cal_distance(topo_1, topo_2):
    distance = math.pow(topo_1[0] - topo_2[0], 2)
    distance += math.pow(topo_1[1] - topo_2[1], 2)
    distance = math.sqrt(distance)
    return distance


def cal_trans_time(datasize, rate):
    try:
        time = datasize / rate
    except:
        time = 1000
    return time


def cal_trans_energy(time, p):
    energy = time * p
    return energy


def cal_channel_gain(distance, fading, eta=4):
    # return math.pow(distance, -eta) * math.pow(fading, 2)
    if distance != 0:
        value = math.pow(distance, -eta) * math.pow(fading, 2)
    else:
        value = math.pow(0.1, -eta) * math.pow(fading, 2)
    return value

def local_computing_per_device(time_constraint, workload_size, min_f, max_f, k):
    """
    an api to get the information of local computing per device
    :param time_constraint:
    :param workload_size:
    :param min_f:
    :param max_f:
    :param k:
    :return:
    """
    f = cal_f(time_constraint, workload_size, min_f, max_f)
    energy = cal_comp_energy(k, f, workload_size)
    t = cal_comp_time(workload_size, f)
    return f, energy, t


def local_compting_rest_bit(time_constraint, datasize, workload_size, min_f, max_f, k):
    f, _, t = local_computing_per_device(time_constraint, workload_size, min_f, max_f, k)
    if t <= time_constraint:
        return 0, 0
    comp_bit = t * f
    rate = comp_bit / workload_size
    rest_data_size = (1 - rate) * datasize
    rest_workload_size = workload_size - comp_bit
    return rest_data_size, rest_workload_size


def local_computing_energy_true_energy(time_constraint, workload_size, min_f, max_f, k):
    f, energy, t = local_computing_per_device(time_constraint, workload_size, min_f, max_f, k)
    if t <= time_constraint:
        pass
    else:
        rate = time_constraint / t
        energy = energy * rate
    return energy


def direct_edge_computing_per_device(i, task, env, device, fading):
    """
    :param i: mobile device id
    :param task:
    :param env:
    :param device:
    :return: energy, time, power
    """

    def cal_trans_power(trans_time, datasize, b, awgn, distance, fading, min_p, max_p):
        min_trans_rate = datasize / trans_time
        channel_gain = cal_channel_gain(distance, fading)
        if channel_gain == 0:
            channel_gain = 0.000001
        p = math.pow(awgn, 2) / channel_gain * math.pow(2, min_trans_rate / b) - 1
        p = get_fix_value(p, min_p, max_p)
        return p

    data_size = task.data_size[i]
    workload_size = task.workload_size[i]
    time_constraint = task.time_constraint[i]
    mobile_topo = task.mobile_topology[i]
    edge_topo = task.edge_topology

    comp_time = cal_comp_time(workload_size, device.edge_f)
    trans_time = time_constraint - comp_time * utils.FLOAT_CONFIG
    if trans_time > 0:
        distance = cal_distance(mobile_topo, edge_topo)
        p = cal_trans_power(trans_time, data_size, env.bandwidth, env.edge_awgn, distance, fading,
                            device.mobile_min_p, device.mobile_max_p)
        rate = cal_trans_rate(env.bandwidth, env.edge_awgn, p, distance, fading)
        trans_time = cal_trans_time(data_size, rate)
        trans_energy = cal_trans_energy(trans_time, p)
    else:
        trans_time = time_constraint
        trans_energy = utils.max_energy
        p = device.mobile_max_p
    return trans_energy, trans_time + comp_time, p


def dec_computing_rest_bit(i, task, env, device, fading):
    _, t, p = direct_edge_computing_per_device(i, task, env, device, fading)
    if t <= task.time_constraint[i]:
        return 0, 0
    comp_time = task.workload_size[i] / device.edge_f
    tran_time = t - comp_time
    if tran_time <= task.time_constraint[i]:
        # full offloading, but can't execute
        rest_data_size = 0
        true_comp_time = task.time_constraint[i] - tran_time
        cal_workloadsize = true_comp_time * device.edge_f
        rest_workloadsize = task.workload_size[i] - cal_workloadsize
    else:
        distance = cal_distance(task.mobile_topology[i], task.edge_topology)
        trans_data_bit = cal_trans_rate(env.bandwidth, env.edge_awgn, p, distance, fading)
        rest_data_size = task.data_size[i] - trans_data_bit
        rest_workloadsize = task.workload_size[i]
    return rest_data_size, rest_workloadsize


def dec_computing_true_energy(i, task, env, device, fading):
    energy, t, p = direct_edge_computing_per_device(i, task, env, device, fading)
    if t <= task.time_constraint[i]:
        pass
    else:
        comp_time = task.workload_size[i] / device.edge_f
        if t - comp_time <= task.time_constraint[i]:
            pass
        else:
            energy = p * task.time_constraint[i]
    return energy


def cc_computing_per_device(i, j, task, env, device, fading):
    """
    :param i:
    :param j:
    :param task:
    :param env:
    :param device:
    :param fading:
    :return: power, f, time, energy
    """
    mobile_topo = task.mobile_topology[i]
    relay_topo = task.relay_topology[j]
    distance = cal_distance(mobile_topo, relay_topo)
    channel_gain = cal_channel_gain(distance, fading)
    G = channel_gain / math.pow(env.relay_awgn, 2)

    def cc_fun(K, N=G, d=task.data_size[i], t=task.time_constraint[i], w=task.workload_size[i], B=env.bandwidth, k=device.relay_k,
               p_min=device.mobile_min_p):
        # 用于CC模型下的binary search
        try:
            v_1 = d / (B * math.log2(K))
        except:
            print("d = {}, b = {}, K = {}".format(d, B, K))
            exit(0)
        # m_sqrt = math.log(2) * K * math.pow(math.log2(K), 2) / (2 * N * k * math.log2(1 + p_min * N))
        m_sqrt = (math.log(2) * K * math.pow(math.log2(K), 2) + K - 1 ) / (2 * K * k)
        v_2 = math.pow(m_sqrt, 1 / 3)
        v = v_1 + w / v_2 - t
        return v

    def binary_search(K_min, K_max):
        K_opt = -1
        K_max_v = cc_fun(K_max)
        K_min_v = cc_fun(K_min)
        if K_max_v * K_min_v > 0:
            # 表示二者同正或者同负
            if K_min < 0:
                K_opt = K_max
            else:
                K_opt = K_min
        else:
            K_max_v = cc_fun(K_max)
            K_min_v = cc_fun(K_min)
            while K_max_v * K_min_v < 0:
                K_mid = (K_max + K_min) / 2
                K_mid_v = cc_fun(K_mid)
                if K_mid_v == 0:
                    K_opt = K_mid
                    break
                if K_mid_v < 0:
                    K_min = K_mid
                    K_min_v = cc_fun(K_min)
                else:
                    K_max = K_mid
                    K_max_v = cc_fun(K_max)
                if math.fabs(K_max - K_min) < math.pow(10, -9):
                    K_opt = K_min
                    break
        return K_opt

    data_size = task.data_size[i]
    workload_size = task.workload_size[i]
    K_min = 1.0000001
    K_max = 1 + device.mobile_max_p * G
    # print("device.mobile_max_p = {}, G = {}".format(device.mobile_max_p, G))
    K_opt = binary_search(K_min, K_max)
    p_opt = get_fix_value((K_opt - 1) / G, device.mobile_min_p, device.mobile_max_p)

    trans_rate = cal_trans_rate(env.bandwidth, env.relay_awgn, p_opt, distance, fading)
    trans_time = cal_trans_time(data_size, trans_rate)
    trans_energy = cal_trans_energy(trans_time, p_opt)
    comp_time = task.time_constraint[i] - trans_time
    f = cal_f(comp_time, workload_size)
    f_opt = get_fix_value(f, device.relay_min_f, device.relay_max_f)
    comp_time = cal_comp_time(workload_size, f_opt)
    comp_energy = cal_comp_energy(device.relay_k, f_opt, workload_size)
    return p_opt, f_opt, trans_time + comp_time, trans_energy + comp_energy


def cc_computing_rest_bit(i, j, task, env, device, fading):
    p, f, t, energy = cc_computing_per_device(i, j, task, env, device, fading)
    if t < task.time_constraint[i]:
        return 0, 0
    comp_time = cal_comp_time(task.workload_size[i], f)
    trans_time = t - comp_time
    rest_data_size = 0
    rest_workload_size = 0
    if trans_time <= task.time_constraint[i]:
        exceed_comp_time = t - task.time_constraint[i]
        ratio = exceed_comp_time / comp_time
        rest_workload_size = ratio * task.workload_size[i]
    else:
        ratio = task.time_constraint[i] / t
        trans_bit = int(task.data_size[i] * ratio)
        rest_data_size = task.data_size[i] - trans_bit
        rest_workload_size = int(task.workload_size[i] * (1 - ratio))
    return rest_data_size, rest_workload_size


def cc_computing_true_energy(i, j, task, env, device, fading):
    p, f, t, energy = cc_computing_per_device(i, j, task, env, device, fading)
    if t <= task.time_constraint[i]:
        pass
    else:
        comp_time = task.workload_size[i] / f
        tran_time = t - comp_time
        if comp_time <= task.time_constraint[i]:
            tran_energy = p * tran_time
            _comp_time = task.time_constraint[i] - tran_time
            rate = _comp_time / comp_time
            compt_energy = cal_comp_energy(device.relay_k, f, int(task.workload_size[i] * rate))
            energy = tran_energy + compt_energy
        else:
            energy = task.time_constraint[i] * p
    return energy


def df_computing_per_device(i, j, task, env, device, fading):
    """
    :param i:
    :param j:
    :param task:
    :param env:
    :param device:
    :return: p_m, p_r, energy, time
    """
    mobile_topo = task.mobile_topology[i]
    relay_topo = task.relay_topology[j]
    edge_topo = task.edge_topology
    distance = cal_distance(mobile_topo, edge_topo)
    hat_b = cal_channel_gain(distance, fading) / math.pow(env.edge_awgn, 2)
    distance = cal_distance(mobile_topo, relay_topo)
    hat_a = cal_channel_gain(distance, fading) / math.pow(env.relay_awgn, 2)
    distance = cal_distance(relay_topo, edge_topo)
    hat_c = cal_channel_gain(distance, fading) / math.pow(env.edge_awgn, 2)
    comp_time = task.workload_size[i] / device.edge_f * utils.FLOAT_CONFIG
    tmp = task.data_size[i] / (env.bandwidth * (task.time_constraint[i] - comp_time))
    C_min = math.pow(2, tmp) - 1
    p_m_opt = C_min / hat_a
    p_r_opt = C_min * (hat_a - hat_b) / (hat_a * hat_c)
    p_m_opt = get_fix_value(p_m_opt, device.mobile_min_p, device.mobile_max_p)
    p_r_opt = get_fix_value(p_r_opt, device.relay_min_p, device.relay_max_p)

    def cal_df_trans_rate(p_s, p_r):
        channel_1 = 1 + p_s * hat_a
        channel_2 = 1 + p_s * hat_b + p_r * hat_c
        channel = min([channel_1, channel_2])
        rate = env.bandwidth / 2 * math.log2(channel)
        return rate

    def cal_df_trans_energy(p_s, p_r, time):
        energy = (p_s + p_r) * time / 2
        return energy

    trans_rate = cal_df_trans_rate(p_m_opt, p_r_opt)
    trans_time = cal_trans_time(task.data_size[i], trans_rate)
    energy = cal_df_trans_energy(p_m_opt, p_r_opt, trans_time)
    comp_time = cal_comp_time(task.workload_size[i], device.edge_f)
    return p_m_opt, p_r_opt, energy, trans_time + comp_time


def df_computing_rest_bit(i, j, task, env, device, fading):
    p_m, p_r, energy, t = df_computing_per_device(i, j, task, env, device, fading)
    rest_data_size = 0
    rest_workload_size = 0
    if t <= task.time_constraint[i]:
        pass
    else:
        comp_time = task.workload_size[i] / device.edge_f
        trans_time = t - comp_time
        if trans_time <= task.time_constraint[i]:
            rest_data_size = 0
            rest_workload_size = 0
        else:
            ratio = task.time_constraint[i] / trans_time
            rest_data_size = int(task.data_size[i] * (1 - ratio))
            rest_workload_size = int(task.workload_size[i] * (1 - ratio))
    return rest_data_size, rest_workload_size


def df_computng_true_energy(i, j, task, env, device):
    p_m, p_r, energy, t = df_computing_per_device(i, j, task, env, device)
    if t <= task.time_constraint[i]:
        pass
    else:
        comp_time = task.workload_size[i] / device.edge_f
        tran_time = t - comp_time
        tran_time = min([tran_time, task.time_constraint[i]])
        energy = tran_time * (p_m + p_r) / 2
    return energy


def get_local_computing_info(task, device):
    local_computing = CalInfo()
    workload_size = task.workload_size
    time_constraint = task.time_constraint
    for i in range(task.mobile_num):
        f, energy, t = local_computing_per_device(time_constraint[i], workload_size[i], device.mobile_min_f,
                                                  device.mobile_max_f, device.mobile_k)
        local_computing.f.append(f)
        local_computing.energy.append(energy)
        local_computing.time.append(t)
    return local_computing


def get_direct_edge_computing_info(task, env, device, fading):
    dec_computing = CalInfo()
    for i in range(task.mobile_num):
        energy, time, p = direct_edge_computing_per_device(i, task, env, device, fading)
        dec_computing.energy.append(energy)
        dec_computing.time.append(time)
        dec_computing.mobile_power.append(p)
    return dec_computing


def get_cc_computing_info(task, env, device, fading):
    cc_computing = CalInfo()
    for i in range(task.mobile_num):
        tmp_p = []
        tmp_f = []
        tmp_time = []
        tmp_energy = []
        for j in range(task.relay_num):
            p, f, time, energy = cc_computing_per_device(i, j, task, env, device, fading)
            tmp_p.append(p)
            tmp_f.append(f)
            tmp_time.append(time)
            tmp_energy.append(energy)
        cc_computing.mobile_power.append(tmp_p)
        cc_computing.f.append(tmp_f)
        cc_computing.time.append(tmp_time)
        cc_computing.energy.append(tmp_energy)
    return cc_computing


def get_df_computing_info(task, env, device, fading):
    df_computing = CalInfo()
    for i in range(task.mobile_num):
        tmp_p_m = []
        tmp_p_r = []
        tmp_energy = []
        tmp_time = []
        for j in range(task.relay_num):
            p_m, p_r, energy, time = df_computing_per_device(i, j, task, env, device, fading)
            tmp_p_m.append(p_m)
            tmp_p_r.append(p_r)
            tmp_energy.append(energy)
            tmp_time.append(time)
        df_computing.mobile_power.append(tmp_p_m)
        df_computing.relay_power.append(tmp_p_r)
        df_computing.energy.append(tmp_energy)
        df_computing.time.append(tmp_time)
    return df_computing


def cal_object_value(mode, match, computing_info, weight):
    """
    暂时使用，后面换成动态的需要进行修改
    :param mode:
    :param match:
    :param computing_info:
    :return:
    """
    total_energy = 0
    max_energy = 0
    for i in range(len(mode)):
        if mode[i] <= 2:
            tmp = computing_info[int(mode[i] - 1)].energy[i]
        else:
            tmp = computing_info[int(mode[i] - 1)].energy[i][match[i]]
        max_energy = max([max_energy, tmp / weight[i]])
        total_energy += tmp / weight[i]
    # while max_energy < 0.1:
    #     max_energy *= 10
    #     if max_energy <= 0:
    #         break
    return total_energy, max_energy


def cal_result_unpredict(mode, match, computing_info, weight, true_bandwidth,
                         task, env, device, fading):
    total_energy = 0
    max_energy = 0
    rest_datasize = [0 for _ in range(len(mode))]
    rest_workload = [0 for _ in range(len(mode))]
    for i in range(len(mode)):
        if mode[i] == 1:
            energy = computing_info[int(mode[i] - 1)].energy[i]
            f = computing_info[0].f[i]
            w = task.workload_size[i]
            t = w / f
            if t > task.time_constraint[i]:
                process = f * task.time_constraint[i]  # the process bit
                ratio = process / w  # the process rate, 1-ratio means the un-process
                rest_datasize[i] = int((1 - ratio) * task.data_size[i])
                rest_workload[i] = int((1 - ratio) * task.workload_size[i])
        elif mode[i] == 2:
            p = computing_info[1].mobile_power[i]
            distance = cal_distance(task.mobile_topology[i], task.edge_topology)
            trans_rate = cal_trans_rate(true_bandwidth, env.edge_awgn, p, distance, fading)
            trans_time = cal_trans_time(task.data_size[i], trans_rate)
            if trans_time <= task.time_constraint[i]:
                energy = trans_time * p
                rest_datasize[i] = 0
                rest_workload[i] = 0
            else:
                energy = p * task.time_constraint[i]
                ratio = task.time_constraint[i] / trans_time
                rest_datasize[i] = int((1 - ratio) * task.data_size[i])
                rest_workload[i] = int((1 - ratio) * task.workload_size[i])
        elif mode[i] == 3:
            r = match[i]
            p = computing_info[2].mobile_power[i][r]
            f = computing_info[2].f[i][r]
            distance = cal_distance(task.mobile_topology[i], task.relay_topology[r])
            trans_rate = cal_trans_rate(true_bandwidth, env.relay_awgn, p, distance, fading)
            trans_time = cal_trans_time(task.data_size[i], trans_rate)
            if trans_time <= task.time_constraint[i]:
                energy = cal_trans_energy(trans_time, p) + cal_comp_energy(device.relay_k, f, task.workload_size[i])
                rest_datasize[i] = 0
                rest_workload[i] = 0
            else:
                ratio = task.time_constraint[i] / trans_time
                rest_datasize[i] = int((1 - ratio) * task.data_size[i])
                rest_workload[i] = int((1 - ratio) * task.workload_size[i])
                energy = cal_trans_energy(task.time_constraint[i], p) + cal_comp_energy(device.relay_k, f,
                                                                                        int(task.workload_size[
                                                                                                i] * ratio))
        else:
            r = match[i]
            p_m = computing_info[3].mobile_power[i][r]
            p_r = computing_info[3].relay_power[i][r]
            distance = cal_distance(task.mobile_topology[i], task.edge_topology)
            channel_gain = cal_channel_gain(distance, fading)
            A = channel_gain * p_m / math.pow(env.edge_awgn, 2)
            distance = cal_distance(task.mobile_topology[i], task.relay_topology[r])
            channel_gain = cal_channel_gain(distance, fading)
            B = channel_gain * p_m / math.pow(env.relay_awgn, 2)
            distance = cal_distance(task.relay_topology[r], task.edge_topology)
            channel_gain = cal_channel_gain(distance, fading)
            C = channel_gain * p_r / math.pow(env.edge_awgn, 2)
            con = min([A, B+C])
            trans_rate = true_bandwidth / 2 * math.log2(1+con)
            trans_time = cal_trans_time(task.data_size[i], trans_rate)
            energy = (p_m + p_r) * trans_time / 2
            if trans_time <= task.time_constraint[i]:
                energy = (p_m+p_r) * trans_time / 2
                rest_datasize[i] = 0
                rest_workload[i] = 0
            else:
                energy = (p_m+p_r) * task.time_constraint[i] / 2
                ratio = task.time_constraint[i] / trans_time
                rest_datasize[i] = int((1-ratio)*task.data_size[i])
                rest_workload[i] = int((1-ratio)*task.workload_size[i])
        total_energy += energy / weight[i]
        max_energy = max([max_energy, energy / weight[i]])
    return total_energy, max_energy, rest_datasize, rest_workload


def cal_result_predict(mode, match, computing_info, weight, true_fading,
                         task, env, device):
    computing_info = get_computing_info(task, env, device, true_fading)
    total_energy, max_energy = cal_object_value(mode, match, computing_info, weight)
    rest_datasize = [0 for _ in range(task.mobile_num)]
    rest_workload = [0 for _ in range(task.mobile_num)]
    return total_energy, max_energy, rest_datasize, rest_workload


def get_computing_info(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    return computing_info


def otca(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]

    default_mode = [-1 for _ in range(task.mobile_num)]
    """
    1. local computing
    2. direct edge computing
    3. cc computing
    4. df computing
    """
    default_energy = [1 for _ in range(task.mobile_num)]
    for i in range(task.mobile_num):
        if local_computing.time[i] <= task.time_constraint[i]:
            default_energy[i] = local_computing.energy[i]
            default_mode[i] = 1
        if dec_computing.time[i] <= task.time_constraint[i]:
            if dec_computing.energy[i] < default_energy[i]:
                default_energy[i] = dec_computing.energy[i]
                default_mode[i] = 2
    matrix_mode = [[-1] * task.relay_num for _ in range(task.mobile_num)]
    matrix_energy = [[0] * task.relay_num for _ in range(task.mobile_num)]
    weight = cal_weight(computing_info, task.mobile_num, task=task)
    for i in range(task.mobile_num):
        for j in range(task.relay_num):
            if cc_computing.time[i][j] <= task.time_constraint[i]:
                if cc_computing.energy[i][j] < default_energy[i]:
                    matrix_energy[i][j] = 1 - default_energy[i] + cc_computing.energy[i][j]
                    matrix_mode[i][j] = 3
            if df_computing.time[i][j] <= task.time_constraint[i] and \
                    df_computing.energy[i][j] < default_energy[i]:
                if matrix_mode[i][j] == 0 or \
                        (df_computing.energy[i][j] < cc_computing.energy[i][j]):
                    matrix_energy[i][j] = 1 - default_energy[i] + df_computing.energy[i][j]
                    matrix_mode[i][j] = 4
            matrix_energy[i][j] = matrix_energy[i][j] / weight[i]
    from munkres import Munkres
    m = Munkres()
    indexes = m.compute(matrix_energy)
    final_mode = default_mode.copy()
    final_match = [-1] * task.mobile_num
    for row, column in indexes:
        if matrix_mode[row][column] != -1:
            final_mode[row] = matrix_mode[row][column]
            final_match[row] = column
    for m in final_mode:
        if m == -1:
            print("final_mode..., canot matching.....")
    return final_mode, final_match, computing_info


def cal_weight(computing_info, mobile_num, task=None):
    weight = [0 for _ in range(mobile_num)]
    if task is None:
        for i in range(mobile_num):
            weight[i] = [computing_info[0].energy[i] + computing_info[1].energy[i]]
            weight[i] += sum(computing_info[2].energy[i]) / len(computing_info[2].energy[i])
            weight[i] += sum(computing_info[3].energy[i]) / len(computing_info[3].energy[i])
    else:
        for i in range(mobile_num):
            weight[i] = (1-(task.data_size[i] / task.max_datasize)) + (1-(task.workload_size[i] / task.max_workloadsize)) ** 3
    return weight


def cal_pred_through(through, t, front=20):
    if front < 0:
        return through[t-1]
    pred_list = through[t-front:t]
    max_rate = max(through)
    fading = 0
    for rate in pred_list:
        fading = fading + math.pow(2, float(rate)/max_rate) - 1
    return fading / len(pred_list)


def oecaa(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]

    default_mode = [-1 for _ in range(task.mobile_num)]
    """
    1. local computing
    2. direct edge computing
    3. cc computing
    4. df computing
    """
    default_energy = [utils.max_energy for _ in range(task.mobile_num)]
    for i in range(task.mobile_num):
        if local_computing.time[i] <= task.time_constraint[i]:
            default_energy[i] = local_computing.energy[i]
            default_mode[i] = 1
        if dec_computing.time[i] <= task.time_constraint[i]:
            if dec_computing.energy[i] < default_energy[i]:
                default_energy[i] = dec_computing.energy[i]
                default_mode[i] = 2
    matrix_mode = [[-1] * task.relay_num for _ in range(task.mobile_num)]
    matrix_energy = [[1] * task.relay_num for _ in range(task.mobile_num)]
    weight = cal_weight(computing_info, task.mobile_num, task=task)
    for i in range(task.mobile_num):
        for j in range(task.relay_num):
            if cc_computing.time[i][j] <= task.time_constraint[i]:
                if cc_computing.energy[i][j] < default_energy[i]:
                    matrix_energy[i][j] = cc_computing.energy[i][j]
                    matrix_mode[i][j] = 3
            if df_computing.time[i][j] <= task.time_constraint[i] and \
                    df_computing.energy[i][j] < default_energy[i]:
                if matrix_mode[i][j] == -1 or \
                        df_computing.energy[i][j] < cc_computing.energy[i][j]:
                    matrix_energy[i][j] = df_computing.energy[i][j]
                    matrix_mode[i][j] = 4
            matrix_energy[i][j] = matrix_energy[i][j] / weight[i]

    min_max = utils.Min_max()
    default_energy[i] = default_energy[i] / weight[i]
    # print(default_energy)
    # print(task.workload_size)
    # print([default_energy[i]/task.workload_size[i] for i in range(task.mobile_num)])
    # exit(242)
    match = min_max.start(np.array(matrix_energy), np.array(default_energy))
    final_mode = default_mode.copy()
    final_match = [-1] * task.mobile_num
    for i in range(task.mobile_num):
        if match[i] == -1:
            continue
        j = int(match[i])
        if matrix_mode[i][j] != -1:
            final_mode[i] = matrix_mode[i][j]
            final_match[i] = j

    return final_mode, final_match, computing_info


def dqn(task, env, device, fading, min_max=True):
    import A2C_
    import torch
    if min_max:
        checkpoint = torch.load(utils.dqn_min_max_model_pth)
    else:
        checkpoint = torch.load(utils.dqn_model_pth)
    # print("9 + task.relay_num*2: ", 9 + task.relay_num*2)
    policy_model = A2C_.Policy(1, task.relay_num, 9 + task.relay_num*2)
    policy_model.load_state_dict(checkpoint['policy_net'])
    match_model = A2C_.Match(1, task.relay_num, 9 + task.relay_num*2)
    match_model.load_state_dict(checkpoint['match_net'])
    policy_list = []
    match_list = []
    for i in range(task.mobile_num):
        state = utils.get_state_by_index(i, task, env, fading)
        policy = policy_model(torch.FloatTensor(state))
        policy_list.append(policy.detach().numpy())
        match = match_model(torch.FloatTensor(state))
        match_list.append(match.detach().numpy())
    policy = [0 for _ in range(task.mobile_num)]
    match = [-1 for _ in range(task.mobile_num)]
    relay_used = [False for _ in range(task.relay_num)]
    for i in range(task.mobile_num):
        p = np.argmax(policy_list[i])
        if p < 2:
            policy[i] = p + 1
            continue
        flag = False
        sort_j = np.argsort(np.multiply(match_list[i], -1))  # 倒叙索引
        for j in sort_j:
            if not relay_used[j]:
                match[i] = j
                relay_used[j] = True
                policy[i] = p + 1
                flag = True
            if flag:
                break
        if not flag:
            p = np.argmax(policy_list[i][:2])
            policy[i] = p + 1
    compute_info = get_computing_info(task, env, device, fading)
    return policy, match, compute_info


def comp(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    default_mode = [-1 for _ in range(task.mobile_num)]
    """
    1. local computing
    2. direct edge computing
    3. cc computing
    4. df computing
    """
    default_energy = [1 for _ in range(task.mobile_num)]
    default_match = [-1 for _ in range(task.mobile_num)]
    relay_used = [0 for _ in range(task.relay_num)]
    for i in range(task.mobile_num):
        relay_idx = -1
        if sum(relay_used) != task.relay_num:
            relay_idx = random.randint(0, task.relay_num-1)
            for _ in range(task.relay_num):
                relay_idx = (relay_idx+1) % task.relay_num
                if relay_used[relay_idx] != 1:
                    break
            if relay_used[relay_idx] == 1:
                relay_idx = -1
        if local_computing.time[i] <= task.time_constraint[i]:
            default_energy[i] = local_computing.energy[i]
            default_mode[i] = 1
        if dec_computing.time[i] <= task.time_constraint[i]:
            if dec_computing.energy[i] < default_energy[i]:
                default_energy[i] = dec_computing.energy[i]
                default_mode[i] = 2
        if relay_idx != -1:
            if cc_computing.energy[i][relay_idx] < default_energy[i]:
                default_energy[i] = cc_computing.energy[i][relay_idx]
                default_mode[i] = 3
                default_match[i] = relay_idx
                relay_used[relay_idx] = 1
            if df_computing.energy[i][relay_idx] < default_energy[i]:
                default_energy[i] = df_computing.energy[i][relay_idx]
                default_mode[i] = 4
                default_match[i] = relay_idx
                relay_used[relay_idx] = 1
    return default_mode, default_match, computing_info


def local(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    default_mode = [1 for _ in range(task.mobile_num)]
    """
    1. local computing
    2. direct edge computing
    3. cc computing
    4. df computing
    """
    default_match = [-1 for _ in range(task.mobile_num)]
    return default_mode, default_match, computing_info

def cc(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    default_mode = [3 for _ in range(task.mobile_num)]
    """
    1. local computing
    2. direct edge computing
    3. cc computing
    4. df computing
    """
    # default_match = [min([i, task.relay_num-1]) for i in range(task.mobile_num)]
    default_match = [-1 for _ in range(task.mobile_num)]
    relay_used = [False for _ in range(task.relay_num)]
    for i in range(task.mobile_num):
        distance = math.inf
        relay_idx = -1
        for j in range(task.relay_num):
            if relay_used[j]:
                continue
            mobile_topo = task.mobile_topology[i]
            relay_topo = task.relay_topology[j]
            _distance = cal_distance(mobile_topo, relay_topo)
            if _distance < distance:
                distance = _distance
                relay_idx = j
        default_match[i] = relay_idx
        relay_used[relay_idx] = True
    return default_mode, default_match, computing_info

def dec(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    default_mode = [2 for _ in range(task.mobile_num)]
    """
    1. local computing
    2. direct edge computing
    3. cc computing
    4. df computing
    """
    default_match = [-1 for _ in range(task.mobile_num)]
    return default_mode, default_match, computing_info

def df(task, env, device, fading):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device, fading)
    cc_computing = get_cc_computing_info(task, env, device, fading)
    df_computing = get_df_computing_info(task, env, device, fading)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    default_mode = [4 for _ in range(task.mobile_num)]
    """
    1. local computing
    2. direct edge computing
    3. cc computing
    4. df computing
    """
    # default_match = [min([i, task.relay_num-1]) for i in range(task.mobile_num)]
    default_match = [-1 for _ in range(task.mobile_num)]
    relay_used = [False for _ in range(task.relay_num)]
    for i in range(task.mobile_num):
        distance = math.inf
        relay_idx = -1
        for j in range(task.relay_num):
            if relay_used[j]:
                continue
            mobile_topo = task.mobile_topology[i]
            relay_topo = task.relay_topology[j]
            _distance = cal_distance(mobile_topo, relay_topo)
            if _distance < distance:
                distance = _distance
                relay_idx = j
        default_match[i] = relay_idx
        relay_used[relay_idx] = True
    return default_mode, default_match, computing_info

def rand(task, env, device, fading):
    compute_info = get_computing_info(task, env, device, fading)
    relay_used = [False for _ in range(task.relay_num)]
    mode = [0 for _ in range(task.mobile_num)]
    match = [-1 for _ in range(task.mobile_num)]
    for i in range(task.mobile_num):
        A = compute_info[0].energy[i]
        B = compute_info[1].energy[i]
        C = sum(compute_info[2].energy[i]) / task.relay_num
        D = sum(compute_info[3].energy[i]) / task.relay_num
        E = A + B + C + D
        r_a = (E - A) / (3 * E)
        r_b = (E - B) / (3 * E) + r_a
        r_c = (E - C) / (3 * E) + r_b
        r_d = (E - D) / (3 * E) + r_c
        t = random.random()
        if t <= r_a:
            mode[i] = 1
        elif t <= r_b:
            mode[i] = 2
        else:
            flag = False
            k = random.randint(0, task.relay_num-1)
            for _ in range(task.relay_num):
                if not relay_used[k]:
                    if t <= r_c:
                        mode[i] = 3
                    else:
                        mode[i] = 4
                    match[i] = k
                    relay_used[k] = True
                    flag = True
                if flag:
                    break
                k = (k+1) % task.relay_num
    return mode, match, compute_info


def ns3_process(tx_time, mode, match, task, env, device, weight):
    total_energy, max_energy = 0, 0
    rest_datasize = [0 for _ in range(task.mobile_num)]
    rest_workloadsize = [0 for _ in range(task.mobile_num)]
    for i in range(task.mobile_num):
        if mode[i] == 1:
            _, energy, t = local_computing_per_device(task.time_constraint[i], task.workload_size[i], device.mobile_min_f, device.mobile_max_f, device.mobile_k)
            if t <= task.time_constraint[i]:
                rest_datasize[i] = 0
                rest_workloadsize[i] = 0
            else:
                ratio = task.time_constraint[i] / t
                rest_datasize[i] = int((1-ratio)*task.data_size[i])
                rest_workloadsize[i] = int((1-ratio)*task.workload_size[i])
        elif mode[i] == 2:
            _, t, p = direct_edge_computing_per_device(i, task, env, device)
            if tx_time[i] <= 0 or tx_time[i] >= task.time_constraint[i]:
                tx_time[i] = t - task.workload_size[i] / device.edge_f
            energy = tx_time[i] * p
        elif mode[i] == 3:
            r = match[i]
            p, f, t, energy = cc_computing_per_device(i, r, task, env, device)
            if tx_time[i] <= 0 or tx_time[i] >= task.time_constraint[i]:
                tx_time[i] = t - task.workload_size[i] / f
            comp_time = task.time_constraint[i] - tx_time[i]
            f = cal_f(comp_time, task.workload_size[i], device.relay_min_f, device.relay_max_f)
            energy = p * tx_time[i] + cal_comp_energy(device.relay_k, f, task.workload_size[i])
        else:
            r = match[i]
            p_m, p_r, energy, t = df_computing_per_device(i, r, task, env, device)
            if tx_time[i] <= 0 or tx_time[i] >= task.time_constraint[i]:
                tx_time[i] = t - task.workload_size[i] / device.edge_f
            energy = (p_m+p_r) * tx_time[i] / 2
        total_energy += energy / weight[i]
        max_energy = max([max_energy, energy/weight[i]])
    return total_energy, max_energy, rest_datasize, rest_workloadsize


if __name__ == '__main__':
    mobile_num = 10
    relay_num = 10
    task = utils.Task(0.2, 0.5, 10, 25)
    bandwidth = 10 * utils.bandwidth_unit
    env = utils.Environment(bandwidth=bandwidth)
    device = utils.Device()
    task.init_env(mobile_num, relay_num)
    task.set_time_constraint(0.05)

    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device)
    cc_computing = get_cc_computing_info(task, env, device)
    df_computing = get_df_computing_info(task, env, device)
    print("\t{}\t{}".format("min", "max"))
    print("local:\t{:.3f}\t{:.3f}".format(min(local_computing.energy) * 1000, max(local_computing.energy) * 1000))
    print("direct:\t{:.3f}\t{:.3f}".format(min(dec_computing.energy) * 1000, max(dec_computing.energy) * 1000))
    print("cc:\t{:.3f}\t{:.3f}".format(min(min(cc_computing.energy)) * 1000, max(max(cc_computing.energy)) * 1000))
    print("df:\t{:.3f}\t{:.3f}".format(min(min(df_computing.energy)) * 1000, max(max(df_computing.energy)) * 1000))


    mode, match, computing_info = otca(task, env, device)
    weight = cal_weight(computing_info, mobile_num, task)
    total_energy, max_energy, _, _ = cal_result_unpredict(mode, match, computing_info, weight, int(
            bandwidth), task, env, device)
    print(total_energy, max_energy)
    print(mode)
    print("*"*40)
    # task.set_time_constraint(2)
    # env.bandwidth = 30 * utils.bandwidth_unit
    # env.pathloss = 0

    # local_computing = get_local_computing_info(task, device)
    # dec_computing = get_direct_edge_computing_info(task, env, device)
    # cc_computing = get_cc_computing_info(task, env, device)
    # df_computing = get_df_computing_info(task, env, device)
    # print("\t{}\t{}".format("min", "max"))
    # print("local:\t{:.3f}\t{:.3f}".format(min(local_computing.energy) * 1000, max(local_computing.energy) * 1000))
    # print("direct:\t{:.3f}\t{:.3f}".format(min(dec_computing.energy) * 1000, max(dec_computing.energy) * 1000))
    # print("cc:\t{:.3f}\t{:.3f}".format(min(min(cc_computing.energy)) * 1000, max(max(cc_computing.energy)) * 1000))
    # print("df:\t{:.3f}\t{:.3f}".format(min(min(df_computing.energy)) * 1000, max(max(df_computing.energy)) * 1000))
    env = utils.Environment(bandwidth=bandwidth)
    mode, match, computing_info = otca(task, env, device)
    weight = cal_weight(computing_info, mobile_num, task)
    total_energy, max_energy, _, _ = cal_result_unpredict(mode, match, computing_info, weight, int(bandwidth), task, env, device)
    print(total_energy, max_energy)
    print(mode)
    print("*" * 40)