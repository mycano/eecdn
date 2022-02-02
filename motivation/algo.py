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
    t = workload / f
    return t


def cal_trans_rate(b, awgn, power, distance, pathloss):
    channel_conf = math.pow(distance, -pathloss)
    rate = b * math.log2(1 + power * channel_conf / (b * awgn))
    return rate


def cal_distance(topo_1, topo_2):
    distance = math.pow(topo_1[0] - topo_2[0], 2)
    distance += math.pow(topo_1[1] - topo_2[1], 2)
    distance = math.sqrt(distance)
    return distance


def cal_trans_time(datasize, rate):
    time = datasize / rate
    return time


def cal_trans_energy(time, p):
    energy = time * p
    return energy


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


def direct_edge_computing_per_device(i, task, env, device):
    """
    :param i: mobile device id
    :param task:
    :param env:
    :param device:
    :return: energy, time, power
    """

    def cal_trans_power(trans_time, datasize, b, awgn, distance, pathloss, min_p, max_p):
        min_trans_rate = datasize / trans_time
        p = b * awgn / math.pow(distance, -pathloss) * math.pow(2, min_trans_rate / b) - 1
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
        p = cal_trans_power(trans_time, data_size, env.bandwidth, env.edge_awgn, distance, env.pathloss,
                            device.mobile_min_p, device.mobile_max_p)
        rate = cal_trans_rate(env.bandwidth, env.edge_awgn, p, distance, env.pathloss)
        trans_time = cal_trans_time(data_size, rate)
        trans_energy = cal_trans_energy(trans_time, p)
    else:
        trans_time = time_constraint
        trans_energy = utils.max_energy
        p = device.mobile_max_p
    return trans_energy, trans_time + comp_time, p


def dec_computing_rest_bit(i, task, env, device):
    _, t, p = direct_edge_computing_per_device(i, task, env, device)
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
        trans_data_bit = cal_trans_rate(env.bandwidth, env.edge_awgn, p, distance, env.pathloss)
        rest_data_size = task.data_size[i] - trans_data_bit
        rest_workloadsize = task.workload_size[i]
    return rest_data_size, rest_workloadsize


def dec_computing_true_energy(i, task, env, device):
    energy, t, p = direct_edge_computing_per_device(i, task, env, device)
    if t <= task.time_constraint[i]:
        pass
    else:
        comp_time = task.workload_size[i] / device.edge_f
        if t - comp_time <= task.time_constraint[i]:
            pass
        else:
            energy = p * task.time_constraint[i]
    return energy


def cc_computing_per_device(i, j, task, env, device):
    """
    :param i:
    :param j:
    :param task:
    :param env:
    :param device:
    :return: power, f, time, energy
    """
    mobile_topo = task.mobile_topology[i]
    relay_topo = task.relay_topology[j]
    distance = cal_distance(mobile_topo, relay_topo)
    channel_conf = math.pow(distance, -env.pathloss)
    N = channel_conf / (env.relay_awgn * env.bandwidth)

    def cc_fun(K, N=N, d=task.data_size[i], t=task.time_constraint[i], B=env.bandwidth, k=device.relay_k,
               p_min=device.mobile_min_p):
        v_1 = d / (B * math.log2(K))
        m_sqrt = math.log(2) * K * math.pow(math.log2(K), 2) / (2 * N * k * math.log2(1 + p_min * N))
        v_2 = math.pow(m_sqrt, 1 / 3)
        v = v_1 + v_2 - t
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
    K_max = 1 + device.mobile_max_p * N
    K_opt = binary_search(K_min, K_max)
    p_opt = get_fix_value((K_opt - 1) / N, device.mobile_min_p, device.mobile_max_p)
    trans_rate = cal_trans_rate(env.bandwidth, env.relay_awgn, p_opt, distance, env.pathloss)
    trans_time = cal_trans_time(data_size, trans_rate)
    trans_energy = cal_trans_energy(trans_time, p_opt)
    # tmp = math.log(2) * K_opt * math.pow(math.log2(K_opt), 2) / (2*N*device.relay_k*math.log2(1+device.mobile_min_p*N))
    # f = math.pow(tmp, 1/3)
    comp_time = task.time_constraint[i] - trans_time
    f = cal_f(comp_time, workload_size)
    f_opt = get_fix_value(f, device.relay_min_f, device.relay_max_f)
    comp_time = cal_comp_time(workload_size, f_opt)
    comp_energy = cal_comp_energy(device.relay_k, f_opt, workload_size)
    return p_opt, f_opt, trans_time + comp_time, trans_energy + comp_energy


def cc_computing_rest_bit(i, j, task, env, device):
    p, f, t, energy = cc_computing_per_device(i, j, task, env, device)
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


def cc_computing_true_energy(i, j, task, env, device):
    p, f, t, energy = cc_computing_per_device(i, j, task, env, device)
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


def df_computing_per_device(i, j, task, env, device):
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
    pathloss = env.pathloss
    distance = cal_distance(mobile_topo, edge_topo)
    hat_b = math.pow(distance, -pathloss) / env.edge_awgn / (env.bandwidth/2)
    distance = cal_distance(mobile_topo, relay_topo)
    hat_a = math.pow(distance, -pathloss) / env.relay_awgn / (env.bandwidth/2)
    distance = cal_distance(relay_topo, edge_topo)
    hat_c = math.pow(distance, -pathloss) / env.edge_awgn / (env.bandwidth/2)
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
        rate = env.bandwidth * math.log2(channel)
        return rate

    def cal_df_trans_energy(p_s, p_r, time):
        energy = (p_s + p_r) * time / 2
        return energy

    trans_rate = cal_df_trans_rate(p_m_opt, p_r_opt)
    trans_time = cal_trans_time(task.data_size[i], trans_rate)
    energy = cal_df_trans_energy(p_m_opt, p_r_opt, trans_time)
    comp_time = cal_comp_time(task.workload_size[i], device.edge_f)
    return p_m_opt, p_r_opt, energy, trans_time + comp_time


def df_computing_rest_bit(i, j, task, env, device):
    p_m, p_r, energy, t = df_computing_per_device(i, j, task, env, device)
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


def get_direct_edge_computing_info(task, env, device):
    dec_computing = CalInfo()
    for i in range(task.mobile_num):
        energy, time, p = direct_edge_computing_per_device(i, task, env, device)
        dec_computing.energy.append(energy)
        dec_computing.time.append(time)
        dec_computing.mobile_power.append(p)
    return dec_computing


def get_cc_computing_info(task, env, device):
    cc_computing = CalInfo()
    for i in range(task.mobile_num):
        tmp_p = []
        tmp_f = []
        tmp_time = []
        tmp_energy = []
        for j in range(task.relay_num):
            p, f, time, energy = cc_computing_per_device(i, j, task, env, device)
            tmp_p.append(p)
            tmp_f.append(f)
            tmp_time.append(time)
            tmp_energy.append(energy)
        cc_computing.mobile_power.append(tmp_p)
        cc_computing.f.append(tmp_f)
        cc_computing.time.append(tmp_time)
        cc_computing.energy.append(tmp_energy)
    return cc_computing


def get_df_computing_info(task, env, device):
    df_computing = CalInfo()
    for i in range(task.mobile_num):
        tmp_p_m = []
        tmp_p_r = []
        tmp_energy = []
        tmp_time = []
        for j in range(task.relay_num):
            p_m, p_r, energy, time = df_computing_per_device(i, j, task, env, device)
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
                         task, env, device):
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
            trans_rate = cal_trans_rate(true_bandwidth, env.edge_awgn, p, distance, env.pathloss)
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
            trans_rate = cal_trans_rate(true_bandwidth, env.relay_awgn, p, distance, env.pathloss)
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
            channel_conf = math.pow(distance, -env.pathloss)
            A = channel_conf * p_m / ( true_bandwidth * env.edge_awgn / 2)
            channel_conf = math.pow(cal_distance(task.mobile_topology[i], task.relay_topology[r]), -env.pathloss)
            B = channel_conf * p_m / ( true_bandwidth * env.relay_awgn / 2)
            channel_conf = math.pow(cal_distance(task.relay_topology[r], task.edge_topology), -env.pathloss)
            C = channel_conf * p_r / ( true_bandwidth * env.edge_awgn / 2)
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


def cal_result_predict(mode, match, computing_info, weight, true_bandwidth,
                         task, env, device):
    env.bandwidth = true_bandwidth
    computing_info = get_computing_info(task, env, device)
    total_energy, max_energy = cal_object_value(mode, match, computing_info, weight)
    rest_datasize = [0 for _ in range(task.mobile_num)]
    rest_workload = [0 for _ in range(task.mobile_num)]
    return total_energy, max_energy, rest_datasize, rest_workload


def get_computing_info(task, env, device):
    local_computing = get_local_computing_info(task, device)
    dec_computing = get_direct_edge_computing_info(task, env, device)
    cc_computing = get_cc_computing_info(task, env, device)
    df_computing = get_df_computing_info(task, env, device)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    return computing_info
