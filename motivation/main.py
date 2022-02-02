'''
Author: myyao
Date: 2022-02-02 12:52:00
Description: 
'''
# coding: utf-8
# @Time    : 2021/2/19 9:09
# @Author  : myyao
# @FileName: motivation.py
# @Software: PyCharm
# @e-mail  : myyaocn@outlook.com
# description : the experience of motivation

import utils
import algo
import random
random.seed(10)


def cal_energy(mode, match, computingInfo):
    energy = [0 for _ in range(len(mode))]
    for i in range(len(mode)):
        if mode[i] == 1:
            energy[i] = computingInfo[0].energy[i]
        if mode[i] == 2:
            energy[i] = computingInfo[1].energy[i]
        if mode[i] == 3:
            j = match[i]
            energy[i] = computingInfo[2].energy[i][j]
        if mode[i] == 4:
            j = match[i]
            energy[i] = computingInfo[3].energy[i][j]
    return energy


def print_computing_info(computing_info, relay_num=0):
    print("_"*50)
    print("local computing: ", computing_info[0].energy[0])
    print("direct: [{}, {}]".format(min(computing_info[1].energy), max(computing_info[1].energy)))
    print("direct power: [{}, {}]".format(min(computing_info[1].mobile_power), max(computing_info[1].mobile_power)))
    print("*"*50)
    print("This is the result of TABLE 1: 'An example for energy efficiency computation'")
    print("*"*50)
    if relay_num != 0:
        print("cc: [{}, {}]".format(min(min(computing_info[2].energy)), max(max(computing_info[2].energy))))
        print("df: [{}, {}]".format(min(min(computing_info[3].energy)),max(max(computing_info[3].energy))))
        print("cc power: [{}, {}]".format(min(min(computing_info[2].mobile_power)), max(max(computing_info[2].mobile_power))))
        print("df mobile power: [{}, {}]".format(min(min(computing_info[3].mobile_power)), max(max(computing_info[3].mobile_power))))
        print("df relay power: [{}, {}]".format(min(min(computing_info[3].relay_power)), max(max(computing_info[3].relay_power))))
    print("_"*50)

def motivation(mobile_num=4, relay_num=4):
    task = utils.Task(0.14, 0.14, 11.2, 11.2)
    bandwidth = 10 * utils.bandwidth_unit
    env = utils.Environment(bandwidth=bandwidth, pathloss=0.1, relay_awgn=-174, edge_awgn=-70)
    device = utils.Device()
    time_constraint = 0.1
    task.init_env(mobile_num, relay_num, time_constraint=time_constraint)
    task.max_workloadsize = 15
    task.max_datasize = 0.5
    local_computing = algo.get_local_computing_info(task, device)
    dec_computing = algo.get_direct_edge_computing_info(task, env, device)
    cc_computing = algo.get_cc_computing_info(task, env, device)
    df_computing = algo.get_df_computing_info(task, env, device)
    computing_info = [local_computing, dec_computing, cc_computing, df_computing]
    print_computing_info(computing_info, relay_num=1)


if __name__ == '__main__':
    # num = [0, 1, 2, 3, 4, 10]
    num = [4]
    for n in num:
        print("relay num:", n)
        motivation(relay_num=n)