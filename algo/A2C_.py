# coding: utf-8
# @Time    : 2021/1/26 9:36
# @Author  : myyao
# @FileName: A2C_.py
# @Software: PyCharm
# @e-mail  : myyaocn@outlook.com
# description :
import random
import time

import torch
import torch.nn as nn
import numpy as np
import utils
import torch.nn.functional as F
import algo


class Critic_loss(nn.Module):
    def __init__(self):
        super(Critic_loss, self).__init__()

    def forward(self, q_eval, reward):
        q_eval = q_eval.reshape(-1)
        t = torch.pow(q_eval, 2)
        reward = reward.reshape(-1)
        k = torch.multiply(reward, q_eval)
        action_loss = torch.mean(t - 2 * k)
        return torch.abs(action_loss)

class Critic(nn.Module):
    def __init__(self, mobile_num, relay_num, num_input, hidden_size=256):
        super(Critic, self).__init__()

        self.mobile_num = mobile_num
        self.relay_num = relay_num
        self.linear_1 = nn.Linear(num_input, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)

        self.critic = nn.Sequential(
            nn.Linear(num_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        q_value = self.critic(x)
        return q_value


class Policy_loss(nn.Module):
    def __init__(self):
        super(Policy_loss, self).__init__()

    def forward(self, q_eval, reward, action, beta=0.1):
        reward = reward.reshape(-1)
        a_n = torch.zeros([action.shape[0], 1])
        mobile_num = int(action.shape[1]/4)
        for i in range(a_n.shape[0]):
            for j in range(mobile_num):
                a_n[i] += torch.max(action[i, 4*j:4*j+4])
            a_n[i] /= mobile_num
        log = torch.multiply(torch.log(a_n), a_n)
        a_n_cse = torch.sum(log, dim=1)
        A = torch.multiply(a_n_cse, reward)
        B = log * beta
        policy_loss = torch.abs(torch.mean(torch.add(A, B)))
        return policy_loss


class Policy(nn.Module):
    def __init__(self, mobile_num, relay_num, num_input, hidden_size=256):
        super(Policy, self).__init__()
        self.mobile_num = mobile_num
        self.relay_num = relay_num
        self.actor_policy = nn.Sequential(
            nn.Linear(num_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.mobile_num * 4)
        )

    def forward(self, x):
        policy = self.actor_policy(x)
        policy = policy.reshape(-1, 4)
        policy = torch.softmax(policy, dim=1)
        policy = policy.reshape(-1)
        return policy


class Match_loss(nn.Module):
    def __init__(self):
        super(Match_loss, self).__init__()

    def forward(self, q_eval, reward, action, beta=0.1):
        reward = reward.reshape(-1)
        # print(torch.sum(torch.mul(torch.log(action), action), dim=1).shape)
        # print(reward.shape)
        A = torch.multiply(torch.sum(torch.mul(torch.log(action), action), dim=1).reshape(1, -1), action.reshape(-1, 1)) * reward
        B = torch.multiply(torch.log(action), action) * beta
        # print(A.shape, B.shape)
        policy_loss = torch.mean(torch.matmul(A, B))
        return policy_loss


class Match(nn.Module):
    def __init__(self, mobile_num, relay_num, num_input, hidden_size=256):
        super(Match, self).__init__()
        self.mobile_num = mobile_num
        self.relay_num = relay_num
        self.actor_match = nn.Sequential(
            nn.Linear(num_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.mobile_num * self.relay_num)
        )

    def forward(self, x):
        match = self.actor_match(x)
        match = match.reshape(self.mobile_num, self.relay_num)
        match = torch.softmax(match, dim=1)
        match = match.reshape(-1)
        return match


class DQN_A2C():
    def __init__(self, mobile_num, relay_num, episilo=0.9, gamma=0.90, lr=0.001, batch_size=128, memory_capacity=2000,
                 network_iteration=100):
        super(DQN_A2C, self).__init__()
        self.mobile_num = mobile_num
        self.relay_num = relay_num
        self.episilo = episilo
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.network_iteration = network_iteration
        self.num_input = self.mobile_num*7 + self.relay_num*2 + 2
        self.q_net = Critic(self.mobile_num, self.relay_num, self.num_input)
        self.policy_net = Policy(self.mobile_num, self.relay_num, self.num_input)
        self.match_net = Match(self.mobile_num, self.relay_num, self.num_input)
        self.q_net_target = Critic(self.mobile_num, self.relay_num, self.num_input)
        self.policy_net_target = Policy(self.mobile_num, self.relay_num, self.num_input)
        self.match_net_target = Match(self.mobile_num, self.relay_num, self.num_input)
        self.beta = 0.01

        for (eval, target) in zip(self.q_net.parameters(), self.q_net_target.parameters()):
            target.data = eval.data
        for (eval, target) in zip(self.policy_net.parameters(), self.policy_net_target.parameters()):
            target.data = eval.data
        for (eval, target) in zip(self.match_net_target.parameters(), self.match_net_target.parameters()):
            target.data = eval.data
        self.action_space = self.mobile_num * 4 + self.mobile_num * self.relay_num

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, self.num_input * 2 + 1 + self.action_space))
        self.beta = 0.01

        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.match_optimizer = torch.optim.Adam(self.match_net.parameters(), lr=self.lr)

        self.q_loss = Critic_loss()
        self.action_loss = nn.MSELoss()
        self.match_loss = Match_loss()
        self.tau = 0.01  # network update rate

    @classmethod
    def get_state(self, task, env, fading):
        bandwidth = env.bandwidth
        data_size = task.data_size
        workload_size = task.workload_size
        time = task.time_constraint
        location = []
        state = []
        for t in task.mobile_topology:
            # 2 * mobile_num
            location.extend(t)
        for t in task.relay_topology:
            # 2 * relay_num
            location.extend(t)
        location.extend(task.edge_topology)  # 2
        for t in data_size:
            # mobile_num
            state.append(t / float(task.max_datasize))
        for t in workload_size:
            # mobile_num
            state.append(t / float(task.max_workloadsize))
        for _ in range(task.mobile_num):
            # mobile_num
            state.append(bandwidth / float(utils.bandwidth_unit * 100))
        for t in time:
            # mobile_num
            state.append(t)
        for t in location:
            state.append(t / 100)
        state.extend([fading] * task.mobile_num)  # mobile_num
        return np.array(state)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.rand() <= self.episilo:
            policy_value = self.policy_net.forward(state)
            match_value = self.match_net.forward(state)
            # print("model")
        else:
            policy_value = torch.rand(self.mobile_num * 4)
            match_value = torch.rand(self.mobile_num * self.relay_num)
            policy_value = F.softmax(policy_value, 0)
            match_value = F.softmax(match_value, 0)
        #     print("rabd")
        # print(policy_value)
        # print(match_value)
        action = torch.cat([policy_value, match_value], dim=0)
        return action

    def store_transition(self, state, action, reward, next_state):
        reward = [reward] if not isinstance(reward, list) else reward
        action = action.reshape(-1)
        transition = np.hstack([state, action.detach().numpy(), reward, next_state])
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, gamma=0.1, alpha=0.1, beta_policy=0.1, beta_match=0.1):
        # update the parameter of target net
        # equation: o^t = tau * o (1-tau)o^t
        if self.learn_step_counter % self.network_iteration == 0:
            with torch.no_grad():
                for (eval, target) in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                    target.data = self.tau * eval.data + (1 - self.tau) * target.data
                for (eval, target) in zip(self.policy_net.parameters(), self.policy_net_target.parameters()):
                    target.data = self.tau * eval.data + (1 - self.tau) * target.data
                for (eval, target) in zip(self.match_net_target.parameters(), self.match_net_target.parameters()):
                    target.data = self.tau * eval.data + (1 - self.tau) * target.data
        self.learn_step_counter += 1

        # sample batch from transition data
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_input])
        batch_action = torch.FloatTensor(batch_memory[:, self.num_input:self.num_input + self.action_space])
        batch_reward = torch.FloatTensor(
            batch_memory[:, self.num_input + self.action_space:self.num_input + self.action_space + 1])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_input:])

        # policy
        q_eval = self.q_net(batch_state)
        next_value = self.q_net_target(batch_next_state)
        y_t = torch.FloatTensor(batch_reward) + gamma * next_value
        batch_action = self.policy_net(batch_state)
        policy_loss = self.action_loss(q_eval, batch_reward) * beta_policy
        # policy_loss = self.action_loss(q_eval, batch_reward, batch_action.reshape(batch_reward.shape[0], -1)) * beta_policy
        # q_eval = self.q_net(batch_state)
        # next_value = self.q_net_target(batch_next_state)
        # y_t = torch.FloatTensor(batch_reward) + gamma * next_value
        # delta_t = self.q_loss(y_t, q_eval)
        # policy_loss = beta_policy * delta_t
        # policy_loss = beta_policy * delta_t
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # match
        # q_eval = self.q_net(batch_state)
        # next_value = self.q_net_target(batch_next_state)
        # y_t = torch.FloatTensor(batch_reward) + gamma * next_value
        # delta_t = self.match_loss(y_t, q_eval)
        # match_loss = beta_match * delta_t
        q_eval = self.q_net(batch_state)
        next_value = self.q_net_target(batch_next_state)
        y_t = torch.FloatTensor(batch_reward) + gamma * next_value
        delta_t = self.q_loss(y_t, q_eval)
        match_loss = beta_match * delta_t
        self.match_optimizer.zero_grad()
        match_loss.backward()
        self.match_optimizer.step()

        # q_eval
        q_eval = self.q_net(batch_state)
        next_value = self.q_net_target(batch_next_state)
        y_t = torch.FloatTensor(batch_reward) + gamma * next_value
        delta_t = self.q_loss(y_t, q_eval)
        critic_loss = alpha * delta_t
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        return critic_loss, policy_loss, match_loss



def env_observation(action, task, env, device, fading, transform=False):
    next_state = DQN_A2C.get_state(task, env, fading)
    if transform:
        policy, match = utils.process_action_value(action, task.mobile_num, task.relay_num)
    else:
        policy = action[:task.mobile_num]
        match = action[task.mobile_num:]
    for i in range(task.mobile_num):
        if policy[i] == 1:
            d, w = algo.local_compting_rest_bit(task.time_constraint[i], task.data_size[i], task.workload_size[i], device.mobile_min_f, device.mobile_max_f, device.mobile_k)
        elif policy[i] == 2:
            d, w = algo.dec_computing_rest_bit(i, task, env, device, fading)
        elif policy[i] == 3:
            j_index = match[i]
            d, w = algo.cc_computing_rest_bit(i, j_index, task, env, device, fading)
        elif policy[i] == 4:
            j_index = match[i]
            d, w = algo.df_computing_rest_bit(i, j_index, task, env, device, fading)
        else:
            d, w = 0, 0
        next_state[i] = d
        next_state[i+task.mobile_num] = w
    return next_state


def train(task, env, device, episodes=400, min_max=False):
    a2c = DQN_A2C(task.mobile_num, task.relay_num)
    print("collecting experience....")
    reward_list = []
    for i in range(episodes):
        start_time = time.time()
        task.set_time_constraint(random.uniform(0.1, 0.2))
        env.bandwidth = int(random.uniform(1, 5) * utils.bandwidth_unit)
        fading = random.uniform(0, 1)
        # comp = 0
        # dqn = 0
        # record = 1000
        task.init_env(mobile_num, relay_num, random.uniform(0.1, 0.5))
        # print("*"*100)
        mode, match, _ = algo.oecaa(task, env, device, fading) if min_max else algo.otca(task, env, device, fading)
        mode.extend(match)
        base = utils.get_reward(mode, task, env, device, fading, transform=False)
        base = base[1] if min_max else base[0]

        # get state
        state = a2c.get_state(task, env, fading)
        action = a2c.choose_action(state)

        # p, m = utils.process_action_value(action, task.mobile_num, task.relay_num)
        # print(action[:4 * task.mobile_num])
        # print(p)
        # print(action[4 * task.mobile_num:])
        # print(m)
        # input()

        # next_state = env_observation(action, task, env, device, transform=True)
        # state = next_state
        reward = utils.get_reward(action, task, env, device, fading, transform=True)
        reward = reward[1] if min_max else reward[0]
        a2c.store_transition(state, action, reward, state)
        if a2c.memory_counter >= a2c.memory_capacity:
            critic_loss, policy_loss, match_loss = a2c.learn(gamma=0.01, alpha=0.3, beta_policy=1, beta_match=0.5)
            if (i+1) % 100 == 0:
                print("epsilon: {}, loss: {:.5f}, {:.5f}, {:.5f}".format(i + 1, critic_loss, policy_loss, match_loss))
                print("energy: base:{:.5f}, dqn:{:.5f}".format(1000-1000*base, 1000-1000*reward))
    if min_max:
        # torch.save(a2c.q_net.state_dict(), 'q_net_minmax.pth')
        # torch.save(a2c.policy_net.state_dict(), 'policy_net_minmax.pth')
        # torch.save(a2c.match_net.state_dict(), 'match_net_minmax.pth')
        torch.save({
            'q_net': a2c.q_net.state_dict(),
            'policy_net': a2c.policy_net.state_dict(),
            'match_net': a2c.match_net.state_dict()
        }, 'model_min_max.pth')
        print("model has been saved in model_min_max.pth")
    else:
        # torch.save(a2c.q_net.state_dict(), 'q_net.pth')
        # torch.save(a2c.policy_net.state_dict(), 'policy_net.pth')
        # torch.save(a2c.match_net.state_dict(), 'match_net.pth')
        torch.save({
            'q_net': a2c.q_net.state_dict(),
            'policy_net': a2c.policy_net.state_dict(),
            'match_net': a2c.match_net.state_dict()
        }, 'model.pth')
        print("model has been saved in model.pth")
    # how to load?
    # q_eval = Critic(task.mobile_num, task.relay_num, self.num_input)
    # checkpoint = torch.load(PATH)
    # q_eval.load_state_dict(checkpoint['q_net'])
    for _ in range(10):
        # new env
        task.set_time_constraint(random.uniform(0.1, 0.5))
        env.bandwidth = int(random.uniform(10, 50) * utils.bandwidth_unit)
        task.gen_new_env(random.uniform(0.1, 0.5))
        fading = random.uniform(0, 1)
        # base
        mode, match, _ = algo.oecaa(task, env, device, fading) if min_max else algo.otca(task, env, device, fading)
        mode.extend(match)
        base = utils.get_reward(mode, task, env, device, fading, transform=False)
        base = base[1] if min_max else base[0]
        # dqn
        state = a2c.get_state(task, env, fading)
        action = a2c.choose_action(state)
        reward = utils.get_reward(action, task, env, device, fading, transform=True)
        reward = reward[1] if min_max else reward[0]
        print("base: {:.5f}, dqn: {:.5f}".format(base, reward))


if __name__ == '__main__':
    mobile_num = 1
    relay_num = 20
    task = utils.Task(0.3, 0.5, 15, 20)
    bandwidth = 5 * utils.bandwidth_unit
    env = utils.Environment(bandwidth=bandwidth)
    device = utils.Device()
    task.init_env(mobile_num, relay_num, 0.1)

    train(task, env, device, episodes=5000, min_max=True)