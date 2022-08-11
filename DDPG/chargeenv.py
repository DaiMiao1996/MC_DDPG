__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np
import random
import math
import gym
import itertools

from gym import spaces
from gym.error import DependencyNotInstalled


class ChargeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.side_length = 1000
        self.N = 80  # 传感器的数量
        self.alpha = 36
        self.beta = 30
        self.x = random.uniform(0, 1)
        self.y = random.uniform(0, 1)
        self.pc = 0.9
        self.pr_th = 0.001
        self.eu = 5.6
        self.reset_flag = 1
        self.sensor_pos = None
        self.out_penalty = 80  # 动作出界惩罚
        self.capacity = np.random.random(self.N) * 2
        # 动作空间
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1, 2), dtype=np.float32
        )
        # 状态空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, self.N + 2), dtype=np.float32)
        self.state = 0
        # self.observation_space = np.empty(shape=(1,self.N+2))

        # 创建传感器位置

        self.sensor_pos = np.random.random([self.N, 2])
        np.save('sensor_pos',self.sensor_pos)

    def step(self, action):
        # action的范围是-1~+1（加噪声），在这个地方逆归一化
        state = self.state
        # print('aaaaaaaaaaaa',[state[80],state[81]])
        next_state = state.copy()
        next_state[self.N], next_state[self.N + 1] = action[0][0], action[0][1]

        # 出界判断
        # out_flag = 0  # 出界标志，出界则引入惩罚
        #
        # for i in range(2):
        #     if next_state[self.N + i] > self.side_length or next_state[self.N + i] < 0:
        #         next_state[self.N + i] = next_state[self.N + i] - action[0][i]
        #         out_flag = 1

        # for i in range(2):
        #     if next_state[self.N+i] > self.side_length:
        #         next_state[self.N+i] = next_state[self.N+i] - 1000 # 淼哥的主意，想让这个方位尽量动起来
        #         out_flag = 1
        #     if next_state[self.N+i] < 0:
        #         next_state[self.N+i] = next_state[self.N+i] + 1000
        #         out_flag = 1
        d_move = math.sqrt(
            math.pow(next_state[self.N] - state[self.N], 2) + math.pow(next_state[self.N + 1] - state[self.N + 1],
                                                                       2)) * 1000
        cost_m = d_move * self.eu

        # 计算充电状态损耗
        pos_x = next_state[self.N]
        pos_y = next_state[self.N + 1]
        done = True  # False:未完成充电
        T_max = 0
        s_pos = self.sensor_pos
        d_th = math.sqrt(self.alpha * self.pc / self.pr_th) - self.beta
        c_total = 0  # 当前状态充电总量
        for i in range(self.N):  # 把范围内的传感器充电置零，并获得最长充电时间
            d = math.sqrt(math.pow(s_pos[i][0] - pos_x, 2) + math.pow(s_pos[i][1] - pos_y, 2)) * 1000
            if (d < d_th and state[i] > 0):
                pr = self.alpha * self.pc / math.pow(d + self.beta, 2)
                t = state[i] / pr
                c_total = c_total + state[i]
                # print(c_total)
                if (t > T_max):
                    T_max = t
                next_state[i] = 0
        cost_p = self.pc * T_max
        reward = c_total * 10000 - (cost_p + cost_m)
        n = 0
        for i in range(self.N):
            if (state[i] != 0):
                n += 1
        if n != 0:
            done = False
        # print(n)
        self.state = next_state.copy()
        return next_state, reward, done

    # 问题先记录一下，关于python形参的问题
    def reset(self):
        # 创建传感器电量
        self.x = random.uniform(0, 1)
        self.y = random.uniform(0, 1)
        state_t = self.capacity
        state_t = np.append(state_t, self.x)
        state_t = np.append(state_t, self.y)
        self.state = state_t

        return self.state

    def render(self, mode="human"):
        return None

    def close(self):
        return None
