import torch
import numpy as np

class ENV:
    def __init__(self, cfg):
        self.width = cfg.f_w   # the width of field
        self.height = cfg.f_h  # the height of field
        self.N = cfg.s_N       # num of sensors
        self.b_C = cfg.b_C     # battery capacity
        self.p_c = cfg.p_c     # charging power
        self.alpha = cfg.alpha # param for wireless charging
        self.beta = cfg.beta   # param for wireless charging
        self.m_p = cfg.m_p     # moving price
        self.state_dim = self.N + 2   # dimension of state
        self.action_dim = 2           # dimension of action
        self.d_th = (self.alpha * self.p_c / cfg.p_th)**0.5 - self.beta # effective charging range

        # 初始化并固定传感器坐标
        self.loc = np.random.rand(2*self.N)
        for i in range(self.N):
            self.loc[2*i] = self.loc[2*i] * self.width
            self.loc[2*i+1] = self.loc[2*i+1] * self.height

        # 初始化传感器充电需求
        self.dem = np.random.rand(self.N)
        
        # 初始化移动充电桩坐标
        self.posi = np.random.rand(2)

        # 初始化环境状态信息
        self.state = np.concatenate((self.dem, self.posi), axis = 0)


    def reset(self) -> np:
        # 重随传感器充电需求
        self.dem = np.random.rand(self.N) * self.b_C
        
        # 重随移动充电桩坐标
        self.posi = np.random.rand(2)

        # 构建重随后的环境状态
        self.state = np.concatenate((self.dem, self.posi), axis = 0)
        return self.state.copy()  # 防止后续状态的更新对已记录的状态造成修改


    def step(self, action):
        # 将actor网络输出的[0, 1]动作逆转换成环境中的动作，即确定移动充电桩的新驻留点坐标
        a = action.copy()  # 防止该动作在存入经验池时发生改变
        a[0] = a[0] * self.width
        a[1] = a[1] * self.height

        # 过滤新驻留点周围的有充电需求的传感器设备
        t_max = 0
        dem_sfy = 0
        for n in range(self.N):
            d = ((self.loc[2*n] - a[0])**2 + (self.loc[2*n + 1] - a[1])**2)**0.5
            if d <= self.d_th and self.state[n] > 0:
                dem_sfy += self.state[n]
                t = self.state[n] / (self.p_c * self.alpha / (self.beta + d)**2)
                t_max = t if t > t_max else t_max
                # 更新环境状态中关于充电需求的信息
                self.state[n] = 0
        
        # 计算充电成本
        charging_cost = self.p_c * t_max
        # 计算移动成本
        move_cost = self.m_p * ((a[0] - self.state[self.N] * self.width)**2 + \
            (a[1] - self.state[self.N + 1] * self.height)**2)**0.5
        # 计算单步奖赏
        # reward = 1000 * dem_sfy / (charging_cost + move_cost) if charging_cost + move_cost != 0 else -1
        reward = 10000 * dem_sfy - (charging_cost + move_cost)

        # 更新环境状态中关于移动充电桩位置的信息
        self.state[self.N] = a[0] / self.width
        self.state[self.N + 1] = a[1] / self.height

        # 判断是否所有传感器都充满电，即episode是否结束
        done = True
        for i in range(self.N):
            if self.state[i] > 0:
                done = False
                break
        
        # 返回下一状态、奖赏和episode终止标志
        return self.state.copy(), reward, done  # numpy向量的深拷贝、标量、bool






