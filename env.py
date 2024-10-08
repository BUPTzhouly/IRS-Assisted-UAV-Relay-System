import numpy as np
import math
import tkinter as tk
import time

#  用户功率，时隙、频带均匀分配

K_L = 2   # 每个RIS左边用户个数2
K_R = 2   # 每个RIS右边用户个数2
region_l = 800      # 飞行区域长度800
region_w = 800      # 飞行区域宽度800
region_user = 400   # 用户区域长度（宽度）400
v_max = 10          # UAV最大飞行速率10
H_UAV = 200         # UAV飞行高度200
UAV_ini = [0, 0, H_UAV]     # UAV起始位置
H_RIS = 1   # RIS高度
position_RIS = [[400, 600, H_RIS], [200, 300, H_RIS], [600, 200, H_RIS]]
M = 40      # RIS单元个数   40
carry_frequency = 3.6*10**9         # 载频频率
lamda = 3*10**8/carry_frequency     # 载波波长
d_RIS =lamda/2      # RIS单元间距
tai = 1             # 时隙长度s
Noise_power = 10**-9        # 噪声功率-60dB   -9
BW = 3000000/3  # 带宽
Rician_factor = 10          # 瑞利因子10dB

path_loss_exponents = 1.2           # uav-ris,ris-user路损指数
path_loss_exponents_direct = 3.4    # uav-user路损指数
kesi_0 = 10**-4.5   # 参考距离为1m时的信道功率-45dB
P_max = 1   # 用户发射功率
np.random.seed(2)   # 设置了随机数生成器的种子为2

class Maze(object):

    def __init__(self):
         # super(Maze, self).__init__()        ##########
         # self.title('maze')                  ##########
         # self.geometry('{0}x{1}'.format(300, 300))       ##########
         self.user_num = (K_L+K_R)*3            # 用户数目
         self.s_dim = 2                         # s的维度 存储：状态
         self.a_dim = 1+1+M*6+3+(K_L+K_R)*3     # a的维度 存储：速率、角度、STAR-RIS相移*2*3、时隙分配、用户发射功率(KL+KR)*3
         self.a_bound = np.zeros(self.a_dim)    # 动作的上界
         self.UE_L = np.zeros((3, K_L, 2))      # 一个3*K_L*2的数组  ？3，用户数K_L，xOy坐标2
         self.UE_R = np.zeros((3, K_R, 2))      # 一个3*K_R*2的数组  ？3，用户数K_R，xOy坐标2
         self.UAV_new = UAV_ini                 # 新旧坐标都初始化为UAV_ini
         self.UAV_old = UAV_ini

         #  STAR_RIS的相移
         self.theta_ref = np.zeros((3, M, M))   # ref是反射    一个3*M*M的数组  RIS的位置有3个
         self.theta_tra = np.zeros((3, M, M))   # tra是透射
         self.theta_ref[0] = np.eye(M)          # np.eye()生成单位矩阵
         self.theta_ref[1] = np.eye(M)
         self.theta_ref[2] = np.eye(M)
         self.theta_tra[0] = np.eye(M)
         self.theta_tra[1] = np.eye(M)
         self.theta_tra[2] = np.eye(M)
         self.a_bound[0] = v_max    # 速率上界v_max
         self.a_bound[1] = np.pi    # 速度方向上界2pi
         for i in range(2, 1+1+M*2*3):      # RIS相移上界2pi
            self.a_bound[i] = np.pi
         for i in range(1+1+M*2*3, 1+1+M*2*3+3):   # 时隙分配上界限0.5s
            self.a_bound[i] = tai/2
         for i in range(1+1+M*2*3+3, self.a_dim):  # 用户发射功率上界P_max
            self.a_bound[i] = P_max/2

    # 随机产生用户位置
    def position_user(self):
        # 左边区域随机撒点
        for j in range(3):
            for i in range(K_L):
                x_user = np.random.uniform(position_RIS[j][0] - region_user / 2, position_RIS[j][0])
                # numpy库中的一个函数，用于生成一个在闭区间[low, high]内均匀分布的随机浮点数
                y_user = np.random.uniform(position_RIS[j][1] - region_user / 2, position_RIS[j][1] + region_user / 2)
                self.UE_L[j][i][0] = x_user         # 用户的x坐标
                self.UE_L[j][i][1] = y_user         # 用户的y坐标
            for i in range(K_R):
                x_user = np.random.uniform(position_RIS[j][0], position_RIS[j][0] + region_user / 2)
                y_user = np.random.uniform(position_RIS[j][1] - region_user / 2, position_RIS[j][1] + region_user / 2)
                self.UE_R[j][i][0] = x_user         # 用户的x坐标
                self.UE_R[j][i][1] = y_user         # 用户的y坐标

    # 计算两点距离
    def distance(self, location1_x, location1_y, location1_z, location2_x, location2_y, location2_z):
        return math.sqrt(
            (location1_x - location2_x) ** 2 + (location1_y - location2_y) ** 2 + (location1_z - location2_z) ** 2)

    # 计算UE到RIS的信道增益
    def UE_RIS(self, distance, UE_x, RIS_number):
        kesi_UE_RIS = kesi_0 / (distance ** path_loss_exponents)    # kesi_0：参考距离1米处的信道功率
        h_LoS = np.zeros(M, dtype=complex)
        h_NLoS = np.zeros(M, dtype=complex)
        cos_AOA = (position_RIS[RIS_number][0] - UE_x) / distance
        for i in range(M):
            h_LoS[i] = np.e ** (complex(0, -(2 * np.pi * distance / lamda))) * np.e ** (
                complex(0, -(2 * np.pi * i * distance * cos_AOA / lamda)))
            h_NLoS[i] = complex(np.random.normal(), np.random.normal()) / math.sqrt(2)  # 均值为0，方差为1的复高斯随机变量
        h = math.sqrt(kesi_UE_RIS) * ((math.sqrt(Rician_factor / (Rician_factor + 1)) * h_LoS + math.sqrt(1 / (Rician_factor + 1)) * h_NLoS))
        return h

    # 计算RIS到UAV的信道增益
    def RIS_UAV(self, distance, RIS_number):
        kesi_RIS_UAV = kesi_0 / (distance ** path_loss_exponents)
        h_LoS = np.zeros(M, dtype=complex)
        h_NLoS = np.zeros(M, dtype=complex)
        cos_AOD = (self.UAV_new[0] - position_RIS[RIS_number][0]) / distance
        for i in range(M):
            h_LoS[i] = np.e ** (complex(0, - (2 * np.pi * distance / lamda))) * np.e ** (complex(0,- (2 * np.pi * i * distance * cos_AOD / lamda)))
            h_NLoS[i] = complex(np.random.normal(), np.random.normal()) / math.sqrt(2)  # 均值为0，方差为1的复高斯随机变量
        h = math.sqrt(kesi_RIS_UAV) * ((math.sqrt(Rician_factor / (Rician_factor + 1))* h_LoS+math.sqrt(1 / (Rician_factor+1))*h_NLoS))
        return h

    # 计算用户直接到无人机的信道增益。
    def UE_UAV(self, distance):
        kesi_UE_UAV = kesi_0 / (distance ** path_loss_exponents_direct)
        h_LoS = np.e**(complex(0, - (2 * np.pi * distance / lamda)))
        h_NLoS = complex(np.random.normal(), np.random.normal()) / math.sqrt(2)
        h = math.sqrt(kesi_UE_UAV) * ((math.sqrt(Rician_factor / (Rician_factor + 1))*h_LoS+math.sqrt(1 / (Rician_factor+1))*h_NLoS))
        return h

    # 计算整体的路径损失
    def pathloss(self):
        h_kl = np.zeros((3, K_L), dtype=complex)    # 二维数组
        h_kr = np.zeros((3, K_R), dtype=complex)
        h_UE_kl_RIS = np.zeros((3, K_L, M), dtype=complex)      # 三维数组
        h_UE_kl_UAV = np.zeros((3, K_L), dtype=complex)
        h_UE_kr_RIS = np.zeros((3, K_R, M), dtype=complex)
        h_UE_kr_UAV = np.zeros((3, K_R), dtype=complex)
        h_RIS_UAV = np.zeros((3, M), dtype=complex)
        for j in range(3):
            # 判断是反射用户还是透射用户
            if self.UAV_new[0] <= position_RIS[j][0]:
                r_t_falg_L = 1
                r_t_falg_R = 0
            else:
                r_t_falg_L = 0
                r_t_falg_R = 1
            distance_RIS_UAV = self.distance(position_RIS[j][0], position_RIS[j][1], position_RIS[j][2], self.UAV_new[0],
                                             self.UAV_new[1], self.UAV_new[2])
            h_RIS_UAV[j] = self.RIS_UAV(distance_RIS_UAV, j)
            for i in range(K_L):
                distance_UE_RIS = self.distance(self.UE_L[j][i][0], self.UE_L[j][i][1], 0, position_RIS[j][0],
                                                position_RIS[j][1], position_RIS[j][2])
                distance_UE_UAV = self.distance(self.UE_L[j][i][0], self.UE_L[j][i][1], 0, self.UAV_new[0],
                                                self.UAV_new[1], self.UAV_new[2])
                h_UE_kl_RIS[j][i] = self.UE_RIS(distance_UE_RIS, self.UE_L[j][i][0], j)
                h_UE_kl_UAV[j][i] = self.UE_UAV(distance_UE_UAV)
                h_kl[j][i] = np.dot(np.dot(np.array(h_RIS_UAV[j]).conjugate(),
                                        (r_t_falg_L * self.theta_ref[j] + r_t_falg_R * self.theta_tra[j])), h_UE_kl_RIS[j][i]) + \
                          h_UE_kl_UAV[j][i]
            for i in range(K_R):
                distance_UE_RIS = self.distance(self.UE_R[j][i][0], self.UE_R[j][i][1], 0, position_RIS[j][0],
                                                position_RIS[j][1], position_RIS[j][2])
                distance_UE_UAV = self.distance(self.UE_R[j][i][0], self.UE_R[j][i][1], 0, self.UAV_new[0],
                                                self.UAV_new[1], self.UAV_new[2])
                h_UE_kr_RIS[j][i] = self.UE_RIS(distance_UE_RIS, self.UE_R[j][i][0], j)
                h_UE_kr_UAV[j][i] = self.UE_UAV(distance_UE_UAV)
                h_kr[j][i] = np.dot(np.dot(np.array(h_RIS_UAV[j]).conjugate(),
                                        (r_t_falg_R * self.theta_ref[j] + r_t_falg_L * self.theta_tra[j])), h_UE_kr_RIS[j][i]) + \
                          h_UE_kr_UAV[j][i]
        return h_kl, h_kr

    # 计算信噪比和速率
    def rate(self):
        h_kl, h_kr = self.pathloss()
        #   根据增益大小排序
        #   计算干扰
        self.SINR_kl = np.zeros((3, K_L))
        self.rate_kl = np.zeros((3, K_L))
        self.energy_kl = np.zeros((3, K_L))
        self.SINR_kr = np.zeros((3, K_R))
        self.rate_kr = np.zeros((3, K_R))
        self.energy_kr = np.zeros((3, K_R))
        interfrence_kl = np.zeros((3, K_L))
        interfrence_kr = np.zeros((3, K_R))
        for o in range(3):
            for i in range(K_L):
                for j in range(K_L):
                    if abs(h_kl[o][i]) > abs(h_kl[o][j]):
                        interfrence_kl[o][i] += (self.action[1+1+M*6+3+o*(K_L+K_R)+j]+P_max/2)*abs(h_kl[o][j]) ** 2
            for i in range(K_R):
                for j in range(K_R):
                    if abs(h_kr[o][i]) > abs(h_kr[o][j]):
                        interfrence_kr[o][i] += (self.action[1+1+M*6+3+o*(K_L+K_R)+K_L+j]+P_max/2)*abs(h_kr[o][j]) ** 2
            for i in range(K_L):
                self.SINR_kl[o][i] = 1 + (self.action[1+1+M*6+3+o*(K_L+K_R)+i]+P_max/2)*abs(h_kl[o][i]) ** 2 / (interfrence_kl[o][i] + Noise_power ** 2)
                self.rate_kl[o][i] = BW * (self.action[1+1+M*6+o]+tai/2) * np.log2(self.SINR_kl[o][i])
                self.energy_kl[o][i] = (self.action[1+1+M*6+3+o*(K_L+K_R)+i]+P_max/2)*(self.action[1+1+M*6+o]+tai/2)
            for i in range(K_R):
                self.SINR_kr[o][i] = 1 + (self.action[1+1+M*6+3+o*(K_L+K_R)+K_L+i]+P_max/2)*abs(h_kr[o][i]) ** 2 / (interfrence_kr[o][i] + Noise_power ** 2)
                self.rate_kr[o][i] = BW * (1-(self.action[1+1+M*6+o]+tai/2)) * np.log2(self.SINR_kr[o][i])
                self.energy_kr[o][i] = (self.action[1+1+M*6+3+o*(K_L+K_R)+K_L+i]+P_max/2)*(1-(self.action[1+1+M*6+o]+tai/2))

    #  状态转移
    def step(self, action):
        self.action = action
        self.UAV_old = self.UAV_new
        self.UAV_new = [self.UAV_old[0]+tai*action[0]*np.cos(action[1]),self.UAV_old[1]+tai*action[0]*np.sin(action[1]),
                        H_UAV]
        # self.UAV_new = np.clip(self.UAV_new, 0, region_l)              ###################
        # self.canvas.delete(self.uav)
        # self.uav = self.canvas.create_oval(self.UAV_new[0]/2- 4, self.UAV_new[1]/2- 4, self.UAV_new[0]/2 + 4,     ###########
        #                                           self.UAV_new[1]/2+ 4, fill='yellow')             ###########
        # self.render()             ###########
        s_ = np.array(self.UAV_new[0:2])
        for j in range(3):
            row_r, col_r = np.diag_indices_from(self.theta_ref[j])
            self.theta_ref[j][row_r, col_r] = self.action[2+j*2*M:2+j*2*M+M]
            row_t, col_t = np.diag_indices_from(self.theta_tra[j])
            self.theta_tra[j][row_t, col_t] = self.action[2+j*2*M+M:2+j*2*M+M+M]
        self.rate()
        r_k = np.zeros(self.user_num)   # 这一时隙内的真实速率
        energy = np.zeros(self.user_num)    # 这一时隙内的能耗
        reward = np.zeros(self.user_num)
        for o in range(3):
            for i in range(K_L):
               r_k[o*(K_L+K_R)+i] = self.rate_kl[o][i]
               energy[o*(K_L+K_R)+i] = self.energy_kl[o][i]
            for j in range(K_R):
               r_k[o * (K_L + K_R) +K_L+j] = self.rate_kr[o][j]
               energy[o * (K_L + K_R) +K_L+j] = self.energy_kr[o][j]
        for i in range(self.user_num):
            reward[i] = r_k[i]
        r = sum(reward)
        return s_, r, r_k, energy

    #  绘图
    def _build_maze(self):  # 1:1绘图比例
         self.canvas = tk.Canvas(self, bg='white', height=region_l/2+2, width=region_w/2+2)
         #  画用户位置
         ue_l_center = self.UE_L/2
         ue_r_center = self.UE_R/2

         for j in range(3):
             self.userplt_i = self.canvas.create_oval(position_RIS[j][0]/2 - 5, position_RIS[j][1]/2 - 5,
                                                      position_RIS[j][0]/2 + 5,
                                                      position_RIS[j][1]/2 + 5, fill='green')
             for i in range(K_L):
                 self.userplt_i = self.canvas.create_oval(ue_l_center[j][i][0] - 2, ue_l_center[j][i][1] - 2,
                                                          ue_l_center[j][i][0] + 2,
                                                          ue_l_center[j][i][1] + 2, fill='black')
             for i in range(K_R):
                 self.userplt_i = self.canvas.create_oval(ue_r_center[j][i][0] - 2, ue_r_center[j][i][1] - 2,
                                                      ue_r_center[j][i][0] + 2,
                                                      ue_r_center[j][i][1] + 2, fill='black')

         self.canvas.pack()
    # 重置UAV
    def reset_uav(self):
        # self.UAV_new = UAV_ini
        # self.uav = self.canvas.create_oval(UAV_ini[0]/2 - 4, UAV_ini[1]/2 - 4, UAV_ini[0]/2+ 4,         ########
        #                                     UAV_ini[1]/2+ 4, fill='yellow')                                ########
        # self.update()                                                                                   ########
        # time.sleep(0.1)                                                                                 ########
        return np.array(UAV_ini[0:2])

    # def render(self):                                                                                     ########
        # time.sleep(0.1)                                                                                 ########
        # self.update()                                                                                   ########
        # self.canvas.delete(self.uav)                                                                      ########


