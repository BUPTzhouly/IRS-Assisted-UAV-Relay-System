"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2     1.13.1
gym 0.9.2
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from env import Maze
import xlsxwriter
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# EP_MAX = 6000
EP_MAX = 1000    # zly+  最大训练回合数

EP_LEN = 200    # 每个回合最大步数
region_l = 800
GAMMA = 0.9     # 未来奖励的折现率
A_LR = 0.0001   # actor的学习率
C_LR = 0.0002   # critic的学习率
BATCH = 64      # 每BATCH步更新一次模型
A_UPDATE_STEPS = 10     # 每次更新时，对actor进行多少次优化
C_UPDATE_STEPS = 10     # 每次更新时，对critic进行多少次优化
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL散度惩罚
    dict(name='clip', epsilon=0.2),                 # clipping方法。Clipped surrogate objective, find this is better
][1]        # 选择优化方法，这里使用的是clipping方法


class PPO(object):

    def __init__(self):
        env = Maze()    # 创建环境实例
        self.sess = tf.Session()                # 这是什么函数
        self.tfs = tf.placeholder(tf.float32, [None, env.s_dim], 'state')       # 这是什么函数

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 300, tf.nn.relu)     # 构建了一个全连接层，用Relu激活函数
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))      # 计算损失
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)      # 选择损失最小的动作

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)      # 定义了当前策略pi和旧策略oldpi，用于计算策略梯度和实施PPO的核心策略更新。
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # 从策略分布中采样一个动作。
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]     # 策略更新

        self.tfa = tf.placeholder(tf.float32, [None, env.a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')     # 什么是优势
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):        # 这段什么意思
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)   # 计算新旧策略之间的比率，用于PPO的重要性采样。
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':      # 根据选择的优化方法（KL散度惩罚或裁剪），计算actor的损失函数
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):      # 在每个批次后更新策略。它首先更新旧策略参数，计算优势函数，然后根据选择的方法更新actor和critic。
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):     # 构建actor网络，生成动作分布
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 600, tf.nn.relu, trainable=trainable)
            mu = env.a_bound * tf.layers.dense(l1, env.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, env.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):     # 根据当前策略选择动作
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        for i in range(env.a_dim):
            a[i] = np.clip(a[i], -env.a_bound[i], env.a_bound[i])
        return a  # 限制动作区间

    def get_v(self, s):     # 计算状态的价值，辅助计算优势函数
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


def save(data, name):
    f = xlsxwriter.Workbook(filename=name)  # 创建工作簿
    sheet1 = f.add_worksheet()  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
           sheet1.write(i, j, data[i, j])
    f.close()


if __name__ == '__main__':
    env = Maze()
    env.position_user()  # 用户撒点!!!!!!!
    print(env.UE_L, env.UE_R)
    # env._build_maze()         # 为什么注释掉了
    ppo = PPO()
    EP_R = []
    user_rate = [[] for i in range(env.user_num)]
    user_energy = [[] for i in range(env.user_num)]
    uav_pos = [[]for i in range(EP_MAX)]
    buffer_s, buffer_a, buffer_r = [], [], []
    for ep in range(EP_MAX):
        s = env.reset_uav()
        s = (s / region_l)
        ep_r = 0
        ep_r_k = np.zeros(env.user_num)  # 1*18矩阵
        energy_k = np.zeros(env.user_num)  # 1*18矩阵
        for t in range(EP_LEN):  # in one episode
            a = ppo.choose_action(s)
            uav_pos[ep].append(s[0])
            uav_pos[ep].append(s[1])
            s_, r, r_k, energy = env.step(a)    #
            s_ /= region_l
            r = (r * 10 ** (-6)-17) / 200
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)  # normalize reward, find to be useful
            s = s_
            ep_r += r
            ep_r_k += r_k  # 矩阵
            energy_k += energy  # 矩阵
            # update ppo
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)

        for i in range(env.user_num):
            user_rate[i].append(ep_r_k[i])  # T时长的速率之和，还不是平均速率
            user_energy[i].append(energy_k[i])
        EP_R.append(ep_r)
        print('{0:.1f}%'.format(ep/EP_MAX * 100), '|Ep_r: %.2f' % ep_r, )

    #  plot reward change and test
    plt.plot(np.arange(len(EP_R)), EP_R)

    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.ioff()  # zly+
    plt.show()
    #
    #plt.plot(np.arange(len(EP_R)), np.sum(user_rate, axis=0))
    #plt.xlabel('Episode')
    #plt.ylabel('Moving reward')
    #plt.ion()
    #plt.ioff()  # zly+
    #plt.show()
    #
    #
    #for i in range(env.user_num):
    #     plt.plot(np.arange(len(user_rate[i])), user_rate[i])
    #     plt.xlabel('Episode')
    #     plt.ylabel('Moving reward')
    #     plt.ion()
    #     plt.ioff()  # zly+
    #     plt.show()
    #
    #     plt.plot(np.arange(len(user_energy[i])), user_energy[i])
    #     plt.xlabel('Episode')
    #     plt.ylabel('Moving reward')
    #     plt.ion()
    #     plt.show()

    result1 = pd.DataFrame({'reward': EP_R, 'sum_rate': np.sum(user_rate, axis=0)})
    result1.to_csv("result2" ".csv")
    save(np.array(user_rate), 'userate.xlsx')
    save(np.array(user_energy), 'user_energy.xlsx')
    save(np.array(uav_pos), 'uavposi.xlsx')
