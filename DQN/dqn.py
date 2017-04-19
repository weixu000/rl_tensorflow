import numpy as np
import tensorflow as tf
from DQN.network import FeaturesNet, QLayerNet
from DQN.memory import Memory
from DQN.target import Target
import json


class DQN:
    """
    DQN总的类
    """

    def __init__(self, observation_shape, action_n, log_dir,
                 features: FeaturesNet, Q_layers: QLayerNet, memory: Memory, target: Target,
                 LEARNING_RATE=2E-3, INITIAL_EPS=0.5, FINAL_EPS=0, EPS_DECAY_RATE=0.95, EPS_DECAY_STEP=2000):
        """
        :param observation_shape: observation数组尺寸
        :param observation_shape: 动作个数
        :param log_dir: log文件路径
        :param features: 网络feature部分
        :param Q_layers: 网络Q值部分
        :param memory: 记忆
        :param target: 目标Q值计算
        :param LEARNING_RATE: 学习速率
        :param INITIAL_EPS: 初始epsilon
        :param FINAL_EPS: 最终epsilon
        :param EPS_DECAY_RATE: epsilon衰减率
        :param EPS_DECAY_STEP: epsilon何时衰减一个EPS_DECAY_RATE
        """
        self.LEARNING_RATE = LEARNING_RATE
        self.INITIAL_EPS = INITIAL_EPS
        self.FINAL_EPS = FINAL_EPS
        self.EPS_DECAY_RATE = EPS_DECAY_RATE
        self.EPS_DECAY_STEP = EPS_DECAY_STEP

        self.observation_shape = observation_shape
        self.action_n = action_n
        self.log_dir = log_dir
        self.n_episodes = 0
        self.n_timesteps = 0

        self.__features = features
        self.__Q_layers = Q_layers
        self.__memory = memory
        self.__target = target

        # 构建网络
        self.__graph = tf.Graph()
        self.__layers_n = [self.observation_shape]  # 输入层为observation
        with self.__graph.as_default():
            self.__layers = self.__features.create_features(self.__layers_n)  # 初始化网络feature部分
            self.__Q_layers.create_Q_layers(self.action_n, self.__layers_n, self.__layers)  # 初始化网络Q值部分
            self.__memory.create_loss(self.__layers[0], self.__layers[-1], self.LEARNING_RATE)  # 初始化误差
            self.init = tf.global_variables_initializer()
            self.__target.create_target(self.init, self.__layers[0], self.__layers[-1])  # 初始化目标Q值计算

    def normalize_state(self, state):
        """
        正规化状态
        这个环境不能用
        """
        return state
        # return (state - self.env.observation_space.low) / (
        #     self.env.observation_space.high - self.env.observation_space.low)

    def __getitem__(self, state):
        """
        给定单个状态，计算所有动作的Q值
        :param state: 状态
        :return: 所有动作的Q值
        """
        return self.__target[self.normalize_state(state)]

    def greedy_action(self, state):
        """
        返回当前状态，当前Q值下认为最优的动作
        """
        return np.argmax(self[state])

    def epsilon_greedy(self, state):
        """
        epsilon可能选取任意动作
        """
        eps = self.FINAL_EPS + (self.INITIAL_EPS - self.FINAL_EPS) * self.EPS_DECAY_RATE ** (
            self.n_timesteps / self.EPS_DECAY_STEP)
        return self.greedy_action(state) if np.random.rand() > eps else np.random.choice(self.action_n)

    def perceive(self, state, action, reward, nxt_state, done):
        """
        接受经验
        :param state: 状态
        :param action: 动作onehot表示
        :param reward: 
        :param nxt_state: 下一个状态
        :param done: 是否终态
        """
        # 将动作单个数值转化成onehot向量：2变成[0,0,1,0,0]
        onehot = np.zeros((self.__layers_n[-1]))
        onehot[action] = 1

        # 正规化状态
        state = self.normalize_state(state)
        nxt_state = self.normalize_state(nxt_state)

        self.__memory.perceive(state, onehot, reward, nxt_state, done)  # memory记忆经验
        self.__memory.replay(self.__target.compute_y)  # target产生y训练网络
        self.n_timesteps += 1
        if done: self.n_episodes += 1

    def save_hyperparameters(self):
        """
        保存参数到parameters.json
        """
        params = dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))
        for i, x in {'Feature': self.__features, 'Q_Layers': self.__Q_layers, 'Memory': self.__memory,
                     'Target': self.__target}.items():
            params[i] = x.save_hyperparameters()
        with open(self.log_dir + 'parameters.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)
