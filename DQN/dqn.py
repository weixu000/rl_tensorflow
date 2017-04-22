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

    def __init__(self, observation_shape, obervation_range, observations_in_state, action_n, log_dir,
                 features: FeaturesNet, Q_layers: QLayerNet, memory: Memory, target: Target,
                 LEARNING_RATE=2E-3, EPS_INITIAL=0.5, EPS_DECAY_RATE=1 - 1E5):
        """
        :param observation_shape: observation数组尺寸
        :param obervation_range: obeservation数组范围
        :param observations_in_state: 多少个observation组成一个state
        :param action_n: 动作个数
        :param log_dir: log文件路径
        :param features: 网络feature部分
        :param Q_layers: 网络Q值部分
        :param memory: 记忆
        :param target: 目标Q值计算
        :param LEARNING_RATE: 学习速率
        :param EPS_INITIAL: 初始epsilon
        :param EPS_DECAY_RATE: epsilon衰减率
        """
        self.LEARNING_RATE = LEARNING_RATE
        self.EPS_INITIAL = EPS_INITIAL
        self.EPS_DECAY_RATE = EPS_DECAY_RATE

        self.observation_shape = list(observation_shape)
        self.observation_range = obervation_range
        self.observations_in_state = observations_in_state
        self.__observation_buff = []  # 缓存一连串observation用以构成state
        if self.observations_in_state == 1:
            self.state_shape = self.observation_shape  # observation视作state
        else:
            if len(self.observation_shape) == 1:
                self.state_shape = [self.observation_shape[0] * self.observations_in_state]  # 向量observation加长视作state
            else:
                self.state_shape = self.observation_shape + [self.observations_in_state]  # 矩阵observation加一维视作state
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
        self.__layers_n = [self.state_shape]  # 输入层为state
        with self.__graph.as_default():
            self.__layers = self.__features.create_features(self.__layers_n)  # 初始化网络feature部分
            self.__Q_layers.create_Q_layers(self.action_n, self.__layers_n, self.__layers)  # 初始化网络Q值部分
            self.__memory.create_loss(self.__layers[0], self.__layers[-1], self.LEARNING_RATE)  # 初始化误差
            self.init = tf.global_variables_initializer()
            self.__target.create_target(self.init, self.__layers[0], self.__layers[-1])  # 初始化目标Q值计算

    def normalize_observation(self, observation):
        """
        正规化状态
        """
        if self.observation_range:
            return (observation - self.observation_range[0]) / (
                self.observation_range[1] - self.observation_range[0])
        else:
            return observation

    def __getitem__(self, state):
        """
        给定单个状态，计算所有动作的Q值
        :param state: 状态
        :return: 所有动作的Q值
        """
        return self.__target[np.array(state).reshape(self.state_shape)]

    def greedy_action(self, observation):
        """
        返回当前状态，当前Q值下认为最优的动作
        """
        tmp = self.__observation_buff + [self.normalize_observation(observation)]
        if len(tmp) >= self.observations_in_state:
            return np.argmax(self[tmp[-self.observations_in_state:]])
        else:
            return np.random.choice(self.action_n)  # 现有observation不够，随机选择动作

    def epsilon_greedy(self, observation):
        """
        epsilon可能选取任意动作
        """
        eps = self.EPS_INITIAL * self.EPS_DECAY_RATE ** self.n_timesteps
        return self.greedy_action(observation) if np.random.rand() > eps else np.random.choice(self.action_n)

    def step(self, observation, action, reward, done):
        """
        接受观察，生成状态，记忆经验
        :param observation: 观察
        :param action: 动作onehot表示
        :param reward: 
        :param done: 是否终态
        """
        # 将动作单个数值转化成onehot向量：2变成[0,0,1,0,0]
        onehot = np.zeros((self.__layers_n[-1]))
        onehot[action] = 1
        # 正规化观察
        observation = self.normalize_observation(observation)

        self.__observation_buff.append(observation)
        if len(self.__observation_buff) > self.observations_in_state:
            state = np.array(self.__observation_buff[:self.observations_in_state]).reshape(self.state_shape)
            nxt_state = np.array(self.__observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
            self.__memory.perceive(state, self.__prev_action, reward, nxt_state, done)  # 上一次动作作为记忆
            self.__memory.replay(self.__target.compute_y)  # target产生y训练网络
            del self.__observation_buff[0]

        self.__prev_action = onehot  # 缓存一次动作
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
