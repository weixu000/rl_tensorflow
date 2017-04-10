import numpy as np
import tensorflow as tf
import gym
from agent import QNetwork


class FCQN(QNetwork):
    def __init__(self, env, env_name):
        self.HIDDEN_LAYERS = [30, 20]
        assert type(env.action_space) == gym.spaces.discrete.Discrete
        super().__init__(env, env_name)

    def create_feature(self):
        self.layers_n += [self.env.observation_space.shape[0]] + self.HIDDEN_LAYERS  # 全连接特征层中神经元个数
        with tf.name_scope('input_layer'):
            self.layers = [tf.placeholder(tf.float32, [None, self.layers_n[0]])]  # 输入层
        self.layers += self.create_FC_stream(self.layers[-1], self.layers_n, 'feature_layer')


class ConvQN(QNetwork):
    def __init__(self, env, env_name):
        self.CONVOLUTION = [
            ('conv', {'weight': [5, 5], 'bias': 32, 'strides': [1, 1, 1, 1]}),
            ('pooling', {'ksize': [1, 2, 2, 1], 'strides': [1, 2, 2, 1]})]
        super().__init__(env, env_name)

    def conv2d(self, layer):
        W = tf.Variable(tf.truncated_normal(layer['weights'], stddev=0.1), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=layer['bias']), name='bias')
        self.layers.append(tf.nn.relu(tf.nn.conv2d(self.layers[-1], W, strides=layer['strides'], padding='SAME') + b))

    def max_pool(self, layer):
        return tf.nn.max_pool(self.layers[-1], ksize=layer['ksize'],
                              strides=layer['strides'], padding='SAME')

    def create_feature(self):
        self.layers_n += [self.env.observation_space.shape]
        with tf.name_scope('input_layer'):
            self.layers = [tf.placeholder(tf.float32, [None] + self.layers_n[0])]  # 输入层
        for i, (layer_t, layer) in enumerate(self.CONVOLUTION):
            if layer_t == 'conv':
                with tf.name_scope('conv_layer{}'.format(i)):
                    self.conv2d(layer)
            elif layer_t == 'conv':
                with tf.name_scope('pooling_layer{}'.format(i)):
                    self.max_pool(layer)


class OriginalQLayer(FCQN):
    """
    DQN原来的网络结构
    """

    def __init__(self, env, env_name):
        self.Q_HIDDEN_LAYERS = [20, 10]
        super().__init__(env, env_name)

    def create_Q_layers(self):
        self.layers += self.create_FC_stream(self.layers[-1],
                                             [self.layers_n[-1]] + self.Q_HIDDEN_LAYERS,
                                             'Q_hidden_layer')
        self.layers_n += self.Q_HIDDEN_LAYERS + [self.env.action_space.n]
        with tf.name_scope('Q_layer'):
            self.layers.append(
                self.create_z(self.layers[-1], self.layers_n[-2], self.layers_n[-1]))  # 输出层，没有激活函数


class DuelingDQN(FCQN):
    """
    Dueling网络
    """

    def __init__(self, env, env_name):
        self.STATE_HIDDEN_LAYERS = [10, 5]
        self.ADVANTAGE_HIDDEN_LAYERS = [10, 5]
        super().__init__(env, env_name)

    def create_Q_layers(self):
        """
        建立网络
        """
        # 每层网络中神经元个数
        self.layers_n += [[self.STATE_HIDDEN_LAYERS + [1], self.ADVANTAGE_HIDDEN_LAYERS + [self.env.action_space.n]]] \
                         + [self.env.action_space.n]
        # 状态价值隐藏层
        state_stream = self.create_FC_stream(self.layers[-1], [self.HIDDEN_LAYERS[-1]] + self.STATE_HIDDEN_LAYERS,
                                             'state_hidden_layer')
        # 状态价值，不用RELU
        with tf.name_scope('state_value'):
            state_stream.append(self.create_z(state_stream[-1], self.STATE_HIDDEN_LAYERS[-1], 1))

        # 动作优势隐藏层
        advantage = self.create_FC_stream(self.layers[-1], [self.HIDDEN_LAYERS[-1]] + self.ADVANTAGE_HIDDEN_LAYERS,
                                          'advantage_hidden_layer')
        # 动作优势，不用RELU
        with tf.name_scope('action_advantage'):
            advantage.append(self.create_z(advantage[-1], self.ADVANTAGE_HIDDEN_LAYERS[-1], self.env.action_space.n))

        self.layers.append([state_stream, advantage])  # 并行加入网络中
        # 输出Q层
        with tf.name_scope('Q_layer'):
            self.layers.append(
                state_stream[-1] + advantage[-1] - tf.reduce_mean(advantage[-1], axis=1, keep_dims=True))
