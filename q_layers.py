import tensorflow as tf
from features import FCQN


class OriginalDQN(FCQN):
    def __init__(self, env, env_name):
        super().__init__(env, env_name)

    def create_Q_layers(self):
        self.layers_n.append(self.env.action_space.n)  # 每层网络中神经元个数
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
