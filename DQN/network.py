import tensorflow as tf
import numpy as np


def create_z(prev, col):
    """
    建立z=x*W+b
    :param prev: 上一层
    :param row: W行数
    :param col: W列数
    :return: z
    """
    W = tf.Variable(tf.truncated_normal([prev.shape[1].value, col]), name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[col]), name='biases')  # 正偏置促进学习
    return tf.matmul(prev, W) + b


def create_FC_stream(prev, layers_n, name):
    """
    按leyers_n生成全链接网络流
    :param prev: 上一全链接层
    :param layers_n: 网络结构list
    :param name: name_scope
    :return: 每层tensor
    """
    layers = [prev]
    for i, cur in enumerate(layers_n):
        with tf.name_scope('{}{}'.format(name, i)):
            layers.append(tf.nn.relu(create_z(layers[-1], cur)))  # 隐藏层用RELU
    return layers[1:]


class FeaturesNet:
    """
    网络feature部分基类
    """

    def create_features(self, layers_n):
        """
        初始化网络feature部分
        :param layers_n: 存储网络结构的list
        :return: 网络feature部分各层tensor
        """
        raise NotImplementedError()

    def save_hyperparameters(self):
        return dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))


class FCFeatures(FeaturesNet):
    """
    全链接feature网络
    """

    def __init__(self, HIDDEN_LAYERS=(30, 20)):
        """
        :param HIDDEN_LAYERS: 每层隐藏层神经元数量
        """
        self.HIDDEN_LAYERS = list(HIDDEN_LAYERS)

    def create_features(self, layers_n):
        layers_n += self.HIDDEN_LAYERS
        with tf.name_scope('input_layer'):
            layers = [tf.placeholder(tf.float32, [None] + layers_n[0])]  # 输入层
        layers += create_FC_stream(layers[-1], layers_n[1:], 'feature_layer')
        return layers


class ConvFeatures(FeaturesNet):
    def __init__(self, CONVOLUTION, FC_CONNECTION_N):
        """
        :param CONVOLUTION: [('conv', {'weights': ..., 'strides': ...}),('pooling', {'ksize': ..., 'strides': ...})]
        :param FC_CONNECTION_N: 最后全连接层
        """
        self.CONVOLUTION = list(CONVOLUTION)
        self.FC_CONNECTION_N = FC_CONNECTION_N

    def __conv2d(self, layer_t):
        """
        生成卷积层
        :param layer_t: 卷积层参数
        """
        W = tf.Variable(tf.truncated_normal(layer_t['weight'], stddev=0.1), name='kernel')
        b = tf.Variable(tf.constant(0.1, shape=(layer_t['weight'][3],)), name='bias')
        self.layers.append(
            tf.nn.relu(
                tf.nn.conv2d(self.layers[-1], W, strides=[1] + list(layer_t['strides']) + [1], padding='SAME') + b))

    def __max_pool(self, layer_t):
        """
        生成最大池化层
        :param layer_t: 池化层参数
        """
        self.layers.append(tf.nn.max_pool(self.layers[-1], ksize=[1] + list(layer_t['ksize']) + [1],
                                          strides=[1] + list(layer_t['strides']) + [1], padding='SAME'))

    def create_features(self, layers_n):
        with tf.name_scope('input_layer'):
            self.layers = [tf.placeholder(tf.float32, [None] + layers_n[0])]  # 输入层
        with tf.name_scope('feature_layer'):
            for i, (layer_t, layer) in enumerate(self.CONVOLUTION):
                if layer_t == 'conv':  # 卷积层
                    with tf.name_scope('conv_layer{}'.format(i)):
                        self.__conv2d(layer)
                elif layer_t == 'pooling':  # 池化层
                    with tf.name_scope('pooling_layer{}'.format(i)):
                        self.__max_pool(layer)
            # 加一层全连接层，是之后的行为一致
            self.layers += create_FC_stream(
                tf.reshape(self.layers[-1], shape=[-1, np.prod(self.layers[-1].shape.as_list()[1:])]),
                self.FC_CONNECTION_N, 'fully_connected')
        # 更新layers_n
        layers_n += self.CONVOLUTION + [self.FC_CONNECTION_N]
        return self.layers


class QLayerNet:
    """
    网络Q值部分基类
    """

    def create_Q_layers(self, action_n, layers_n, layers):
        """
        初始化网络Q值部分
        :param action_n: 动作个数
        :param layers_n: 网络结构list
        :param layers: 之前网络
        :return: 网络Q值部分个层tensor
        """
        raise NotImplementedError()

    def save_hyperparameters(self):
        return dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))


class OriginalQLayer(QLayerNet):
    """
    DQN原来的网络结构
    """

    def __init__(self, Q_HIDDEN_LAYERS=(20, 10)):
        """
        :param Q_HIDDEN_LAYERS: 每层隐藏层神经元数量
        """
        self.Q_HIDDEN_LAYERS = list(Q_HIDDEN_LAYERS)

    def create_Q_layers(self, action_n, layers_n, layers):
        layers += create_FC_stream(layers[-1], self.Q_HIDDEN_LAYERS, 'Q_hidden_layer')
        layers_n += self.Q_HIDDEN_LAYERS + [action_n]
        with tf.name_scope('Q_layer'):
            layers.append(create_z(layers[-1], layers_n[-1]))  # 输出层，没有激活函数


class DuelingDQN(QLayerNet):
    """
    Dueling网络
    """

    def __init__(self, STATE_HIDDEN_LAYERS, ADVANTAGE_HIDDEN_LAYERS):
        """
        :param STATE_HIDDEN_LAYERS: 状态流隐藏层神经元数量
        :param ADVANTAGE_HIDDEN_LAYERS: 优势流隐藏层神经元数量
        """
        self.STATE_HIDDEN_LAYERS = list(STATE_HIDDEN_LAYERS)
        self.ADVANTAGE_HIDDEN_LAYERS = list(ADVANTAGE_HIDDEN_LAYERS)

    def create_Q_layers(self, action_n, layers_n, layers):
        # 状态价值隐藏层
        state_stream = create_FC_stream(layers[-1], self.STATE_HIDDEN_LAYERS,
                                        'state_hidden_layer')
        # 状态价值，不用RELU
        with tf.name_scope('state_value'):
            state_stream.append(create_z(state_stream[-1], 1))

        # 动作优势隐藏层
        advantage = create_FC_stream(layers[-1], self.ADVANTAGE_HIDDEN_LAYERS,
                                     'advantage_hidden_layer')
        # 动作优势，不用RELU
        with tf.name_scope('action_advantage'):
            advantage.append(create_z(advantage[-1], action_n))

        layers.append([state_stream, advantage])  # 并行加入网络中
        # 输出Q层
        with tf.name_scope('Q_layer'):
            layers.append(
                state_stream[-1] + advantage[-1] - tf.reduce_mean(advantage[-1], axis=1, keep_dims=True))
        # 每层网络中神经元个数
        layers_n += [[self.STATE_HIDDEN_LAYERS + [1], self.ADVANTAGE_HIDDEN_LAYERS + [action_n]]] \
                    + [action_n]
