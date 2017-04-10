import tensorflow as tf
from agent import QNetwork


class FCQN(QNetwork):
    def __init__(self, env, env_name):
        self.HIDDEN_LAYERS = [30, 20]
        super().__init__(env, env_name)

    def create_feature(self):
        self.layers_n += [self.env.observation_space.shape[0]] + self.HIDDEN_LAYERS  # 每层网络中神经元个数
        with tf.name_scope('input_layer'):
            self.layers = [tf.placeholder(tf.float32, [None, self.layers_n[0]])]  # 输入层
        self.layers += self.create_FC_stream(self.layers[-1], self.layers_n, 'hidden_layer')
