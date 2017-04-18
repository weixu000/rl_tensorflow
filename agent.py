import json
import numpy as np
import tensorflow as tf


class QNetwork:
    """
    普通DQN结构网络
    """
    WRITE_SUMMARY = False

    def __init__(self, env, log_dir):
        self.LEARNING_RATE = 2E-3
        self.INITIAL_EPS = 0.5
        self.FINAL_EPS = 0
        self.EPS_DECAY_RATE = 0.95
        self.EPS_DECAY_STEP = 2000
        self.GAMMA = 1

        self.env = env
        self.log_dir = log_dir
        self.n_episodes = 0
        self.n_timesteps = 0  # 总步数

        # 构建网络
        self.graph = tf.Graph()
        self.layers_n = []
        self.layers = []
        with self.graph.as_default():
            self.create_feature()
            self.create_Q_layers()
            self.create_loss()
            if self.WRITE_SUMMARY: self.create_summary()
            # self.compute_grad = [x[0] for x in self.compute_grad]  # 留下梯度，去掉变量，方便train_sess
            self.init = tf.global_variables_initializer()

    def create_z(self, prev, row, col):
        """
        建立z=x*W+b
        """
        W = tf.Variable(tf.truncated_normal([row, col]), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=[col]), name='biases')  # 正偏置促进学习
        if self.WRITE_SUMMARY:
            tf.summary.histogram('W', W)
            tf.summary.histogram('b', b)
        return tf.matmul(tf.reshape(prev, [-1, row]), W) + b

    def create_FC_stream(self, input, layers_n, name):
        layers = [input]
        for i, (lst, cur) in enumerate(zip(layers_n[:-1], layers_n[1:])):
            with tf.name_scope('{}{}'.format(name, i)):
                layers.append(tf.nn.relu(self.create_z(layers[-1], lst, cur)))  # 隐藏层用RELU
        return layers[1:]

    def create_feature(self):
        """
        特征层
        """
        raise NotImplementedError()

    def create_Q_layers(self):
        """
        网络特征层之后部分
        """
        raise NotImplementedError()

    def create_loss(self):
        """
         生成误差，建立优化器
        """
        # 训练用到的指示值y（相当于图像识别的标签），action（最小化特定动作Q的误差），loss
        with tf.name_scope('train'):
            self.y = tf.placeholder(tf.float32, [None], name='target')
            self.action_onehot = tf.placeholder(tf.float32, [None, self.layers_n[-1]], name='action_onehot')
            self.action_value = tf.reduce_sum(self.layers[-1] * self.action_onehot, reduction_indices=1,
                                              name='sample_Q')
            self.loss_vec = self.y - self.action_value
            self.loss = tf.reduce_mean(tf.square(self.loss_vec), name='loss')

            optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
            self.compute_grad = optimizer.compute_gradients(self.loss)
            self.apply_grad = optimizer.apply_gradients(self.compute_grad)

    def create_summary(self):
        """
        tensorboard记录
        """
        if self.WRITE_SUMMARY:
            # 将误差loss和梯度grad都记录下来
            with tf.name_scope('evaluate'):
                tf.summary.scalar('_loss', self.loss)
                for grad, var in self.compute_grad: tf.summary.histogram(var.name + '_grad', grad)
                self.summary = tf.summary.merge_all()

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
        """
        ret = np.zeros((self.layers_n[-1],))
        for s in self.sessions:
            ret += s[0].run(self.layers[-1], feed_dict={self.layers[0]: [self.normalize_state(state)]})[0]

        return ret / len(self.sessions)

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
        return self.greedy_action(state) if np.random.rand() > eps else self.env.action_space.sample()

    def process_experience(self, state, action, reward, nxt_state, done):
        """
        处理单个经验
        """
        # 将动作单个数值转化成onehot向量：2变成[0,0,1,0,0]
        onehot = np.zeros((self.layers_n[-1]))
        onehot[action] = 1

        # 正规化状态
        state = self.normalize_state(state)
        nxt_state = self.normalize_state(nxt_state)

        return state, onehot, reward, nxt_state, done

    def perceive(self, state, action, reward, nxt_state, done):
        """
        接受经验
        """
        self.n_timesteps += 1
        if done:
            self.n_episodes += 1
            if self.WRITE_SUMMARY:
                for sess, writer in self.sessions:
                    writer.add_summary(sess.run(self.summary), self.n_episodes)

    def train_sess(self, sess, batch):
        """
        输入数据，训练网络
        """
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        sess.run(self.apply_grad, feed_dict={self.layers[0]: state_batch,
                                             self.action_onehot: action_batch,
                                             self.y: y_batch})

    def train(self, batch):
        """
        用batch训练网络
        """
        raise NotImplementedError()

    def save_hyperparameters(self):
        """
        保存参数到parameters.json
        """
        with open(self.log_dir + 'parameters.json', 'w') as f:
            json.dump(dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items())), f,
                      indent=4, sort_keys=True)
