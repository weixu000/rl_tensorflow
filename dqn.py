import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import json
import time


class FCQN:
    SUMMARY_WHEN = 500

    def __init__(self, env, hidden_layers, env_name, log_dir):
        # 要求状态为向量，动作离散
        assert type(env.action_space) == gym.spaces.discrete.Discrete and \
               type(env.observation_space) == gym.spaces.box.Box

        # 建立若干成员变量
        self.env = env
        self.log_dir = '/'.join(['log', env_name, self.NAME, log_dir, time.strftime('%m-%d-%H-%M')]) + '/'
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.eps = self.INITIAL_EPS

        self.create_graph(hidden_layers)

    def create_graph(self, hidden_layers):
        # 构建网络
        self.layers_n = [self.env.observation_space.shape[0]] + hidden_layers + [self.env.action_space.n]  # 每层网络中神经元个数
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('input_layer'):
                self.layers = [tf.placeholder(tf.float32, [None, self.layers_n[0]])]  # 输入层
            for i, (lst, cur) in enumerate(zip(self.layers_n[:-2], self.layers_n[1:-1])):
                with tf.name_scope('hidden_layer{}'.format(i)):
                    self.layers.append(tf.nn.relu(self.create_FC_layer(lst, cur)))  # 隐藏层用RELU
            with tf.name_scope('output_layer'):
                self.layers.append(self.create_FC_layer(self.layers_n[-2], self.layers_n[-1]))  # 输出层，没有激活函数

            # 训练用到的指示值y（相当于图像识别的标签），action（最小化特定动作Q的误差），loss
            with tf.name_scope('train'):
                self.y = tf.placeholder(tf.float32, [None], name='target')
                self.action_onehot = tf.placeholder(tf.float32, [None, self.layers_n[-1]], name='action_onehot')
                self.action_value = tf.reduce_sum(self.layers[-1] * self.action_onehot, reduction_indices=1,
                                                  name='sample_Q')
                self.loss_vec = self.y - self.action_value
                self.loss = tf.reduce_mean(tf.square(self.loss_vec), name='loss')
                tf.summary.scalar('_loss', self.loss)
                self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
                self.learning_rate = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE, self.global_step,
                                                                self.DECAY_STEPS, self.DECAY_RATE, staircase=False,
                                                                name='learning_rate')
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                                                 self.global_step)

            with tf.name_scope('evaluate'):
                self.summary = tf.summary.merge_all()
                self.policy = tf.argmax(self.layers[-1], axis=1)
                self.state_value = tf.reduce_max(self.layers[-1], axis=1)

            self.init = tf.global_variables_initializer()

    def create_FC_layer(self, lst, cur):
        """
        建立单个全连接层的z=W*x+b
        """
        W = tf.Variable(tf.truncated_normal([lst, cur]), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=[cur]), name='biases')  # 正偏置促进学习
        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)
        return tf.matmul(self.layers[-1], W) + b

    def normalize_state(self, state):
        """
        正规化状态
        这个环境不能用
        """
        return state
        # return (state - self.env.observation_space.low) / (
        #     self.env.observation_space.high - self.env.observation_space.low)

    def greedy_action(self, state):
        """
        返回当前状态，当前Q值下认为最优的动作
        """
        return np.argmax(self[state])

    def epsilon_greedy(self, state):
        # self.eps -= (self.eps - self.FINAL_EPS) / self.EPS_DECAY  # 每次等比减小eps
        self.eps *= self.EPS_DECAY_RATE ** (1 / self.EPS_DECAY_STEP)
        return self.greedy_action(state) if np.random.rand() > self.eps else self.env.action_space.sample()

    def softmax(self, state):
        ts = np.exp(self[state])
        ts /= ts.sum()
        r = np.random.rand()
        for i, t in enumerate(ts):
            r -= t
            if r <= 0: return i

    def process_experience(self, state, action, reward, nxt_state, done):
        # 将动作单个数值转化成onehot向量：2变成[0,0,1,0,0]
        onehot = np.zeros((self.layers_n[-1]))
        onehot[action] = 1

        # 正规化状态
        state = self.normalize_state(state)
        nxt_state = self.normalize_state(nxt_state)
        return state, onehot, reward, nxt_state, done

    def perceive(self, state, action, reward, nxt_state, done):
        # 将处理后的经验加入记忆中，超出存储的自动剔除
        self.memory.append(self.process_experience(state, action, reward, nxt_state, done))
        self.train()

    def sample_memory(self):
        # 随机抽取batch_size个记忆，分别建立状态、动作、Q、下一状态、完成与否的矩阵（一行对应一个记忆）
        batch = random.sample(self.memory, min(self.BATCH_SIZE, len(self.memory)))
        return [[m[i] for m in batch] for i in range(5)]

    def train_sess(self, sess, writer, memory_batch, y_batch):
        """
        输入数据，训练网络
        """
        state_batch, action_batch, r_batch, nxt_state_batch, done_batch = memory_batch

        feed_dict = {self.layers[0]: state_batch,
                     self.action_onehot: action_batch,
                     self.y: y_batch}

        if sess.run(self.global_step) % self.SUMMARY_WHEN:
            sess.run(self.train_step, feed_dict=feed_dict)
        else:
            run_ans = sess.run([self.train_step, self.summary], feed_dict=feed_dict)
            writer.add_summary(run_ans[1], sess.run(self.global_step))

    def save_hyperparameters(self):
        with open(self.log_dir + 'parameters.json', 'w') as f:
            json.dump(dict(filter(lambda x: x[0][0].isupper(), self.__class__.__dict__.items())), f,
                      indent=4, sort_keys=True)

    def load_hyperparameters(self):
        try:
            with open(self.log_dir + '/parameters.json', 'r') as f:
                self.__class__.__dict__.update(json.load(f))
        except:
            pass

    def generate_state_mesh(self, resolution):
        assert len(resolution) == self.layers_n[0]
        resolution = [np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i], resolution[i])
                      for i in range(self.layers_n[0])]
        return np.meshgrid(resolution)

    def generate_policy(self, sess, resolution):
        mesh = self.generate_state_mesh(resolution)
        policy = sess.run(self.policy, feed_dict={self.layers[0]: np.transpose(mesh)})
        return np.reshape(policy, np.shape(mesh))

    def generate_state_value(self, sess, resolution):
        mesh = self.generate_state_mesh(resolution)
        value = sess.run(self.state_value, feed_dict={self.layers[0]: np.transpose(mesh)})
        return np.reshape(value, np.shape(mesh))


class OriginalFCQN(FCQN):
    NAME = 'Original'

    INITIAL_LEARNING_RATE = 0.001
    DECAY_STEPS = 3000
    DECAY_RATE = 0.95
    INITIAL_EPS = 0.8
    EPS_DECAY_RATE = 0.99
    EPS_DECAY_STEP = 5000
    MEMORY_SIZE = 10000
    GAMMA = 0.9
    BATCH_SIZE = 30
    TRAIN_REPEAT = 2

    # INITIAL_LEARNING_RATE = 0.0003
    # DECAY_STEPS = 50000
    # DECAY_RATE = 0.99
    # INITIAL_EPS = 1
    # EPS_DECAY_RATE = 0.95
    # EPS_DECAY_STEP = 2000
    # MEMORY_SIZE = 10000
    # SUMMARY_WHEN = 500
    # GAMMA = 0.95
    # BATCH_SIZE = 50
    # TRAIN_REPEAT = 2

    def __init__(self, env, hidden_layers, env_name, log_dir):
        super().__init__(env, hidden_layers, env_name, log_dir)

        # 输出日志
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.graph)

        # 初始化tensorflow
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def __getitem__(self, state):
        """
        给定单个状态，计算所有动作的Q值
        """
        return self.sess.run(self.layers[-1], feed_dict={self.layers[0]: [self.normalize_state(state)]})[0]

    def train(self):
        """
        随机取出记忆中的经验训练网络
        """
        # 重复训练train_repeat次
        for _ in range(self.TRAIN_REPEAT):
            state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch = self.sample_memory()

            # 计算公式中的maxQ，如果完成设为0
            nxt_qs = np.max(self.sess.run(self.layers[-1], feed_dict={self.layers[0]: nxt_state_batch}), axis=1)
            nxt_qs[done_batch] = 0
            y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

            self.train_sess(self.sess, self.summary_writer, batch, y_batch)


class DoubleFCQN(FCQN):
    NAME = 'Double'

    INITIAL_LEARNING_RATE = 0.001
    DECAY_STEPS = 5000
    DECAY_RATE = 0.9
    INITIAL_EPS = 0.5
    EPS_DECAY_RATE = 0.9
    EPS_DECAY_STEP = 5000
    MEMORY_SIZE = 10000
    GAMMA = 0.9
    BATCH_SIZE = 50
    TRAIN_REPEAT = 2

    def __init__(self, env, hidden_layers, env_name, log_dir):
        super().__init__(env, hidden_layers, env_name, log_dir)

        self.sess = [(tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir + 'Q1/', self.graph)),
                     (tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir + 'Q2/', self.graph))]

        # 初始化tensorflow
        for s in self.sess: s[0].run(self.init)

    def __getitem__(self, state):
        """
        给定单个状态，计算所有动作的Q值
        """
        ret = np.zeros((self.layers_n[-1],))
        for s in self.sess:
            ret += s[0].run(self.layers[-1], feed_dict={self.layers[0]: [self.normalize_state(state)]})[0]

        return ret / 2

    def train(self):
        """
        随机取出记忆中的经验训练网络
        """
        # 重复训练train_repeat次
        for _ in range(self.TRAIN_REPEAT):
            state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch = self.sample_memory()

            # 任取一个训练
            random.shuffle(self.sess)
            (sess1, writer1), (sess2, writer2) = self.sess

            # sess1计算argmaxQ的onehot表示
            a = np.eye(self.layers_n[-1])[
                np.argmax(sess1.run(self.layers[-1], feed_dict={self.layers[0]: nxt_state_batch}), axis=1)]
            # sess2计算Q
            nxt_qs = sess2.run(self.action_value, feed_dict={self.layers[0]: nxt_state_batch, self.action_onehot: a})
            nxt_qs[done_batch] = 0  # 如果完成设为0
            y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

            self.train_sess(sess1, writer1, batch, y_batch)
