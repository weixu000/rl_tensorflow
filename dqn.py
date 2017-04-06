import tensorflow as tf
import numpy as np
from collections import deque
import gym
import json
import heapq
import time


class FCQN:
    """
    普通DQN结构网络
    """
    WRITE_SUMMARY = False

    def __init__(self, env, env_name):
        self.HIDDEN_LAYERS = [30, 20, 20, 10]
        self.LEARNING_RATE = 2E-3
        self.INITIAL_EPS = 1
        self.FINAL_EPS = 1E-3
        self.EPS_DECAY_RATE = 0.9
        self.EPS_DECAY_STEP = 1000
        self.MEMORY_SIZE = 10000
        self.GAMMA = 1
        self.BATCH_SIZE = 100
        # 要求状态为向量，动作离散
        assert type(env.action_space) == gym.spaces.discrete.Discrete and \
               type(env.observation_space) == gym.spaces.box.Box

        # 建立若干成员变量
        self.env = env
        self.log_dir = '/'.join(['log', env_name, self.NAME, time.strftime('%m-%d-%H-%M')]) + '/'
        self.n_episodes = 0
        self.n_timesteps = 0  # 总步数

        # 构建网络
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_graph()
            self.create_loss()
            if self.WRITE_SUMMARY: self.create_summary()
            self.compute_grad = [x[0] for x in self.compute_grad]  # 留下梯度，去掉变量，方便train_sess
            self.init = tf.global_variables_initializer()

    def create_FC_stream(self, input, layers_n, name):
        """
        :param input: 输入
        :param layers_n: 
        :param name: 
        :return: 
        """
        layers = [input]
        for i, (lst, cur) in enumerate(zip(layers_n[:-1], layers_n[1:])):
            with tf.name_scope('{}{}'.format(name, i)):
                layers.append(tf.nn.relu(self.create_z(layers[-1], lst, cur)))  # 隐藏层用RELU
        return layers[1:]

    def create_graph(self):
        """
        建立网络
        """
        self.layers_n = [self.env.observation_space.shape[0]] + self.HIDDEN_LAYERS + [
            self.env.action_space.n]  # 每层网络中神经元个数
        with tf.name_scope('input_layer'):
            self.layers = [tf.placeholder(tf.float32, [None, self.layers_n[0]])]  # 输入层
        self.layers += self.create_FC_stream(self.layers[-1], self.layers_n[:-1], 'hidden_layer')
        with tf.name_scope('Q_layer'):
            self.layers.append(
                self.create_z(self.layers[-1], self.layers_n[-2], self.layers_n[-1]))  # 输出层，没有激活函数

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
        self.train()

    def train_sess(self, sess, batch, batch_ind):
        """
        输入数据，训练网络
        """
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        sess.run(self.apply_grad, feed_dict={self.layers[0]: state_batch,
                                             self.action_onehot: action_batch,
                                             self.y: y_batch})

    def save_hyperparameters(self):
        """
        保存参数到parameters.json
        """
        with open(self.log_dir + 'parameters.json', 'w') as f:
            json.dump(dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items())), f,
                      indent=4, sort_keys=True)


class DuelingFCQN(FCQN):
    """
    Dueling网络
    """

    def __init__(self, env, env_name):
        self.HIDDEN_LAYERS = [30, 20]
        self.STATE_HIDDEN_LAYERS = [10, 5]
        self.ADVANTAGE_HIDDEN_LAYERS = [10, 5]
        super().__init__(env, env_name)

    def create_graph(self):
        """
        建立网络
        """
        # 每层网络中神经元个数
        self.layers_n = [self.env.observation_space.shape[0]] + self.HIDDEN_LAYERS \
                        + [[self.STATE_HIDDEN_LAYERS + [1],
                            self.ADVANTAGE_HIDDEN_LAYERS + [self.env.action_space.n]]] \
                        + [self.env.action_space.n]

        with tf.name_scope('input_layer'):
            self.layers = [tf.placeholder(tf.float32, [None, self.layers_n[0]])]  # 输入层
        self.layers += self.create_FC_stream(self.layers[-1], self.layers_n[:-2], 'hidden_layer')  # 隐藏层

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


class RandomReplay(DuelingFCQN):
    """
    随机经验回放
    """

    def __init__(self, env, env_name):
        super().__init__(env, env_name)
        self.memory = deque(maxlen=self.MEMORY_SIZE)

    def perceive(self, state, action, reward, nxt_state, done):
        # 将处理后的经验加入记忆中，超出存储的自动剔除
        self.memory.append(self.process_experience(state, action, reward, nxt_state, done))
        super().perceive(state, action, reward, nxt_state, done)

    def sample_memory(self):
        """
        随机抽取batch_size个记忆，分别建立状态、动作、Q、下一状态、完成与否的矩阵（一行对应一个记忆）
        """
        batch_ind = np.random.choice(len(self.memory), min(self.BATCH_SIZE, len(self.memory)), False)
        batch = [x for i, x in enumerate(self.memory) if i in batch_ind]
        return [np.array([m[i] for m in batch]) for i in range(5)], None  # RandomReplay不需要batch_ind


class RankBasedPrioritizedReplay(DuelingFCQN):
    """
    根据优先级抽取记忆回放
    """

    class Experience:
        """
        存储带优先级的经验
        """

        def __lt__(self, other): return self.priority < other.priority

        def __init__(self, priority, data):
            self.priority = priority
            self.data = data

    def __init__(self, env, env_name):
        self.ALPHA = 3  # 幂分布的指数
        self.SORT_WHEN = 5000  # 何时完全排序记忆
        super().__init__(env, env_name)
        self.memory = []
        # 最近的记忆，尚未训练
        self.recent_memory = []

    def perceive(self, state, action, reward, nxt_state, done):
        # 插入最近的记忆
        experience = list(self.process_experience(state, action, reward, nxt_state, done))
        self.recent_memory.append(experience)

        # 记忆过多，则删除误差最小的
        if len(self.memory) >= self.MEMORY_SIZE: heapq.heappop(self.memory)

        # 记忆较多时，进行排序
        if not self.n_timesteps % self.SORT_WHEN: self.memory.sort()

        super().perceive(state, action, reward, nxt_state, done)

    def sample_memory(self):
        # 按幂分布取出记忆
        batch_ind = np.random.power(self.ALPHA, min(self.BATCH_SIZE, len(self.memory))) * len(self.memory)
        batch_ind = list(set(batch_ind.astype(int)))

        # 将最近没有训练的记忆加入
        batch_ind += list(range(len(self.memory), len(self.recent_memory) + len(self.memory)))
        self.memory += [self.Experience(None, x) for x in self.recent_memory]
        self.recent_memory = []

        batch = [x.data for i, x in enumerate(self.memory) if i in batch_ind]
        return [np.array([m[i] for m in batch]) for i in range(5)], batch_ind

    def train_sess(self, sess, batch, batch_ind):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        # 先计算最近的经验
        none_ind = np.array([[i, j] for i, j in enumerate(batch_ind) if self.memory[j].priority is None])
        loss = sess.run(self.loss_vec, feed_dict={self.layers[0]: state_batch[none_ind.T[0]],
                                                  self.action_onehot: action_batch[none_ind.T[0]],
                                                  self.y: y_batch[none_ind.T[0]]})
        for i, j in enumerate(none_ind): self.memory[j[1]].priority = loss[i]

        super().train_sess(sess, batch, batch_ind)

        # 计算误差减少量，以此作为优先级
        errors = np.array([self.memory[i].priority for i in batch_ind]) - sess.run(self.loss_vec,
                                                                                   feed_dict={
                                                                                       self.layers[0]: state_batch,
                                                                                       self.action_onehot: action_batch,
                                                                                       self.y: y_batch})

        # 更新经验
        for i, x in zip(batch_ind, errors): self.memory[i].priority = x
        heapq.heapify(self.memory)


class OriginalFCQN(RandomReplay):
    """
    Nature DQN
    """
    NAME = 'Original'

    def __init__(self, env, env_name):
        super().__init__(env, env_name)

        # 目前只用一个网络
        self.sessions = [(tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir, self.graph))]
        self.sessions[0][0].run(self.init)

    def __getitem__(self, state):
        """
        给定单个状态，计算所有动作的Q值
        """
        return self.sessions[0][0].run(self.layers[-1], feed_dict={self.layers[0]: [self.normalize_state(state)]})[0]

    def train(self):
        """
        随机取出记忆中的经验训练网络
        """
        batch, batch_ind = self.sample_memory()
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        # 计算公式中的maxQ，如果完成设为0
        nxt_qs = np.max(self.sessions[0][0].run(self.layers[-1], feed_dict={self.layers[0]: nxt_state_batch}), axis=1)
        nxt_qs[done_batch] = 0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        self.train_sess(self.sessions[0][0], batch, batch_ind)


class DoubleFCQN(RankBasedPrioritizedReplay):
    """
    Double DQN
    """
    NAME = 'Double'

    def __init__(self, env, env_name):
        super().__init__(env, env_name)

        self.sessions = [(tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir + 'Q1/', self.graph)),
                         (tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir + 'Q2/', self.graph))]
        for s in self.sessions: s[0].run(self.init)

    def __getitem__(self, state):
        """
        给定单个状态，计算所有动作的Q值
        """
        ret = np.zeros((self.layers_n[-1],))
        for s in self.sessions:
            ret += s[0].run(self.layers[-1], feed_dict={self.layers[0]: [self.normalize_state(state)]})[0]

        return ret / 2

    def train(self):
        """
        随机取出记忆中的经验训练网络
        """
        batch, batch_ind = self.sample_memory()
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        # 任取一个训练
        if np.random.rand() >= 0.5:
            (sess1, writer1), (sess2, writer2) = self.sessions
        else:
            (sess2, writer2), (sess1, writer1) = self.sessions

        # sess1计算argmaxQ的onehot表示
        a = np.eye(self.layers_n[-1])[
            np.argmax(sess1.run(self.layers[-1], feed_dict={self.layers[0]: nxt_state_batch}), axis=1)]
        # sess2计算Q
        nxt_qs = sess2.run(self.action_value, feed_dict={self.layers[0]: nxt_state_batch, self.action_onehot: a})
        nxt_qs[done_batch] = 0  # 如果完成设为0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        self.train_sess(sess1, batch, batch_ind)
