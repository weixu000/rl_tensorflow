import numpy as np
import tensorflow as tf
from DQN.network import FeaturesNet, QLayerNet, EnvModel
import matplotlib.pyplot as plt
import json
import os
from collections import deque, defaultdict
from functools import reduce


class Agent:
    def __init__(self, env, observation_shape, obervation_range, observations_in_state, action_n, log_dir):
        """
        :param env: 环境
        :param observation_shape: observation数组尺寸
        :param obervation_range: obeservation数组范围
        :param observations_in_state: 多少个observation组成一个state
        :param action_n: 动作个数
        :param log_dir: log文件路径
        """
        self.env = env

        self.observation_shape = list(observation_shape)
        self.observation_range = obervation_range
        self.observations_in_state = observations_in_state

        if len(self.observation_shape) == 2:
            self.observation_shape = self.observation_shape + [1]
        if self.observations_in_state == 1:
            self.state_shape = self.observation_shape  # observation视作state
        else:
            if len(self.observation_shape) == 1:
                self.state_shape = [self.observation_shape[0] * self.observations_in_state]  # 向量observation加长视作state
            else:
                self.state_shape = self.observation_shape[:2] + [
                    self.observation_shape[2] * self.observations_in_state]  # 矩阵observation加一维视作state

        self.action_n = action_n
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.ckpt_file = self.log_dir + 'ckpts/' + 'session.ckpt'

        self._components = []

    def normalize_observation(self, observation):
        """
        正规化状态
        """
        if self.observation_range:
            return (observation - self.observation_range[0]) / (
                self.observation_range[1] - self.observation_range[0])
        else:
            return observation

    def exploit(self, max_timesteps=None, render=False):
        """
        充分利用Q值进行一个episode
        :return returns
        """
        raise NotImplementedError()

    def explore(self, max_timesteps=None, render=False):
        """
        探索一个episode
        :return returns
        """
        raise NotImplementedError()

    def save_hyperparameters(self):
        """
        保存参数到parameters.json
        """
        params = dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))
        for i in self._components:
            params[i.__class__.__name__] = i.save_hyperparameters()
        with open(self.log_dir + 'parameters.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

    def init_session(self):
        """
        恢复网络
        """
        if os.path.exists(os.path.split(self.ckpt_file)[0]):
            # 读取已经保存的网络
            self.saver.restore(self._session, self.ckpt_file)
        else:
            self._session.run(tf.global_variables_initializer())

    def save_session(self):
        """
        保存网络
        """
        os.makedirs(os.path.split(self.ckpt_file)[0], exist_ok=True)
        self.saver.save(self._session, self.ckpt_file)

    def plot_returns(self):
        raise NotImplementedError()


class DDQN(Agent):
    def __init__(self, env, observation_shape, obervation_range, observations_in_state, action_n, log_dir,
                 features: FeaturesNet, Q_layers: QLayerNet,
                 GAMMA=1.0, LEARNING_RATE=2E-3,
                 MEMORY_SIZE=5000, BATCH_SIZE=100, TRAIN_REPEAT=2,
                 EPS_INITIAL=1, EPS_END=0.1, EPS_STEP=1E-2):
        """
        :param features: 网络feature部分
        :param Q_layers: 网络Q值部分
        :param GAMMA: 衰减因子
        :param LEARNING_RATE: 学习速率
        :param TRAIN_REPEAT: 每次replay重复的batch
        :param EPS_INITIAL: 初始epsilon
        :param EPS_STEP: epsilon衰减率
        :param EPS_END: epsilon终值
        """

        Agent.__init__(self, env, observation_shape, obervation_range, observations_in_state, action_n, log_dir)
        self.GAMMA = GAMMA
        self.LEARNING_RATE = LEARNING_RATE
        self.TRAIN_REPEAT = TRAIN_REPEAT
        self.eps = EPS_INITIAL
        self.EPS_STEP = EPS_STEP
        self.EPS_END = EPS_END
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        self._features = features
        self._Q_layers = Q_layers
        self._components += [features, Q_layers]

        self.__memory = deque(maxlen=self.MEMORY_SIZE)

        self.__explore_returns = []
        self.__exploit_returns = []

        self.create_network(2)
        self.__heads = [self._layers[-1][0][-1], self._layers[-1][1][-1]]

    def create_network(self, n_heads):
        """
        创建Q网络
        :param n_heads: Q值个数
        """
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._layers, self._layers_n = self._features.create(self.state_shape)  # 初始化网络feature部分

            with tf.name_scope('train'):
                self.__y = tf.placeholder(tf.float32, [None], name='target')
                self.__action_onehot = tf.placeholder(tf.float32, [None, self.action_n], name='action_onehot')

            qs, qs_n = [], []
            self._loss = []
            self._train_step = []
            with tf.name_scope('train'):
                self.__y = tf.placeholder(tf.float32, [None], name='target')
                self.__action_onehot = tf.placeholder(tf.float32, [None, self.action_n], name='action_onehot')
            for i in range(n_heads):
                _ = self._Q_layers.create_Q_layers(self.action_n, self._layers[-1])  # 初始化网络Q值部分
                qs.append(_[0])
                qs_n.append(_[1])
                with tf.name_scope('train_{}'.format(i)):
                    action_value = tf.reduce_sum(qs[-1][-1] * self.__action_onehot, reduction_indices=1,
                                                 name='sample_Q')
                    self._loss.append(tf.reduce_mean(tf.square(self.__y - action_value), name='loss'))
                    self._train_step.append(tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self._loss[-1]))
            self._layers.append(qs)
            self._layers_n.append(qs_n)

            self._session = tf.Session()
            self.saver = tf.train.Saver()

            # 读取已经保存的网络
            self.init_session()

    def _exploit_action(self):
        if len(self._observation_buff) >= self.observations_in_state:
            state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
            return np.argmax(reduce(lambda x, y: x + y,
                                    [self._session.run(h, feed_dict={self._layers[0]: [state]})[0]
                                     for h in self.__heads]))
        else:
            return np.random.choice(self.action_n)

    def _do_exploit(self, max_timesteps=None, render=False):
        observation = self.env.reset()
        self._observation_buff = []

        ret = 0
        i_timesteps = 0
        while True:
            if render: self.env.render()
            self._observation_buff.append(self.normalize_observation(observation))

            # 选择动作
            action = self._exploit_action()
            observation, reward, done, _ = self.env.step(action)

            ret += reward
            i_timesteps += 1
            if done or (max_timesteps and i_timesteps == max_timesteps):
                return ret

    def exploit(self, max_timesteps=None, render=False):
        ret = self._do_exploit(max_timesteps, render)
        self.__exploit_returns.append(ret)
        return ret

    def _explore_action(self):
        if len(self._observation_buff) >= self.observations_in_state and np.random.rand() > self.eps:
            action = self._exploit_action()
        else:
            action = np.random.choice(self.action_n)
        self.eps = max(self.eps - self.EPS_STEP, self.EPS_END)
        return action

    def _perceive(self, state, action, reward, nxt_state, done):
        onehot = np.zeros((self.action_n,))  # 将动作单个数值转化成onehot向量
        onehot[action] = 1
        self.__memory.append((state, onehot, reward, nxt_state, done))

    def _do_explore(self, max_timesteps=None, render=False):
        observation = self.env.reset()
        self._observation_buff = []

        ret = 0
        i_timesteps = 0
        while True:
            if render: self.env.render()
            self._observation_buff.append(self.normalize_observation(observation))

            # 实行动作
            action = self._explore_action()
            nxt_observation, reward, done, _ = self.env.step(action)

            # 接受经验
            if len(self._observation_buff) > self.observations_in_state:
                state = np.array(self._observation_buff[:self.observations_in_state]).reshape(self.state_shape)
                nxt_state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
                self._perceive(state, self.__prev_action, reward, nxt_state, done)
                self.replay()
                del self._observation_buff[0]

            # 保存这个动作，下一次存储记忆用
            self.__prev_action = action

            observation = nxt_observation
            ret += reward
            i_timesteps += 1
            if done or (max_timesteps and i_timesteps == max_timesteps):
                return ret

    def explore(self, max_timesteps=None, render=False):
        ret = self._do_explore(max_timesteps, render)
        self.__explore_returns.append(ret)
        return ret

    def replay(self):
        for _ in range(self.TRAIN_REPEAT):
            batch_ind = np.random.choice(len(self.__memory), self.BATCH_SIZE)
            batch = list(map(lambda i: self.__memory[i], batch_ind))
            batch = [np.array([m[i] for m in batch]) for i in range(5)]

            # 任取一个训练
            q1, q2 = np.random.permutation(2)
            self.double_q(self.__heads[q1], self.__heads[q2],
                          batch, self._train_step[q1])

    def double_q(self, q1, q2, batch, train_step):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch
        q1, q2 = self._session.run([q1, q2], feed_dict={self._layers[0]: nxt_state_batch})
        # q1计算argmaxQ的onehot表示
        a = np.eye(self.action_n)[np.argmax(q1, axis=1)]
        # q2计算Q
        nxt_qs = np.sum(q2 * a, axis=1)
        nxt_qs[done_batch] = 0  # 如果完成设为0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        self._session.run(train_step, feed_dict={self._layers[0]: state_batch,
                                                 self.__action_onehot: action_batch,
                                                 self.__y: y_batch})

    def plot_returns(self):
        plt.plot(self.__explore_returns)
        plt.title('Explore returns')
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.savefig(self.log_dir + 'explore_returns.png')
        plt.clf()

        plt.plot(self.__exploit_returns)
        plt.title('Exploit returns')
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.savefig(self.log_dir + 'exploit_returns.png')
        plt.clf()


class BootstrappedDDQN(DDQN):
    def __init__(self, env, observation_shape, obervation_range, observations_in_state, action_n, log_dir,
                 features: FeaturesNet, Q_layers: QLayerNet,
                 GAMMA=1, LEARNING_RATE=1E-3,
                 MEMORY_SIZE=5000, BATCH_SIZE=100, TRAIN_REPEAT=2,
                 N_HEADS=8):
        """
        :param N_HEADS: heads数
        """
        Agent.__init__(self, env, observation_shape, obervation_range, observations_in_state, action_n, log_dir)

        self.LEARNING_RATE = LEARNING_RATE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAIN_REPEAT = TRAIN_REPEAT
        self.N_HEADS = N_HEADS

        self.__memory = deque(maxlen=self.MEMORY_SIZE)
        self._features = features
        self._Q_layers = Q_layers
        self._components += [features, Q_layers]

        self.__head_returns = [[] for _ in range(self.N_HEADS)]

        self.create_network(self.N_HEADS * 2)
        self.__heads = [[self._layers[-1][i][-1], self._layers[-1][i + N_HEADS][-1]] for i in range(N_HEADS)]
        self.__best_head = self.__heads[0]
        self.__explore_head = self.__heads[0]

    def __select_action(self, head):
        if len(self._observation_buff) >= self.observations_in_state:
            state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
            return np.argmax(reduce(lambda x, y: x + y,
                                    [self._session.run(h, feed_dict={self._layers[0]: [state]})[0]
                                     for h in head]))
        else:
            return np.random.choice(self.action_n)

    def _exploit_action(self):
        return self.__select_action(self.__best_head)

    def exploit(self, max_timesteps=None, render=False):
        # 选平均回报最大的head
        # best_head = np.argmax(np.nan_to_num([np.average(self.__head_returns[i][-2:]) for i in range(self.N_HEADS)]))
        best_head = np.argmax(np.nan_to_num([np.average(self.__head_returns[i][-1:]) for i in range(self.N_HEADS)]))
        self.__best_head = self.__heads[best_head]
        # print('Exploit head {}'.format(best_head))

        return DDQN._do_exploit(self, max_timesteps, render)

    def _explore_action(self):
        return self.__select_action(self.__explore_head)

    def _perceive(self, state, action, reward, nxt_state, done):
        onehot = np.zeros((self.action_n,))  # 将动作单个数值转化成onehot向量
        onehot[action] = 1
        self.__memory.append((state, onehot, reward, nxt_state, done, self.bootstrap_mask()))

    def explore(self, max_timesteps=None, render=False):
        # ps = np.nan_to_num([np.average(self.__head_returns[i]) for i in range(self.N_HEADS)])
        # ps = np.exp(np.nan_to_num(2 * ps / ps.sum()))
        # explore_n = np.random.choice(self.N_HEADS, p=ps / ps.sum())
        explore_n = np.random.choice(self.N_HEADS)
        self.__explore_head = self.__heads[explore_n]
        # print('Explore head {}'.format(explore_n))

        ret = DDQN._do_explore(self, max_timesteps, render)
        # 更新heads回报
        self.__head_returns[explore_n].append(ret)
        return ret

    def bootstrap_mask(self):
        return np.random.choice([0, 1], self.N_HEADS)

    def replay(self):
        for _ in range(self.TRAIN_REPEAT):
            batch_ind = np.random.choice(len(self.__memory), self.BATCH_SIZE)
            batch = list(map(lambda i: self.__memory[i], batch_ind))
            batch = [np.array([m[i] for m in batch]) for i in range(6)]

            # 任取一个训练
            q1, q2 = np.random.permutation(2)

            heads = defaultdict(lambda: [])
            for x, y in np.transpose(np.nonzero(batch[5])): heads[y].append(x)
            for x, y in heads.items():
                self.double_q(self.__heads[x][q1], self.__heads[x][q2],
                              [batch[i][y] for i in range(5)],
                              self._train_step[x + q1 * self.N_HEADS])

    def plot_returns(self):
        for i, ret in enumerate(self.__head_returns):
            plt.plot(ret, label='Heads {}'.format(i))
        plt.title('Returns of each heads')
        plt.xlabel('Train episodes')
        plt.ylabel('Returns')
        plt.legend()
        plt.savefig(self.log_dir + 'head_returns.png')
        plt.clf()


class ModelBasedDDQN(BootstrappedDDQN):
    def __init__(self, env, observation_shape, obervation_range, observations_in_state, action_n, log_dir,
                 features: FeaturesNet, Q_layers: QLayerNet, model: EnvModel,
                 GAMMA=1, LEARNING_RATE=1E-3,
                 MEMORY_SIZE=5000, BATCH_SIZE=50, TRAIN_REPEAT=2,
                 N_HEADS=10):
        BootstrappedDDQN.__init__(self, env, observation_shape, obervation_range, observations_in_state, action_n,
                                  log_dir,
                                  features, Q_layers,
                                  GAMMA, LEARNING_RATE,
                                  MEMORY_SIZE, BATCH_SIZE, TRAIN_REPEAT,
                                  N_HEADS)
        self.model = model
        self._components += [model]
        self.model.create_model(self.state_shape)

    def _perceive(self, state, action, reward, nxt_state, done):
        bonus = self.model.perceive(state, action, reward, nxt_state, done)
        BootstrappedDDQN._perceive(self, state, action, reward + bonus, nxt_state, done)
