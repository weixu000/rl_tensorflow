import numpy as np
import tensorflow as tf
from DQN.network import FeaturesNet, QLayerNet
from DQN.memory import Memory
from DQN.target import Target
import json
import os
from collections import deque, defaultdict
from functools import reduce


class Agent:
    def __init__(self, observation_shape, obervation_range, observations_in_state, action_n, log_dir):
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

    def normalize_observation(self, observation):
        """
        正规化状态
        """
        if self.observation_range:
            return (observation - self.observation_range[0]) / (
                self.observation_range[1] - self.observation_range[0])
        else:
            return observation

    def exploit(self, env, render=False):
        """
        充分利用Q值进行一个episode
        :return returns
        """
        raise NotImplementedError()

    def explore(self, env, render=False):
        """
        探索一个episode
        :return returns
        """
        raise NotImplementedError()

    def save_hyperparameters(self):
        """
        保存参数到parameters.json
        """
        return NotImplementedError()

    def load_sessions(self):
        """
        恢复网络
        """
        for i, sess in enumerate(self._sessions):
            self.saver.restore(sess, self.log_dir + 'ckpts/' + str(i) + '.ckpt')

    def save_sessions(self):
        """
        保存网络
        """
        os.makedirs(self.log_dir + 'ckpts/', exist_ok=True)
        for i, sess in enumerate(self._sessions):
            self.saver.save(sess, self.log_dir + 'ckpts/' + str(i) + '.ckpt')


class DQN(Agent):
    """
    DQN总的类
    """

    def __init__(self, observation_shape, obervation_range, observations_in_state, action_n, log_dir,
                 features: FeaturesNet, Q_layers: QLayerNet, memory: Memory, target: Target,
                 LEARNING_RATE=2E-3, EPS_INITIAL=0.5, EPS_END=0.1, EPS_STEP=1E-5):
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
        :param EPS_STEP: epsilon衰减率
        :param EPS_END: epsilon终值
        """

        super().__init__(observation_shape, obervation_range, observations_in_state, action_n, log_dir)
        self.LEARNING_RATE = LEARNING_RATE
        self.eps = EPS_INITIAL
        self.EPS_STEP = EPS_STEP
        self.EPS_END = EPS_END

        self.__features = features
        self.__Q_layers = Q_layers
        self.__memory = memory
        self.__target = target

        # 构建网络
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__layers, self.__layers_n = self.__features.create(self.state_shape)  # 初始化网络feature部分
            _ = self.__Q_layers.create_Q_layers(self.action_n, self.__layers[-1])  # 初始化网络Q值部分
            self.__layers += _[0]
            self.__layers_n += _[1]

            self._loss = self.__memory.create(self.__layers[0], self.__layers[-1], self.LEARNING_RATE)  # 初始化误差
            self._sessions = self.__target.create_target(self.__layers[0], self.__layers[-1])  # 初始化目标Q值计算

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()

        # 读取已经保存的网络
        if os.path.exists(self.log_dir + 'ckpts/'):
            self.load_sessions()
        else:
            for sess in self._sessions: sess.run(init)

    def __greedy(self):
        state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
        ret = reduce(lambda x, y: x + y,
                     [s.run(self.__layers[-1], feed_dict={self.__layers[0]: [state]})[0] for s in
                      self._sessions])
        return np.argmax(ret)

    def exploit(self, env, render=False):
        ret = 0
        observation = env.reset()
        self._observation_buff = []

        while True:
            if render: env.render()
            self._observation_buff.append(self.normalize_observation(observation))

            # 选择动作
            if len(self._observation_buff) >= self.observations_in_state:
                action = self.__greedy()
            else:
                action = np.random.choice(self.action_n)

            observation, reward, done, _ = env.step(action)
            ret += reward
            if done:
                return ret

    def explore(self, env, render=False):
        ret = 0
        observation = env.reset()
        self._observation_buff = []

        while True:
            if render: env.render()
            self._observation_buff.append(self.normalize_observation(observation))

            # 选择动作
            if len(self._observation_buff) >= self.observations_in_state and np.random.rand() > self.eps:
                action = self.__greedy()
            else:
                action = np.random.choice(self.action_n)
            self.eps = max(self.eps - self.EPS_STEP, self.EPS_END)

            # 实行动作
            nxt_observation, reward, done, _ = env.step(action)

            # 接受经验
            if len(self._observation_buff) > self.observations_in_state:
                state = np.array(self._observation_buff[:self.observations_in_state]).reshape(self.state_shape)
                nxt_state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
                self.__memory.perceive(state, self.__prev_action, reward, nxt_state, done)  # 上一次动作作为记忆
                self.__memory.replay(self.__target.compute_y)  # target产生y训练网络
                del self._observation_buff[0]

            # 保存这个动作，下一次存储记忆用
            onehot = np.zeros((self.action_n,))  # 将动作单个数值转化成onehot向量
            onehot[action] = 1
            self.__prev_action = onehot

            observation = nxt_observation
            ret += reward
            if done:
                return ret

    def save_hyperparameters(self):
        """
        保存参数到parameters.json
        """
        params = dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))
        for i in [self.__features, self.__Q_layers, self.__memory, self.__target]:
            params[i.__class__.__name__] = i.save_hyperparameters()
        with open(self.log_dir + 'parameters.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)


class BootstrappedDQN(Agent):
    """
    Bootstrapped DQN
    """

    def __init__(self, observation_shape, obervation_range, observations_in_state, action_n, log_dir,
                 features: FeaturesNet, Q_layers: QLayerNet,
                 LEARNING_RATE=1E-3, N_HEADS=5, GAMMA=1, MEMORY_SIZE=5000, BATCH_SIZE=50, TRAIN_REPEAT=2):
        """
        :param observation_shape: observation数组尺寸
        :param obervation_range: obeservation数组范围
        :param observations_in_state: 多少个observation组成一个state
        :param action_n: 动作个数
        :param log_dir: log文件路径
        :param features: 网络feature部分
        :param features: 网络feature部分
        :param Q_layers: 网络Q值部分
        :param LEARNING_RATE: 学习速率
        :param N_HEADS: heads数
        :param MEMORY_SIZE: 记忆总量大小
        :param BATCH_SIZE: 每次回访的个数
        :param TRAIN_REPEAT: 每次replay重复的batch
        """
        super().__init__(observation_shape, obervation_range, observations_in_state, action_n, log_dir)

        self.LEARNING_RATE = LEARNING_RATE
        self.N_HEADS = N_HEADS
        self.GAMMA = GAMMA
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAIN_REPEAT = TRAIN_REPEAT

        self.__memory = deque(maxlen=self.MEMORY_SIZE)
        self.__features = features
        self.__Q_layers = Q_layers

        self.__head_returns = np.zeros((N_HEADS, 2))  # heads回报。第0列次数，第1列总回报

        # 构建网络
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__layers, self.__layers_n = self.__features.create(self.state_shape)  # 初始化网络feature部分

            qs, qs_n = [], []
            self._loss = []
            self.__train_step = []
            with tf.name_scope('train'):
                self.__y = tf.placeholder(tf.float32, [None], name='target')
                self.__action_onehot = tf.placeholder(tf.float32, [None, self.action_n], name='action_onehot')
            for i in range(self.N_HEADS):
                _ = self.__Q_layers.create_Q_layers(self.action_n, self.__layers[-1])  # 初始化网络Q值部分
                qs.append(_[0])
                qs_n.append(_[1])
                with tf.name_scope('train_{}'.format(i)):
                    action_value = tf.reduce_sum(qs[-1][-1] * self.__action_onehot, reduction_indices=1,
                                                 name='sample_Q')
                    self._loss.append(tf.reduce_mean(tf.square(self.__y - action_value), name='loss'))
                    self.__train_step.append(tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self._loss[-1]))
            self.__layers.append(qs)
            self.__layers_n.append(qs_n)

            self._sessions = [tf.Session(), tf.Session()]

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()

        # 读取已经保存的网络
        if os.path.exists(self.log_dir + 'ckpts/'):
            self.load_sessions()
        else:
            for sess in self._sessions: sess.run(init)

    def exploit(self, env, render=False):
        ret = 0
        observation = env.reset()
        self._observation_buff = []
        best_heads = np.argmax(np.nan_to_num(self.__head_returns[:, 1] / self.__head_returns[:, 0]))  # 选平均回报最大的head

        while True:
            if render: env.render()
            self._observation_buff.append(self.normalize_observation(observation))

            # 选择动作
            if len(self._observation_buff) >= self.observations_in_state:
                state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
                qs = reduce(lambda x, y: x + y,
                            [s.run(self.__layers[-1][best_heads][-1], feed_dict={self.__layers[0]: [state]})[0] for s in
                             self._sessions])
                action = np.argmax(qs)
            else:
                action = np.random.choice(self.action_n)  # 现有observation不够，随机选择动作

            observation, reward, done, _ = env.step(action)
            ret += reward
            if done:
                return ret

    def explore(self, env, render=False):
        ret = 0
        observation = env.reset()
        self._observation_buff = []
        explore_head = np.random.randint(self.N_HEADS)

        while True:
            if render: env.render()
            self._observation_buff.append(self.normalize_observation(observation))

            # 选择动作
            if len(self._observation_buff) >= self.observations_in_state:
                state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
                action = np.argmax(reduce(lambda x, y: x + y,
                                          [s.run(self.__layers[-1][explore_head][-1],
                                                 feed_dict={self.__layers[0]: [state]})[0] for s in
                                           self._sessions]))
            else:
                action = np.random.choice(self.action_n)

            # 实行动作
            nxt_observation, reward, done, _ = env.step(action)

            # 接受经验
            if len(self._observation_buff) > self.observations_in_state:
                state = np.array(self._observation_buff[:self.observations_in_state]).reshape(self.state_shape)
                nxt_state = np.array(self._observation_buff[-self.observations_in_state:]).reshape(self.state_shape)
                self.__memory.append((state, self.__prev_action, reward, nxt_state, done, self.bootstrap_mask()))
                self.replay()
                del self._observation_buff[0]

            # 保存这个动作，下一次存储记忆用
            onehot = np.zeros((self.action_n,))  # 将动作单个数值转化成onehot向量
            onehot[action] = 1
            self.__prev_action = onehot

            observation = nxt_observation
            ret += reward
            if done:
                # 更新heads平均回报
                self.__head_returns[explore_head][1] += ret
                self.__head_returns[explore_head][0] += 1
                return ret

    def bootstrap_mask(self):
        return np.random.choice([0, 1], self.N_HEADS)

    def replay(self):
        for _ in range(self.TRAIN_REPEAT):
            batch_ind = np.random.choice(len(self.__memory), self.BATCH_SIZE)
            batch = list(map(lambda i: self.__memory[i], batch_ind))
            batch = [np.array([m[i] for m in batch]) for i in range(6)]

            # 任取一个训练
            sess1, sess2 = np.random.permutation(self._sessions)

            heads = defaultdict(lambda: [])
            for x, y in np.transpose(np.nonzero(batch[5])): heads[y].append(x)
            for x, y in heads.items():
                state_batch, action_batch, y_batch, nxt_state_batch, done_batch = [
                    batch[i][y] for i in range(5)]

                # sess1计算argmaxQ的onehot表示
                a = np.eye(self.action_n)[
                    np.argmax(sess1.run(self.__layers[-1][x][-1], feed_dict={self.__layers[0]: nxt_state_batch}),
                              axis=1)]

                # sess2计算Q
                nxt_qs = sess2.run(self.__layers[-1][x][-1], feed_dict={self.__layers[0]: nxt_state_batch})
                nxt_qs = np.sum(nxt_qs * a, axis=1)
                nxt_qs[done_batch] = 0  # 如果完成设为0
                y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

                sess1.run(self.__train_step[x], feed_dict={self.__layers[0]: state_batch,
                                                           self.__action_onehot: action_batch,
                                                           self.__y: y_batch})

    def save_hyperparameters(self):
        """
        保存参数到parameters.json
        """
        params = dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))
        for i in [self.__features, self.__Q_layers]:
            params[i.__class__.__name__] = i.save_hyperparameters()
        with open(self.log_dir + 'parameters.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)
