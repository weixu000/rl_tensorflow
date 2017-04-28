import heapq
import numpy as np
from collections import deque
import tensorflow as tf


class Memory:
    """
    记忆基类
    同时也负责计算误差，训练网络
    """

    def create(self, input_layer, Q_layer, learning_rate):
        """
        生成误差，建立优化器（用到learning_rate）
        :param input_layer: 输入层
        :param Q_layer: Q值层
        :param learning_rate:学习速率
        :return 误差
        """
        raise NotImplementedError()

    def perceive(self, state, action, reward, nxt_state, done):
        """
        接受经验
        :param state: 状态
        :param action: 动作onehot表示
        :param reward: 
        :param nxt_state: 动作导致的下一个状态
        :param done: 是否终态
        """
        raise NotImplementedError()

    def replay(self, compute_y):
        """
        产生batch，调用compute_y计算出y训练网络
        :param compute_y: 输入batch，返回需要更新的session和
        """
        raise NotImplementedError()

    def save_hyperparameters(self):
        return dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))


class RandomReplay(Memory):
    """
    随机经验回放
    """

    def __init__(self, MEMORY_SIZE=5000, BATCH_SIZE=50, TRAIN_REPEAT=2):
        """
        :param MEMORY_SIZE: 记忆总量大小
        :param BATCH_SIZE: 每次回访的个数
        :param TRAIN_REPEAT: 每次replay重复的batch
        """
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAIN_REPEAT = TRAIN_REPEAT
        self.__memory = deque(maxlen=self.MEMORY_SIZE)

    def create(self, input_layer, Q_layer, learning_rate):
        self._input_layer = input_layer
        self._Q_layer = Q_layer

        # 训练用到的指示值y（相当于图像识别的标签），jumpAction（最小化特定动作Q的误差），loss
        with tf.name_scope('train'):
            self.__y = tf.placeholder(tf.float32, [None], name='target')
            self.__action_onehot = tf.placeholder(tf.float32, [None, Q_layer.shape[1].value], name='action_onehot')
            action_value = tf.reduce_sum(Q_layer * self.__action_onehot, reduction_indices=1,
                                         name='sample_Q')
            loss = tf.reduce_mean(tf.square(self.__y - action_value), name='loss')
            self.__train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return loss

    def __train_sess(self, sess, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        sess.run(self.__train_step, feed_dict={self._input_layer: state_batch,
                                               self.__action_onehot: action_batch,
                                               self.__y: y_batch})

    def perceive(self, state, action, reward, nxt_state, done):
        self.__memory.append((state, action, reward, nxt_state, done))

    def replay(self, compute_y):
        for _ in range(self.TRAIN_REPEAT):
            batch_ind = np.random.choice(len(self.__memory), self.BATCH_SIZE)
            batch = list(map(lambda i: self.__memory[i], batch_ind))
            sess, batch = compute_y([np.array([m[i] for m in batch]) for i in range(5)])
            self.__train_sess(sess, batch)


class PrioritizedReplay(Memory):
    """
    根据优先级抽取记忆回放
    """

    class Experience:
        """
        存储带优先级的经验
        """

        def __lt__(self, other): return self.priority > other.priority

        def __init__(self, priority, data):
            self.priority = priority
            self.data = data

    def __init__(self, MEMORY_SIZE=5000, BATCH_SIZE=50, TRAIN_REPEAT=2, ALPHA=3, BETA_INITIAL=0.5, BETA_STEP=1E-3,
                 SORT_WHEN=50):
        """
        :param MEMORY_SIZE: 记忆总量大小
        :param BATCH_SIZE: 每次回访的个数
        :param TRAIN_REPEAT: 每次replay重复的batch
        :param ALPHA: 幂分布的指数
        :param SORT_WHEN: 何时完全排序记忆
        """
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.ALPHA = ALPHA
        self.BETA = BETA_INITIAL
        self.BETA_STEP = BETA_STEP
        self.TRAIN_REPEAT = TRAIN_REPEAT
        self.SORT_WHEN = SORT_WHEN

        self._memory = []

    def create(self, input_layer, Q_layer, learning_rate):
        self._input_layer = input_layer
        self._Q_layer = Q_layer

        with tf.name_scope('train'):
            self._y = tf.placeholder(tf.float32, [None], name='target')
            self._action_onehot = tf.placeholder(tf.float32, [None, Q_layer.shape[1].value], name='action_onehot')
            action_value = tf.reduce_sum(Q_layer * self._action_onehot, reduction_indices=1, name='sample_Q')
            self._sample_weights = tf.placeholder(tf.float32, [None], name='IS_weights')
            self._loss_vec = self._y - action_value
            self._loss = tf.reduce_sum(tf.square(self._loss_vec) * self._sample_weights, name='loss')
            self._train_step = tf.train.AdamOptimizer(learning_rate).minimize(self._loss)

        return self._loss

    def _train_sess(self, sess, batch, sample_weights):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        sess.run(self._train_step, feed_dict={self._input_layer: state_batch,
                                              self._action_onehot: action_batch,
                                              self._sample_weights: sample_weights,
                                              self._y: y_batch})

    def _shorten_memory(self):
        # ind = np.arange(len(self._memory))
        # probs = (ind / len(self._memory)) ** (self.ALPHA - 1)
        # probs /= probs.sum()
        ind = np.random.choice(len(self._memory), len(self._memory) // 2)
        self._memory = [m for i, m in enumerate(self._memory) if i not in ind]
        heapq.heapify(self._memory)

    def _compute_sample_weights(self, batch_ind):
        """
        计算IS weights，抵消回访的不随机性
        :param batch_ind: batch的编号
        :return: IS weights
        """
        sample_weights = self.ALPHA * (1 - batch_ind / len(self._memory)) ** (self.ALPHA - 1)
        sample_weights = (sample_weights * len(batch_ind)) ** (-self.BETA)
        self.BETA = min(self.BETA + self.BETA_STEP, 1)
        return sample_weights


class TDErrorPrioritizedReplay(PrioritizedReplay):
    """
    根据TDError优先级抽取记忆回放
    """

    def __init__(self, MEMORY_SIZE=5000, BATCH_SIZE=50, TRAIN_REPEAT=2, ALPHA=3, BETA_INITIAL=0.5, BETA_STEP=1E-3,
                 SORT_WHEN=100):
        super().__init__(MEMORY_SIZE, BATCH_SIZE, TRAIN_REPEAT, ALPHA, BETA_INITIAL, BETA_STEP, SORT_WHEN)
        self.__n_perceived = 0

    def perceive(self, state, action, reward, nxt_state, done):
        self.__n_perceived += 1
        heapq.heappush(self._memory, self.Experience(self._memory[0].priority + 1 if len(self._memory) else 0,
                                                     [state, action, reward, nxt_state, done]))

        if len(self._memory) >= self.MEMORY_SIZE: self._shorten_memory()
        if not self.__n_perceived % self.SORT_WHEN: self._memory.sort()

    def replay(self, compute_y):
        for _ in range(self.TRAIN_REPEAT):
            # 按幂分布取出记忆
            batch_ind = len(self._memory) * (1 - np.random.power(self.ALPHA, self.BATCH_SIZE))
            batch_ind = batch_ind.astype(int)

            # 计算y值
            batch = list(map(lambda i: self._memory[i].data, batch_ind))
            sess, batch = compute_y([np.array([m[i] for m in batch]) for i in range(5)])

            # 训练网络，更新经验优先级
            sample_weights = self._compute_sample_weights(batch_ind)
            self._train_sess(sess, batch, sample_weights)
            self.__update_errors(sess, batch, batch_ind)

    def __update_errors(self, sess, batch, batch_ind):
        """
        更新经验优先级
        :param sess: 用来计算TD误差的session
        :param batch: 用来计算TD误差的batch
        :param batch_ind: batch对应self.__memory的编号
        """
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch
        errors = np.abs(sess.run(self._loss_vec, feed_dict={self._input_layer: state_batch,
                                                            self._action_onehot: action_batch,
                                                            self._y: y_batch}))
        for i, e in zip(batch_ind, errors): self._memory[i].priority = e
        heapq.heapify(self._memory)


class ReturnPrioritizedReplay(PrioritizedReplay):
    """
    根据Return优先级抽取记忆回放
    """

    def __init__(self, MEMORY_SIZE=5000, BATCH_SIZE=50, TRAIN_REPEAT=2, ALPHA=3, BETA_INITIAL=0.5, BETA_STEP=1E-3,
                 SORT_WHEN=50):
        super().__init__(MEMORY_SIZE, BATCH_SIZE, TRAIN_REPEAT, ALPHA, BETA_INITIAL, BETA_STEP, SORT_WHEN)
        self._tmp_memory = []
        self.__n_perceived = 0

    def perceive(self, state, action, reward, nxt_state, done):
        self.__n_perceived += 1
        self._tmp_memory.append((state, action, reward, nxt_state, done))
        if done:
            ret = 0
            for i, m in enumerate(self._tmp_memory): ret += m[2]
            self._memory += [self.Experience(ret, m) for m in self._tmp_memory]
            heapq.heapify(self._memory)
            self._tmp_memory.clear()
            if len(self._memory) >= self.MEMORY_SIZE: self._shorten_memory()
            if not self.__n_perceived % self.SORT_WHEN: self._memory.sort()

    def replay(self, compute_y):
        if not self._memory: return
        for _ in range(self.TRAIN_REPEAT):
            # 按幂分布取出记忆
            batch_ind = len(self._memory) * (1 - np.random.power(self.ALPHA, self.BATCH_SIZE))
            batch_ind = batch_ind.astype(int)

            # 计算y值
            batch = list(map(lambda i: self._memory[i].data, batch_ind))
            # batch = [x.data for i, x in enumerate(self._memory) if i in batch_ind]
            sess, batch = compute_y([np.array([m[i] for m in batch]) for i in range(5)])

            # 训练网络，更新经验优先级
            sample_weights = self._compute_sample_weights(batch_ind)
            self._train_sess(sess, batch, sample_weights)
