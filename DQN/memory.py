import heapq
import numpy as np
from collections import deque
import tensorflow as tf


class Memory:
    """
    记忆基类
    同时也负责计算误差，训练网络
    """

    def create_loss(self, input_layer, Q_layer, learning_rate):
        """
        生成误差，建立优化器（用到learning_rate）
        :param input_layer: 输入层
        :param Q_layer: Q值层
        :param learning_rate:学习速率
        """
        raise NotImplementedError()

    def perceive(self, state, action, reward, nxt_state, done):
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

    def __init__(self, MEMORY_SIZE=10000, BATCH_SIZE=50):
        """
        :param MEMORY_SIZE: 记忆总量大小
        :param BATCH_SIZE: 每次回访的个数
        """
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.__memory = deque(maxlen=self.MEMORY_SIZE)

    def create_loss(self, input_layer, Q_layer, learning_rate):
        self._input_layer = input_layer
        self._Q_layer = Q_layer

        # 训练用到的指示值y（相当于图像识别的标签），action（最小化特定动作Q的误差），loss
        with tf.name_scope('train'):
            self.__y = tf.placeholder(tf.float32, [None], name='target')
            self.__action_onehot = tf.placeholder(tf.float32, [None, Q_layer.shape[1].value], name='action_onehot')
            action_value = tf.reduce_sum(Q_layer * self.__action_onehot, reduction_indices=1,
                                         name='sample_Q')
            loss = tf.reduce_mean(tf.square(self.__y - action_value), name='loss')
            self.__train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def __train_sess(self, sess, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        sess.run(self.__train_step, feed_dict={self._input_layer: state_batch,
                                               self.__action_onehot: action_batch,
                                               self.__y: y_batch})

    def perceive(self, state, action, reward, nxt_state, done):
        self.__memory.append((state, action, reward, nxt_state, done))

    def replay(self, compute_y):
        batch_ind = np.random.choice(len(self.__memory), min(self.BATCH_SIZE, len(self.__memory)), False)
        batch = [x for i, x in enumerate(self.__memory) if i in batch_ind]
        sess, batch = compute_y([np.array([m[i] for m in batch]) for i in range(5)])
        self.__train_sess(sess, batch)


class RankBasedPrioritizedReplay(Memory):
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

    def __init__(self, MEMORY_SIZE=10000, BATCH_SIZE=50, ALPHA=3, BETA=1, SORT_WHEN=2000):
        """
        :param MEMORY_SIZE: 记忆总量大小
        :param BATCH_SIZE: 每次回访的个数
        :param ALPHA: 幂分布的指数
        :param SORT_WHEN: 何时完全排序记忆
        """
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.SORT_WHEN = SORT_WHEN

        self.__n_perceived = 0
        self.__memory = []
        self.__recent_memory = []  # 最近的记忆，尚未训练

    def create_loss(self, input_layer, Q_layer, learning_rate):
        self.__input_layer = input_layer
        self.__Q_layer = Q_layer

        # 训练用到的指示值y（相当于图像识别的标签），action（最小化特定动作Q的误差），loss
        with tf.name_scope('train'):
            self.__y = tf.placeholder(tf.float32, [None], name='target')
            self.__action_onehot = tf.placeholder(tf.float32, [None, Q_layer.shape[1].value], name='action_onehot')
            action_value = tf.reduce_sum(Q_layer * self.__action_onehot, reduction_indices=1,
                                         name='sample_Q')
            self.__sample_weights = tf.placeholder(tf.float32, [None], name='IS_weights')
            self.__loss_vec = self.__y - action_value
            self.__loss = tf.reduce_mean(tf.square(self.__loss_vec * self.__sample_weights), name='loss')
            self.__train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.__loss)

    def __train_sess(self, sess, batch, sample_weights):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        sess.run(self.__train_step, feed_dict={self.__input_layer: state_batch,
                                               self.__action_onehot: action_batch,
                                               self.__sample_weights: sample_weights,
                                               self.__y: y_batch})

    def perceive(self, state, action, reward, nxt_state, done):
        self.__n_perceived += 1
        # 插入最近的记忆
        experience = [state, action, reward, nxt_state, done]
        self.__recent_memory.append(self.Experience(None, experience))

        # 记忆过多，则删除误差最小的
        if len(self.__memory) >= self.MEMORY_SIZE: heapq.heappop(self.__memory)

        # 记忆较多时，进行排序
        if not self.__n_perceived % self.SORT_WHEN: self.__memory.sort()

    def replay(self, compute_y):
        # 按幂分布取出记忆
        batch_ind = np.random.power(self.ALPHA, self.BATCH_SIZE) * len(self.__memory)
        batch_ind = list(batch_ind.astype(int))

        # 强制回访最近的记忆
        batch_ind += list(range(len(self.__memory), len(self.__memory) + len(self.__recent_memory)))
        batch_ind = list(set(batch_ind))
        self.__memory += self.__recent_memory
        self.__recent_memory = []

        # 计算y值
        batch = [x.data for i, x in enumerate(self.__memory) if i in batch_ind]
        sess, batch = compute_y([np.array([m[i] for m in batch]) for i in range(5)])

        # 训练网络，更新经验优先级
        sample_weights = self.__compute_sample_weights(batch_ind)
        self.__train_sess(sess, batch, sample_weights)
        self.__update_errors(sess, batch, batch_ind)

    def __compute_sample_weights(self, batch_ind):
        """
        计算IS weights，抵消回访的不随机性
        :param batch_ind: batch的编号
        :return: IS weights
        """
        sample_weights = self.ALPHA * ((np.array(batch_ind) + 0.5) / len(self.__memory)) ** (self.ALPHA - 1)
        sample_weights = (sample_weights * len(batch_ind)) ** (-self.BETA)
        sample_weights /= sample_weights.max()
        return sample_weights

    def __update_errors(self, sess, batch, batch_ind):
        """
        更新经验优先级
        :param sess: 用来计算TD误差的session
        :param batch: 用来计算TD误差的batch
        :param batch_ind: batch对应self.__memory的编号
        """
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch
        errors = np.abs(sess.run(self.__loss_vec, feed_dict={self.__input_layer: state_batch,
                                                             self.__action_onehot: action_batch,
                                                             self.__y: y_batch}))
        for i, e in zip(batch_ind, errors): self.__memory[i].priority = e
        heapq.heapify(self.__memory)
