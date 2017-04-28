import tensorflow as tf
import numpy as np


class Target:
    """
    产生目标Q值基类
    """

    def create_target(self, input_layer, Q_layer):
        """
        初始化Target
        :param input_layer: 输入层
        :param Q_layer: Q输出层
        :return: session
        """
        self._input_layer = input_layer
        self._Q_layer = Q_layer
        self._sessions = [tf.Session(), tf.Session()]

        return self._sessions

    def compute_y(self, batch):
        """
        计算batch对应的值
        """
        raise NotImplementedError()

    def save_hyperparameters(self):
        return dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))


class SingleDQN(Target):
    """
    Nature DQN
    """

    def __init__(self, GAMMA=1, TARGET_HOLD=200):
        self.GAMMA = GAMMA
        self.TARGET_HOLD = TARGET_HOLD
        self._n_compute = 0

    def compute_y(self, batch):
        self._n_compute += 1
        if self._n_compute % self.TARGET_HOLD == 0: self.__copy_vars()

        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch
        # 计算公式中的maxQ，如果完成设为0
        nxt_qs = np.max(self._sessions[1].run(self._Q_layer, feed_dict={self._input_layer: nxt_state_batch}), axis=1)
        nxt_qs[done_batch] = 0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        return self._sessions[0], batch

    def __copy_vars(self):
        """
        更新目标网络
        """
        with self._sessions[0].graph.as_default():
            for v in tf.trainable_variables():
                self._sessions[1].run(v.assign(self._sessions[0].run(v)))


class DoubleDQN(Target):
    """
    Double DQN
    """

    def __init__(self, GAMMA=1):
        self.GAMMA = GAMMA

    def compute_y(self, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        # 任取一个训练
        sess1, sess2 = np.random.permutation(self._sessions)

        # sess1计算argmaxQ的onehot表示
        a = np.eye(self._Q_layer.shape[1])[
            np.argmax(sess1.run(self._Q_layer, feed_dict={self._input_layer: nxt_state_batch}), axis=1)]
        # sess2计算Q
        nxt_qs = sess2.run(self._Q_layer, feed_dict={self._input_layer: nxt_state_batch})
        nxt_qs = np.sum(nxt_qs * a, axis=1)
        nxt_qs[done_batch] = 0  # 如果完成设为0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        return sess1, batch
