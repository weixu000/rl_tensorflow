import tensorflow as tf
import numpy as np


class Target:
    """
    产生目标Q值基类
    """

    def __init__(self, GAMMA=1):
        self.GAMMA = GAMMA

    def create_target(self, init, input_layer, Q_layer):
        """
        初始化Target
        :param init: tf初始化所有variable
        :param input_layer: 输入层
        :param Q_layer: Q输出层
        """
        self._input_layer = input_layer
        self._Q_layer = Q_layer
        self._create_sessions(init)

    def _create_sessions(self, init):
        """
        建立session
        :param init: tf初始化所有variable
        """
        raise NotImplementedError()

    def compute_y(self, batch):
        """
        计算batch对应的值
        """
        raise NotImplementedError()

    def __getitem__(self, state):
        raise NotImplementedError

    def save_hyperparameters(self):
        return dict(filter(lambda x: x[0][0].isupper(), self.__dict__.items()))


class OriginalDQN(Target):
    """
    Nature DQN
    """

    def __getitem__(self, state):
        return self._session.run(self._Q_layer, feed_dict={self._input_layer: [state]})[0]

    def _create_sessions(self, init):
        self._session = tf.Session()
        self._session.run(init)

    def compute_y(self, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        # 计算公式中的maxQ，如果完成设为0
        nxt_qs = np.max(self._session.run(self._Q_layer, feed_dict={self._input_layer: nxt_state_batch}), axis=1)
        nxt_qs[done_batch] = 0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        return self._session, batch


class DoubleDQN(Target):
    """
    Double DQN
    """

    def __getitem__(self, state):
        ret = np.zeros(self._Q_layer.shape[1])
        for s in self._sessions:
            ret += s.run(self._Q_layer, feed_dict={self._input_layer: [state]})[0]
        return ret / len(self._sessions)

    def _create_sessions(self, init):
        self._sessions = [tf.Session(), tf.Session()]
        for s in self._sessions: s.run(init)

    def compute_y(self, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        # 任取一个训练
        if np.random.rand() >= 0.5:
            sess1, sess2 = self._sessions
        else:
            sess2, sess1 = self._sessions

        # sess1计算argmaxQ的onehot表示
        a = np.eye(self._Q_layer.shape[1])[
            np.argmax(sess1.run(self._Q_layer, feed_dict={self._input_layer: nxt_state_batch}), axis=1)]
        # sess2计算Q
        # nxt_qs = sess2.run(self._action_value, feed_dict={self._input_layer: nxt_state_batch, self._action_onehot: a})
        nxt_qs = sess2.run(self._Q_layer, feed_dict={self._input_layer: nxt_state_batch})
        nxt_qs = np.sum(nxt_qs * a, axis=1)
        nxt_qs[done_batch] = 0  # 如果完成设为0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        return sess1, batch
