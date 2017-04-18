import tensorflow as tf
import numpy as np


class Target:
    """
    产生目标Q值基类
    """

    def __init__(self, GAMMA=1):
        self.GAMMA = GAMMA

    def create_target(self, init, apply_grad, input_layer, Q_layer, action_onehot, y):
        """
        初始化Target
        :param init: tf初始化所有variable
        :param apply_grad: 更新梯度
        :param input_layer: 输入层
        :param Q_layer: Q输出层
        :param actioin_value: 动作Q值
        :param y: 目标Q值
        :return: 
        """
        self._apply_grad = apply_grad
        self._input_layer = input_layer
        self._Q_layer = Q_layer
        self._action_onehot = action_onehot
        self._y = y
        self._create_sessions(init)

    def _create_sessions(self, init):
        """
        建立session
        :param init: tf初始化所有variable
        :return: 
        """
        raise NotImplementedError()

    def _train_sess(self, sess, batch):
        """
        输入数据，训练网络
        """
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        sess.run(self._apply_grad, feed_dict={self._input_layer: state_batch,
                                              self._action_onehot: action_batch,
                                              self._y: y_batch})

    def train(self, batch):
        """
        用batch训练网络
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

    def train(self, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = batch

        # 计算公式中的maxQ，如果完成设为0
        nxt_qs = np.max(self._session.run(self._Q_layer, feed_dict={self._input_layer: nxt_state_batch}), axis=1)
        nxt_qs[done_batch] = 0
        y_batch += self.GAMMA * nxt_qs  # 计算公式，y在抽取时已经保存了reward

        self._train_sess(self._session, batch)


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

    def train(self, batch):
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

        self._train_sess(sess1, batch)
