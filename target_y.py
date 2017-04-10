import tensorflow as tf
import numpy as np
from memory import RandomReplay


class OriginalDQN(RandomReplay):
    """
    Nature DQN
    """
    NAME = 'Original'

    def __init__(self, env, env_name):
        super().__init__(env, env_name)

        # 目前只用一个网络
        self.sessions = [(tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir, self.graph))]
        self.sessions[0][0].run(self.init)

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


class DoubleDQN(RandomReplay):
    """
    Double DQN
    """
    NAME = 'Double'

    def __init__(self, env, env_name):
        super().__init__(env, env_name)

        self.sessions = [(tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir + 'Q1/', self.graph)),
                         (tf.Session(graph=self.graph), tf.summary.FileWriter(self.log_dir + 'Q2/', self.graph))]
        for s in self.sessions: s[0].run(self.init)

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
