import tensorflow as tf
import numpy as np
from collections import deque
import heapq
from q_layers import DuelingDQN


class RandomReplay(DuelingDQN):
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


class RankBasedPrioritizedReplay(DuelingDQN):
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
