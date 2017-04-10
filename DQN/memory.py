import numpy as np
import tensorflow as tf
import heapq
from collections import deque
from DQN.network import *


class RandomReplay(OriginalQLayer):
    """
    随机经验回放
    """

    def __init__(self, env, env_name):
        self.MEMORY_SIZE = 10000
        self.BATCH_SIZE = 50
        super().__init__(env, env_name)
        self.memory = deque(maxlen=self.MEMORY_SIZE)

    def perceive(self, state, action, reward, nxt_state, done):
        # 将处理后的经验加入记忆中，超出存储的自动剔除
        super().perceive(state, action, reward, nxt_state, done)
        self.memory.append(self.process_experience(state, action, reward, nxt_state, done))
        self.replay()

    def replay(self):
        """
        随机抽取batch_size个记忆，分别建立状态、动作、Q、下一状态、完成与否的矩阵（一行对应一个记忆）
        """
        batch_ind = np.random.choice(len(self.memory), min(self.BATCH_SIZE, len(self.memory)), False)
        batch = [x for i, x in enumerate(self.memory) if i in batch_ind]
        self.train([np.array([m[i] for m in batch]) for i in range(5)])


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
        self.MEMORY_SIZE = 10000
        self.BATCH_SIZE = 50
        self.ALPHA = 3  # 幂分布的指数
        self.SORT_WHEN = 5000  # 何时完全排序记忆
        super().__init__(env, env_name)

        self.memory = []
        # 最近的记忆，尚未训练
        self.recent_memory = []

    def perceive(self, state, action, reward, nxt_state, done):
        # 插入最近的记忆
        super().perceive(state, action, reward, nxt_state, done)
        experience = list(self.process_experience(state, action, reward, nxt_state, done))
        self.recent_memory.append(self.Experience(None, experience))

        # 记忆过多，则删除误差最小的
        if len(self.memory) >= self.MEMORY_SIZE: heapq.heappop(self.memory)

        # 记忆较多时，进行排序
        if not self.n_timesteps % self.SORT_WHEN: self.memory.sort()

        self.replay()

    def replay(self):
        # 按幂分布取出记忆
        self.set_default_errors()
        batch_ind = np.random.power(self.ALPHA, min(self.BATCH_SIZE, len(self.memory))) * len(self.memory)
        batch_ind = list(set(batch_ind.astype(int)))

        batch = [x.data for i, x in enumerate(self.memory) if i in batch_ind]
        batch = [np.array([m[i] for m in batch]) for i in range(5)]
        self.train(batch)
        self.update_errors()

    def set_default_errors(self):
        errors = self.compute_errors(self.recent_memory)

        for i, m in enumerate(self.recent_memory): m.priority = errors[i]
        self.memory += self.recent_memory
        self.recent_memory = []
        heapq.heapify(self.memory)

    def update_errors(self, batch):
        errors = self.compute_errors(batch)
        for i, m in enumerate(batch): m.priority = errors[i]
        heapq.heapify(self.memory)

    def compute_errors(self, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = [
            np.array([m[i] for m in [x.data for i, x in enumerate(batch)]]) for i in range(5)]

        errors = np.zeros_like(y_batch)
        for sess, _ in self.sessions:
            errors += sess.run(self.loss_vec, feed_dict={self.layers[0]: state_batch,
                                                         self.action_onehot: action_batch,
                                                         self.y: y_batch})
        return errors / len(self.recent_memory)
