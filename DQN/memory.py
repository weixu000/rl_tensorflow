import heapq
import numpy as np
from collections import deque


class Memory:
    """
    记忆基类
    """

    def perceive(self, state, action, reward, nxt_state, done):
        raise NotImplementedError()

    def replay(self):
        """
        产生batch
        :return: 分别对应状态、动作、Q、下一状态、完成与否的矩阵（一行对应一个记忆）
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

    def perceive(self, state, action, reward, nxt_state, done):
        # 将处理后的经验加入记忆中，超出存储的自动剔除
        self.__memory.append((state, action, reward, nxt_state, done))

    def replay(self):
        batch_ind = np.random.choice(len(self.__memory), min(self.BATCH_SIZE, len(self.__memory)), False)
        batch = [x for i, x in enumerate(self.__memory) if i in batch_ind]
        return [np.array([m[i] for m in batch]) for i in range(5)]


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

    def __init__(self, MEMORY_SIZE=10000, BATCH_SIZE=50, ALPHA=3, SORT_WHEN=5000):
        """
        :param MEMORY_SIZE: 记忆总量大小
        :param BATCH_SIZE: 每次回访的个数
        :param ALPHA: 幂分布的指数
        :param SORT_WHEN: 何时完全排序记忆
        """
        self.MEMORY_SIZE = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.ALPHA = ALPHA
        self.SORT_WHEN = SORT_WHEN

        self.__memory = []
        # 最近的记忆，尚未训练
        self.__recent_memory = []

    def perceive(self, state, action, reward, nxt_state, done):
        # 插入最近的记忆
        experience = [state, action, reward, nxt_state, done]
        self.__recent_memory.append(self.Experience(None, experience))

        # 记忆过多，则删除误差最小的
        if len(self.__memory) >= self.MEMORY_SIZE: heapq.heappop(self.__memory)

        # 记忆较多时，进行排序
        # if not self.n_timesteps % self.SORT_WHEN: self.memory.sort()

    def replay(self):
        # 按幂分布取出记忆
        self.__set_default_errors()
        batch_ind = np.random.power(self.ALPHA, min(self.BATCH_SIZE, len(self.__memory))) * len(self.__memory)
        batch_ind = list(set(batch_ind.astype(int)))

        batch = [x.data for i, x in enumerate(self.__memory) if i in batch_ind]
        batch = [np.array([m[i] for m in batch]) for i in range(5)]
        return batch
        self.__update_errors(batch)

    def __set_default_errors(self):
        errors = self.__compute_errors(self.__recent_memory)

        for i, m in enumerate(self.__recent_memory): m.priority = errors[i]
        self.__memory += self.__recent_memory
        self.__recent_memory = []
        heapq.heapify(self.__memory)

    def __update_errors(self, batch):
        errors = self.__compute_errors(batch)
        for i, m in enumerate(batch): m.priority = errors[i]
        heapq.heapify(self.__memory)

    def __compute_errors(self, batch):
        state_batch, action_batch, y_batch, nxt_state_batch, done_batch = [
            np.array([m[i] for m in [x.data for i, x in enumerate(batch)]]) for i in range(5)]

        errors = np.zeros_like(y_batch)
        for sess, _ in self.sessions:
            errors += sess.run(self.loss_vec, feed_dict={self.layers[0]: state_batch,
                                                         self.action_onehot: action_batch,
                                                         self.y: y_batch})
        return errors / len(self.__recent_memory)
