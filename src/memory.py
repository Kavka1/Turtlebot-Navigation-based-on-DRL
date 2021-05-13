import numpy as np

class Memory(object):
    def __init__(self, memory_size, s_dim, a_dim):
        super(Memory, self).__init__()
        self.s1_buffer = np.zeros([memory_size, s_dim], dtype=np.float32)
        self.s2_buffer = np.zeros([memory_size, s_dim], dtype=np.float32)
        self.a_buffer = np.zeros([memory_size, a_dim], dtype=np.float32)
        self.r_buffer = np.zeros(memory_size, dtype=np.float32)
        self.done_buffer = np.zeros(memory_size, dtype=np.float32)
        self.write , self.size, self.memory_size = 0, 0, memory_size

    def __len__(self):
        return self.size

    def store(self, s, a, r, s_, done):
        self.s1_buffer[self.write] = s
        self.a_buffer[self.write] = a
        self.r_buffer[self.write] = r
        self.s2_buffer[self.write] = s_
        self.done_buffer[self.write] = done
        self.write = (self.write + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample_batch(self, batch_size):
        if batch_size > self.size:
            raise ValueError('No enough samples in memory for batch-learning')
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.s1_buffer[idxs], self.a_buffer[idxs], self.r_buffer[idxs], self.s2_buffer[idxs], self.done_buffer[idxs]


class PER_Memory():
    def __init__(self, memory_size, s_dim, a_dim):
        super(PER_Memory, self).__init__()
        self.s1_buffer = np.zeros([memory_size, s_dim], dtype=np.float32)
        self.s2_buffer = np.zeros([memory_size, s_dim], dtype=np.float32)
        self.a_buffer = np.zeros([memory_size, a_dim], dtype=np.float32)
        self.r_buffer = np.zeros(memory_size, dtype=np.float32)
        self.done_buffer = np.zeros(memory_size, dtype=np.float32)
        self.priority_buffer = np.zeros(memory_size, dtype=np.float32)
        self.write , self.size, self.memory_size = 0, 0, memory_size

    def __len__(self):
        return self.size

    def store(self, s, a, r, s_, done, priority):
        self.s1_buffer[self.write] = s
        self.a_buffer[self.write] = a
        self.r_buffer[self.write] = r
        self.s2_buffer[self.write] = s_
        self.done_buffer[self.write] = done
        self.priority_buffer[self.write] = priority
        self.write = (self.write + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample_batch(self, batch_size):
        if batch_size > self.size:
            raise ValueError('No enough samples in memory for batch-learning')
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.s1_buffer[idxs], self.a_buffer[idxs], self.r_buffer[idxs], self.s2_buffer[idxs], self.done_buffer[idxs], self.priority_buffer[idxs]