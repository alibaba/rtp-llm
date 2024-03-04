from collections import deque
import threading

class ThreadSafeDeque():
    def __init__(self):
        self.deque = deque()
        self.lock = threading.Lock()

    def append(self, item):
        with self.lock:
            return self.deque.append(item)

    def appendleft(self, item):
        with self.lock:
            return self.deque.appendleft(item)

    def pop(self):
        with self.lock:
            return self.deque.pop()

    def popleft(self):
        with self.lock:
            return self.deque.popleft()

    def remove(self, item):
        with self.lock:
            return self.deque.remove(item)

    def __len__(self):
        with self.lock:
            return len(self.deque)
    
    def copy(self):
        with self.lock:
            return self.deque.copy()

    # 确保对原始 deque 的直接访问是线程安全的
    def __getitem__(self, position):
        with self.lock:
            return self.deque[position]

    def __iter__(self):
        with self.lock:
            # 创建一个列表副本用于安全的迭代
            return iter(list(self.deque))