"""In-memory tensor object store shared by embedding-cache tests."""

import threading


class Store:
    def __init__(self, objects=None):
        self.objects = {} if objects is None else objects
        self.saved = threading.Event()
        self.read_started = threading.Event()
        self.read_gate = threading.Event()
        self.read_gate.set()
        self.write_gate = threading.Event()
        self.write_gate.set()
        self.closed = False
        self.fail_read = self.fail_write = False
        self.instance_id = "kve_test"
        self.instance_group = "kve_test_group"

    def save_one(self, key, value):
        self.write_gate.wait(3)
        if self.fail_write:
            raise RuntimeError("unknown save outcome")
        self.objects.setdefault(key, value.clone())
        self.saved.set()

    def object_size(self, key, **kwargs):
        value = self.objects.get(key)
        return None if value is None else value.numel()

    def load_one(self, key, value):
        self.read_started.set()
        self.read_gate.wait(3)
        if self.fail_read:
            raise RuntimeError("read unavailable")
        value.copy_(self.objects[key])

    def close(self):
        self.closed = True
