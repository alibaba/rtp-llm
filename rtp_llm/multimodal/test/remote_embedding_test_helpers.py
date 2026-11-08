"""Shared remote-cache test doubles; no test-suite import side effects."""

import asyncio
import threading

from rtp_llm.multimodal.greennet_hook import (
    GreenNetHandle,
    GreenNetProvider,
    GreenNetVerdict,
)


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

    def save_one(self, key, value, **kwargs):
        self.write_gate.wait(3)
        if self.fail_write:
            raise RuntimeError("unknown save outcome")
        self.objects.setdefault(key, value.clone())
        self.saved.set()

    def object_size(self, key, **kwargs):
        value = self.objects.get(key)
        return None if value is None else value.numel()

    def load_one(self, key, value, **kwargs):
        self.read_started.set()
        self.read_gate.wait(3)
        if self.fail_read:
            raise RuntimeError("read unavailable")
        value.copy_(self.objects[key])

    def close(self):
        self.closed = True


class FakeClient:
    instance_id = "kve_unit"

    def __init__(self, storage):
        self.storage = storage
        self.fail_load = self.fail_save = False
        self.closed = False

    def object_size(self, key, **kwargs):
        return self.storage[key].numel() if key in self.storage else None

    def load_one(self, key, output, **kwargs):
        output.copy_(self.storage[key])
        if self.fail_load:
            output.fill_(23)
            raise TimeoutError("partial destination")

    def save_one(self, key, value, **kwargs):
        if self.fail_save:
            raise TimeoutError("write failed")
        self.storage[key] = value.clone()

    def close(self):
        self.closed = True


class SourceProvider(GreenNetProvider):
    def __init__(self, passed=True, hold=False):
        self.passed = passed
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.calls = []
        self.request_ids = []

    def is_enabled(self):
        return True

    async def preprocess_and_submit(self, *args):
        raise AssertionError("worker-source path must not use URL rewriting")

    async def submit_sources(self, request, mm_inputs, sources):
        self.calls.append(([item.url for item in mm_inputs], list(sources)))
        self.request_ids.append(request.id)
        provider = self

        class Handle(GreenNetHandle):
            async def wait_result(self):
                while not provider.release.is_set():
                    await asyncio.sleep(0.001)
                return GreenNetVerdict(provider.passed, 1 if provider.passed else 2)

            def cancel(self):
                pass

        return Handle()
