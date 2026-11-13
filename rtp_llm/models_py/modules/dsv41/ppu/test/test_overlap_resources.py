"""Resource ownership and distinct eager/Verify DAGs at the CUDA boundary.

CPU tests inspect enqueue order, rejection before submission and failure
cleanup. They do not establish native communication progress or Graph replay.
The existing device/real-WORLD8 suites remain the native qualification.
"""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import overlap_resources, projection_streams
from rtp_llm.models_py.modules.dsv41.ppu.overlap_resources import OverlapResources
from rtp_llm.models_py.modules.dsv41.ppu.verify_moe_overlap import VerifyMoEOverlap


class _Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, trace, name):
        value = torch.Tensor._make_subclass(cls, torch.empty(0), False)
        value.trace, value.label = trace, name
        return value

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda", 0)

    def record_stream(self, stream):
        self.trace.append(("retain", self.label, stream.label))

    def reshape(self, shape):
        return self


class _Stream:
    def __init__(self, boundary, label, index, device="cuda:0"):
        self.boundary, self.label, self.cuda_stream = boundary, label, index
        self.device = torch.device(device)

    def wait_event(self, event):
        self.boundary.trace.append(("wait", self.label, event.label))


class _Event:
    def __init__(self, boundary, label):
        self.boundary, self.label = boundary, label

    def record(self, stream):
        self.boundary.trace.append(("record", self.label, stream.label))


class _CUDA:
    def __init__(self):
        self.trace, self.streams, self.events = [], [], []
        self.default = _Stream(self, "caller", 0)
        self.current, self.capturing = self.default, False

    def new_stream(self, device=None):
        index = len(self.streams) + 1
        stream = _Stream(self, "side" + str(index), index, device)
        self.streams.append(stream)
        return stream

    def new_event(self, **kwargs):
        event = _Event(self, "event" + str(len(self.events)))
        self.events.append(event)
        return event

    @contextmanager
    def on_stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    @contextmanager
    def patched(self):
        replacements = dict(
            Stream=self.new_stream,
            Event=self.new_event,
            device=lambda device: nullcontext(),
            stream=self.on_stream,
            current_device=lambda: 0,
            current_stream=lambda device=None: self.current,
            default_stream=lambda device=None: self.default,
            is_current_stream_capturing=lambda: self.capturing,
        )
        with ExitStack() as stack:
            for name, replacement in replacements.items():
                stack.enter_context(patch.object(torch.cuda, name, replacement))
            yield self


def _verify(boundary):
    device = torch.device("cuda", 0)
    engine = NS(
        backend="megamoe",
        precision="w4a4",
        ep_size=8,
        hidden_size=5120,
        inter_dim=2304,
        num_routed_experts=384,
        _mega_buffer=object(),
        _mega_capacity=32,
        _mega_device=device,
        _mega_generation=1,
        _processed={(layer, device): () for layer in range(40)},
    )
    with patch(
        "rtp_llm.models_py.modules.dsv41.ppu.verify_moe_overlap.is_ppu",
        return_value=True,
    ):
        plan = VerifyMoEOverlap(device, engine)
    plan.bind_caller(boundary.current)
    return plan


class OverlapResourceContracts(unittest.TestCase):
    def test_cpu_and_capture_reject_before_allocating(self):
        with _CUDA().patched() as cuda:
            with self.assertRaisesRegex(ValueError, "torch.cuda"):
                OverlapResources("cpu")
            cuda.capturing = True
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                OverlapResources("cuda:0")
            self.assertEqual((cuda.streams, cuda.events, cuda.trace), ([], [], []))

    def test_bound_caller_stable_resources_and_close_without_device_sync(self):
        with _CUDA().patched() as cuda:
            resources = OverlapResources("cuda")
            initial = resources.storage_signature()
            caller = _Stream(cuda, "native", 77)
            resources.bind_caller(caller)
            first = list(cuda.trace)
            self.assertEqual(
                first,
                [
                    ("record", "event0", "native"),
                    ("wait", "side1", "event0"),
                    ("record", "event1", "side1"),
                    ("wait", "native", "event1"),
                ],
            )
            resources.bind_caller(caller)
            self.assertEqual(cuda.trace, first)
            self.assertEqual(initial, resources.storage_signature())
            with self.assertRaisesRegex(RuntimeError, "caller changed"):
                resources.bind_caller(cuda.default)
            with self.assertRaisesRegex(ValueError, "another device"):
                resources.bind_caller(_Stream(cuda, "other", 3, "cuda:1"))
            cuda.capturing = True
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                resources.bind_caller(caller)
            cuda.capturing = False
            resources.close()
            self.assertNotEqual(initial, resources.storage_signature())
            with self.assertRaisesRegex(RuntimeError, "closed"):
                with resources.submission():
                    self.fail("closed resource submitted")
            self.assertEqual(cuda.trace, first)
            self.assertEqual((len(cuda.streams), len(cuda.events)), (1, 2))

    def test_nonreentrant_exception_and_active_close(self):
        with _CUDA().patched():
            resources = OverlapResources("cuda:0")
            with self.assertRaisesRegex(RuntimeError, "callback"):
                with resources.submission():
                    with self.assertRaisesRegex(RuntimeError, "not reentrant"):
                        with resources.submission():
                            self.fail("reentrant")
                    with self.assertRaisesRegex(RuntimeError, "active"):
                        resources.close()
                    raise RuntimeError("callback")
            self.assertFalse(resources.lock.locked())
            with resources.submission():
                pass

    def test_handles_have_one_owner_and_replacement_requires_recapture(self):
        with _CUDA().patched() as cuda:
            eager = projection_streams.QKVStreams("cuda:0")
            plan = _verify(cuda)
            for owner in (eager, plan):
                for name in ("side_stream", "producer", "done"):
                    self.assertIs(getattr(owner, name), getattr(owner._resources, name))
                    with self.assertRaises(AttributeError):
                        setattr(owner, name, object())
            initial = plan.storage_signature()
            previous = plan._resources.done
            plan._resources.done = cuda.new_event()
            self.assertIs(plan.done, plan._resources.done)
            self.assertNotEqual(initial, plan.storage_signature())
            with self.assertRaisesRegex(RuntimeError, "recapture"):
                plan.validate()
            plan._resources.done = previous
            plan.validate()

    def test_nested_owners_include_views_and_ignore_cpu_values(self):
        with _CUDA().patched() as cuda:
            first, second = _Tensor(cuda.trace, "view"), _Tensor(cuda.trace, "kv")
            overlap_resources.record_stream(
                {"q": (first, [second]), "metadata": (torch.empty(0), None, 7)},
                cuda.default,
            )
            self.assertEqual(
                cuda.trace, [("retain", "view", "caller"), ("retain", "kv", "caller")]
            )

    def test_eager_side_first_done_before_caller_and_nested_results(self):
        with _CUDA().patched() as cuda:
            resources = projection_streams.QKVStreams("cuda:0")
            value = _Tensor(cuda.trace, "input")

            def branch(name):
                def run(value):
                    cuda.trace.append(("branch", name, cuda.current.label))
                    return _Tensor(cuda.trace, name)

                return run

            for _ in range(2):
                cuda.trace.clear()
                resources.run(value, branch("query"), branch("kv"))
                self.assertEqual(
                    cuda.trace,
                    [
                        ("record", "event0", "caller"),
                        ("wait", "side1", "event0"),
                        ("retain", "input", "side1"),
                        ("branch", "kv", "side1"),
                        ("record", "event1", "side1"),
                        ("branch", "query", "caller"),
                        ("retain", "kv", "caller"),
                        ("retain", "query", "caller"),
                        ("wait", "caller", "event1"),
                    ],
                )
            self.assertEqual(resources.calls, 2)
            self.assertEqual((len(cuda.streams), len(cuda.events)), (1, 2))
            resources.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                resources.run(value, branch("query"), branch("kv"))

    def test_eager_wrong_caller_capture_and_callback_failure(self):
        for failure in ("kv", "query"):
            with self.subTest(failure=failure), _CUDA().patched() as cuda:
                resources = projection_streams.QKVStreams("cuda:0")
                value = _Tensor(cuda.trace, "input")
                calls = []

                def callback(name):
                    def run(value):
                        calls.append(name)
                        if name == failure:
                            raise RuntimeError(name)
                        return value

                    return run

                cuda.current = _Stream(cuda, "wrong", 9)
                with self.assertRaisesRegex(ValueError, "default eager"):
                    resources.run(value, callback("query"), callback("kv"))
                cuda.current, cuda.capturing = cuda.default, True
                with self.assertRaisesRegex(ValueError, "default eager"):
                    resources.run(value, callback("query"), callback("kv"))
                self.assertEqual((calls, cuda.trace), ([], []))
                cuda.capturing = False
                with self.assertRaisesRegex(RuntimeError, failure):
                    resources.run(value, callback("query"), callback("kv"))
                self.assertEqual(calls, ["kv"] if failure == "kv" else ["kv", "query"])
                self.assertEqual(cuda.trace[-1], ("wait", "caller", "event1"))
                self.assertFalse(resources._lock.locked())
                self.assertEqual(resources.calls, 0)

    def test_verify_shared_first_routed_side_and_join_before_return(self):
        with _CUDA().patched() as cuda:
            cuda.current = _Stream(cuda, "native", 77)
            plan = _verify(cuda)
            state = NS(
                bulk_counters=True,
                metadata=NS(valid_mask=None),
                routed_rows_device=[None] * 40,
            )
            value = _Tensor(cuda.trace, "input")

            def callback(name, output):
                def run(*args, **kwargs):
                    cuda.trace.append(("branch", name, cuda.current.label))
                    if name == "shared":
                        self.assertEqual(kwargs, {"allow_quant": False})
                    return output

                return run

            owner = NS(
                layer_id=0,
                shared_expert=callback("shared", value),
                gate=callback("gate", (None, None)),
            )
            plan.engine.forward_shared_graph = callback(
                "routed", _Tensor(cuda.trace, "routed")
            )
            with patch.object(plan, "supported", return_value=True):
                cuda.trace.clear()
                plan.forward(owner, value, (32, 5120), state, return_shared=True)
            self.assertEqual(
                cuda.trace,
                [
                    ("record", "event0", "native"),
                    ("wait", "side1", "event0"),
                    ("retain", "input", "side1"),
                    ("branch", "shared", "native"),
                    ("branch", "gate", "side1"),
                    ("branch", "routed", "side1"),
                    ("retain", "routed", "native"),
                    ("record", "event1", "side1"),
                    ("wait", "native", "event1"),
                ],
            )
            self.assertEqual((plan.receipt()["forks"], plan.receipt()["joins"]), (1, 1))

    def test_verify_each_callback_failure_joins_once_without_retry(self):
        for failure in ("shared", "gate", "routed"):
            with self.subTest(failure=failure), _CUDA().patched() as cuda:
                plan = _verify(cuda)
                calls, value = [], _Tensor(cuda.trace, "input")

                def callback(name, output):
                    def run(*args, **kwargs):
                        calls.append(name)
                        if name == failure:
                            raise RuntimeError("original " + name)
                        return output

                    return run

                state = NS(
                    bulk_counters=True,
                    metadata=NS(valid_mask=None),
                    routed_rows_device=[None] * 40,
                )
                owner = NS(
                    layer_id=0,
                    shared_expert=callback("shared", value),
                    gate=callback("gate", (None, None)),
                )
                plan.engine.forward_shared_graph = callback("routed", value)
                cuda.trace.clear()
                with patch.object(plan, "supported", return_value=True):
                    with self.assertRaisesRegex(RuntimeError, "original " + failure):
                        plan.forward(
                            owner, value, (32, 5120), state, return_shared=True
                        )
                expected = ["shared", "gate", "routed"]
                self.assertEqual(calls, expected[: expected.index(failure) + 1])
                self.assertEqual(
                    cuda.trace[-2:],
                    [("record", "event1", "side1"), ("wait", "caller", "event1")],
                )
                self.assertFalse(plan._lock.locked())
                self.assertEqual(
                    (plan.receipt()["forks"], plan.receipt()["joins"]), (1, 1)
                )

    def test_verify_join_failure_preserves_original_callback_error(self):
        for callback_fails in (False, True):
            with self.subTest(callback_fails=callback_fails), _CUDA().patched() as cuda:
                plan = _verify(cuda)
                value = _Tensor(cuda.trace, "input")
                state = NS(
                    bulk_counters=True,
                    metadata=NS(valid_mask=None),
                    routed_rows_device=[None] * 40,
                )

                def shared(*args, **kwargs):
                    if callback_fails:
                        raise RuntimeError("original callback")
                    return value

                owner = NS(
                    layer_id=0, shared_expert=shared, gate=lambda *a, **k: (None, None)
                )
                plan.engine.forward_shared_graph = lambda *a, **k: value
                with patch.object(plan, "supported", return_value=True), patch.object(
                    plan._resources, "join", side_effect=RuntimeError("join failed")
                ):
                    message = "original callback" if callback_fails else "join failed"
                    with self.assertRaisesRegex(RuntimeError, message):
                        plan.forward(
                            owner, value, (32, 5120), state, return_shared=True
                        )
                self.assertFalse(plan._lock.locked())


if __name__ == "__main__":
    unittest.main()
