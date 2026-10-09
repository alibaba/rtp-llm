"""CPU-only API/lifetime tests: no torch import and no GPU initialization."""
import contextlib
import types
import unittest

import importlib.util
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "cp_same_layer_workspace.py"
spec = importlib.util.spec_from_file_location("cp_same_layer_workspace_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
SameLayerCPWorkspace = module.SameLayerCPWorkspace


class Tensor:
    def __init__(self, count, dtype, shape=None):
        self.count, self.dtype, self.shape = count, dtype, shape

    def __getitem__(self, key):
        return Tensor(key.stop, self.dtype)

    def view(self, *shape):
        assert self.count == shape[0] * shape[1]
        return Tensor(self.count, self.dtype, shape)

    def numel(self):
        return self.count

    def element_size(self):
        return 1 if self.dtype == "u8" else 2


class Stream:
    def __init__(self, ident, trace):
        self.cuda_stream, self.trace = ident, trace

    def synchronize(self):
        self.trace.append(("side_sync", self.cuda_stream))

    def wait_event(self, event):
        self.trace.append(("wait_event", self.cuda_stream, event))

    def wait_stream(self, stream):
        self.trace.append(("wait_stream", self.cuda_stream, stream.cuda_stream))


class Event:
    def __init__(self, trace):
        self.trace = trace

    def record(self, stream):
        self.trace.append(("record", stream.cuda_stream, self))

    def synchronize(self):
        self.trace.append(("sync", self))


def setup(ce=True, suffix_format="bf16"):
    trace = []
    main, side = Stream(1, trace), Stream(2, trace)
    backend = types.SimpleNamespace(
        mem_allocator="nccl_allocator",
        register_mem_pool=lambda pool, symm: trace.append(("register", pool, symm)),
        deregister_mem_pool=lambda pool: trace.append(("deregister", pool)),
    )
    cuda = types.SimpleNamespace(
        device=lambda _: contextlib.nullcontext(),
        is_current_stream_capturing=lambda: False,
        current_stream=lambda _: main,
        Event=lambda: Event(trace),
        MemPool=lambda allocator: object(),
        use_mem_pool=lambda pool: contextlib.nullcontext(),
    )
    def empty(n, dtype, device):
        trace.append(("allocate", n, dtype))
        return Tensor(n, dtype)
    torch = types.SimpleNamespace(cuda=cuda, empty=empty, uint8="u8", bfloat16="bf16")
    workspace = SameLayerCPWorkspace(
        "cuda:0", backend, main, side, 2 if ce else 0, torch_module=torch,
        environ={"RTP_LLM_CP_PACKED_KV_OVERLAP": "1", "NCCL_CTA_POLICY": "2" if ce else "0"},
        suffix_format=suffix_format,
    )
    return workspace, trace


def finish(workspace, lease):
    workspace.wait_for_consumer(lease, Event([]))
    workspace.retire(lease)


class WorkspaceTests(unittest.TestCase):
    def test_wire_suffix_allocator_shapes_payload_and_retirement(self):
        w, trace = setup(suffix_format="nvfp4_wire_v1")
        lease = w.acquire_suffix(7)
        self.assertEqual([x[1:] for x in trace if x[0] == "allocate"],
                         [(7 * 648 + 1024, "u8"), (28 * 648 + 1024, "u8")])
        self.assertEqual(lease.tensors["kv_recv"].shape, (28, 648))
        self.assertEqual(w.payload_bytes(), 5 * 7 * 648 + 2048)
        finish(w, lease)
        done = w.slots["suffix"].done
        trace.clear()
        smaller = w.acquire_suffix(3)
        self.assertEqual(smaller.tensors["kv_recv"].shape, (12, 648))
        self.assertEqual(trace[0], ("wait_event", 2, done))
        self.assertFalse(any(x[0] in ("allocate", "register", "sync", "side_sync") for x in trace))
        finish(w, smaller)
        w.close()

    def test_wire_preserves_prefix_layout_and_cross_slot_growth_fence(self):
        w, trace = setup(suffix_format="nvfp4_wire_v1")
        prefix = w.acquire_prefix(2)
        finish(w, prefix)
        self.assertEqual(prefix.kind, "prefix")
        done = w.slots["prefix"].done
        trace.clear()
        suffix = w.acquire_suffix(3)
        self.assertEqual(trace[0], ("side_sync", 2))
        self.assertLess(trace.index(("sync", done)),
                        next(i for i, x in enumerate(trace) if x[0] == "register"))
        finish(w, suffix)
        w.close()

    def test_unknown_suffix_format_fails(self):
        with self.assertRaisesRegex(ValueError, "suffix format"):
            setup(suffix_format="future_format")

    def test_factory_rejects_live_format_change(self):
        from unittest.mock import patch
        w, _ = setup(suffix_format="nvfp4_wire_v1")
        group = object()
        collective = types.SimpleNamespace(
            Group=types.SimpleNamespace(TP_SIDE="side"),
            _get_group=lambda _: group,
            _owned_cp_workspaces={(group, "cuda:0"): w},
        )
        package = types.ModuleType("rtp_llm.models_py.distributed")
        package.collective_torch = collective
        with patch.dict(sys.modules, {package.__name__: package}):
            self.assertIs(module.get_workspace(
                "cuda:0", w.main_stream, w.side_stream,
                suffix_format="nvfp4_wire_v1"), w)
            with self.assertRaisesRegex(RuntimeError, "format changed"):
                module.get_workspace("cuda:0", w.main_stream, w.side_stream)

    def test_prefix_allocation_order_and_shapes(self):
        w, trace = setup()
        lease = w.acquire_prefix(3)
        self.assertEqual([x[1:] for x in trace if x[0] == "allocate"], [
            (3 * 17408 + 1024, "u8"), (12 * 17408 + 1024, "u8"),
            (3 * 65536 + 1024, "u8"), (12 * 65536 + 1024, "u8"),
        ])
        self.assertEqual(lease.tensors["values_recv"].shape, (12, 65536))
        self.assertEqual(len([x for x in trace if x[0] == "register"]), 1)

    def test_smaller_rows_contiguous_rank_packed_and_fenced(self):
        w, trace = setup()
        first = w.acquire_suffix(8)
        finish(w, first)
        done = w.slots["suffix"].done
        trace.clear()
        next_lease = w.acquire_suffix(3)
        self.assertEqual(next_lease.tensors["kv_recv"].shape, (12, 1152))
        self.assertEqual(trace[0], ("wait_event", 2, done))
        self.assertFalse(any(x[0] in ("allocate", "register", "sync", "side_sync") for x in trace))

    def test_grow_waits_before_deregister_and_new_register(self):
        w, trace = setup()
        first = w.acquire_prefix(2)
        finish(w, first)
        trace.clear()
        w.acquire_prefix(3)
        steps = [x[0] for x in trace]
        self.assertEqual(steps[0], "side_sync")
        self.assertLess(steps.index("sync"), steps.index("deregister"))
        self.assertEqual(steps.count("register"), 1)
        self.assertLess(steps.index("deregister"), steps.index("allocate"))
        self.assertLess(steps.index("allocate"), steps.index("register"))

    def test_new_prefix_registration_waits_previous_suffix_consumer(self):
        w, trace = setup()
        suffix = w.acquire_suffix(8)
        finish(w, suffix)
        previous = w.slots["suffix"].done
        trace.clear()
        w.acquire_prefix(4)
        self.assertEqual(trace[0], ("side_sync", 2))
        self.assertLess(trace.index(("sync", previous)),
                        next(i for i, x in enumerate(trace) if x[0] == "register"))

    def test_close_completes_both_slots_before_first_deregister(self):
        w, trace = setup()
        prefix = w.acquire_prefix(4)
        suffix = w.acquire_suffix(8)
        finish(w, prefix)
        finish(w, suffix)
        previous = [slot.done for slot in w.slots.values()]
        trace.clear()
        w.close()
        first_drop = next(i for i, x in enumerate(trace) if x[0] == "deregister")
        self.assertEqual(trace[0], ("side_sync", 2))
        for event in previous:
            self.assertLess(trace.index(("sync", event)), first_drop)
        self.assertEqual(w.payload_bytes(), 0)

    def test_zero_prefix_does_not_create_pool(self):
        w, trace = setup()
        self.assertIsNone(w.acquire_prefix(0))
        self.assertEqual(trace, [])
        suffix = w.acquire_suffix(4)
        self.assertEqual(len([x for x in trace if x[0] == "register"]), 1)
        finish(w, suffix)
        w.close()

    def test_independent_prefix_and_suffix_leases(self):
        w, _ = setup()
        prefix = w.acquire_prefix(2)
        suffix = w.acquire_suffix(4)
        finish(w, prefix)
        self.assertIs(w.slots["suffix"].active, suffix)
        finish(w, suffix)
        w.close()

    def test_ordinary_does_not_register(self):
        w, trace = setup(ce=False)
        p = w.acquire_prefix(1)
        finish(w, p)
        w.close()
        self.assertFalse(any(x[0] in ("register", "deregister") for x in trace))

    def test_illegal_reuse_and_close(self):
        w, _ = setup()
        p = w.acquire_prefix(1)
        with self.assertRaisesRegex(RuntimeError, "active consumer"):
            w.acquire_prefix(2)
        with self.assertRaisesRegex(RuntimeError, "retire all"):
            w.close()
        with self.assertRaisesRegex(RuntimeError, "readiness"):
            w.retire(p)
        finish(w, p)
        with self.assertRaisesRegex(RuntimeError, "expired"):
            w.retire(p)
        w.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            w.acquire_prefix(1)

    def test_graph_and_stream_changes_fail(self):
        w, _ = setup()
        w.torch.cuda.is_current_stream_capturing = lambda: True
        with self.assertRaisesRegex(RuntimeError, "eager-only"):
            w.acquire_prefix(1)
        w.torch.cuda.is_current_stream_capturing = lambda: False
        w.torch.cuda.current_stream = lambda _: Stream(9, [])
        with self.assertRaisesRegex(RuntimeError, "stream changed"):
            w.acquire_suffix(1)

    def test_rank_schedule_mock_sequence_equal(self):
        schedules = []
        for rank in range(4):
            w, trace = setup()
            # Local rank never enters workspace decisions. Global stage rows
            # are identical even when one rank has zero valid prefix blocks.
            for prefix_rows, suffix_rows in ((0, 3), (2, 4), (1, 2), (3, 4)):
                p = w.acquire_prefix(prefix_rows)
                s = w.acquire_suffix(suffix_rows)
                if p is not None:
                    finish(w, p)
                finish(w, s)
            w.close()
            schedules.append([tuple(x[:1]) + (x[1:] if x[0] == "allocate" else ())
                              for x in trace if x[0] in ("allocate", "register", "deregister")])
        self.assertTrue(all(s == schedules[0] for s in schedules))

    def test_memory_accounting(self):
        w, _ = setup()
        w.acquire_prefix(2)
        w.acquire_suffix(3)
        self.assertEqual(w.payload_bytes(), 5 * (2 * 82944 + 3 * 2304) + 8192)

    def test_older_torch_default_policy_needs_no_ce_api(self):
        backend = types.SimpleNamespace(options=types.SimpleNamespace())
        self.assertEqual(module._effective_cta_policy(
            backend, {"NCCL_CTA_POLICY": "0"}), 0)
        with self.assertRaisesRegex(RuntimeError, "actual CTA policy=2"):
            module._effective_cta_policy(backend, {"NCCL_CTA_POLICY": "2"})

    def test_efficiency_selector_preserves_ordinary_allocator(self):
        backend = types.SimpleNamespace(options=types.SimpleNamespace(
            config=types.SimpleNamespace(cta_policy=-2147483648)))
        self.assertEqual(module._effective_cta_policy(
            backend, {"NCCL_CTA_POLICY": "1"}), 1)

    def test_ce_policy_requires_communicator_configuration(self):
        backend = types.SimpleNamespace(options=types.SimpleNamespace(
            config=types.SimpleNamespace(cta_policy=2)))
        self.assertEqual(module._effective_cta_policy(
            backend, {"NCCL_CTA_POLICY": "2"}), 2)

    def test_policy_mismatch_fails(self):
        w, _ = setup()
        with self.assertRaisesRegex(ValueError, "effective CTA policy"):
            SameLayerCPWorkspace("cuda:0", w.backend, w.main_stream, w.side_stream,
                                 0, torch_module=w.torch,
                                 environ={"RTP_LLM_CP_PACKED_KV_OVERLAP": "1", "NCCL_CTA_POLICY": "2"})


if __name__ == "__main__":
    unittest.main()
