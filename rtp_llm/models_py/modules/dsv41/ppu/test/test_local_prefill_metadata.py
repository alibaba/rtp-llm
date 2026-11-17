"""CPU contracts for same-forward Prefill context and position-cast reuse.

Use real row/compact-plan construction; mock only accelerator admission and
stream APIs. Installed-provider authentication has a separate CPU seam.
"""

import os
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import (
    LOOKUP_ENV,
    compact_lookup_state,
)


@contextmanager
def local_metadata_cpu_stream():
    """Simulate only CUDA admission/current-stream APIs on real CPU tensors."""
    stream = NS(handle=19, device=None, capture=False)
    with patch.object(torch.Tensor, "is_cuda", new=property(lambda _: True)), \
         patch.object(torch.cuda, "current_stream", side_effect=lambda device: NS(
             device=device if stream.device is None else stream.device, cuda_stream=stream.handle)), \
         patch.object(torch.cuda, "is_current_stream_capturing", side_effect=lambda: stream.capture), \
         patch.object(torch.cuda, "default_stream", side_effect=AssertionError("must use actual current stream")), \
         patch.object(torch.cuda, "synchronize", side_effect=AssertionError("no metadata synchronization")):
        yield stream



class LocalPrefillMetadataCPU(unittest.TestCase):
    device = "cpu"

    @torch.inference_mode()
    def test_local_metadata_defaults_activate_real_full_and_compact_rows(self):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context

        with patch.dict(os.environ, {}, clear=True):
            state, cp, rows = self._local_metadata_fixture()
            for compact in (False, True):
                if compact:
                    state._activate_compact(torch.empty((140, 1)))
                    cp, rows = state.context, state.rows(state.context)
                with local_metadata_cpu_stream():
                    context = state.local_prefill_step_context(cp, rows, builder=build_step_context)
                    positions = state.local_prefill_positions32(cp, rows)
                    self.assertIsNotNone(context)
                    self.assertIsNotNone(positions)
                    self.assertTrue(torch.equal(positions, rows[0].int()))
                    self.assertIs(state.local_prefill_step_context(cp, rows, builder=build_step_context), context)
            for flag in ("RTP_V41_PPU_LOCAL_PREFILL_ROWS", "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA"):
                other, context, values = self._local_metadata_fixture()
                with self.subTest(flag=flag), patch.dict(os.environ, {flag: "0"}):
                    with local_metadata_cpu_stream():
                        self.assertIsNone(other.local_prefill_positions32(context, values))

    def _local_metadata_fixture(self, compact=False):
        state = compact_lookup_state(self)
        state.runtime.current = state
        if compact:
            state._activate_compact(torch.empty((140, 1)))
        return state, state.context, state.rows(state.context)

    @torch.inference_mode()
    def test_local_metadata_real_epochs_preserve_bytes_versions_and_original_rows(self):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            state, context, rows = self._local_metadata_fixture()
            calls, results = [], []

            def builder(*args, **kwargs):
                calls.append(args)
                return build_step_context(*args, **kwargs)

            for compact in (False, True):
                if compact:
                    state._activate_compact(torch.empty((140, 1)))
                    context, rows = state.context, state.rows(state.context)
                original_rows = rows
                # The production caller unpacks rows and builds a new tuple.
                # Tensor owners, not the temporary container, define the birth.
                call_rows = tuple(value for value in rows)
                self.assertIsNot(call_rows, original_rows)
                expected = build_step_context(*rows[:3], valid_mask=rows[3])
                expected32 = rows[0].to(torch.int32)
                with local_metadata_cpu_stream(), patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("no readback"),
                ), patch.object(torch.Tensor, "item", side_effect=AssertionError("no scalar read")):
                    actual = state.local_prefill_step_context(context, call_rows, builder=builder)
                    positions32 = state.local_prefill_positions32(context, tuple(value for value in rows))
                    self.assertIsNotNone(actual)
                    self.assertIsNotNone(positions32)
                    # Hits must not redo the large birth proof/capture chain.
                    with patch.object(state, "local_indexer_lookup_slices", side_effect=AssertionError("repeated birth proof")):
                        self.assertIs(state.local_prefill_step_context(
                            context, tuple(value for value in rows), builder=builder), actual)
                        self.assertIs(state.local_prefill_positions32(
                            context, tuple(value for value in rows)), positions32)
                for name in ("positions", "valid_mask", "request_ids", "owner_ranks", "request_offsets"):
                    value, reference = getattr(actual, name), getattr(expected, name)
                    self.assertFalse(value.is_inference(), name)
                    self.assertEqual((value.dtype, value.shape, value.stride()),
                                     (reference.dtype, reference.shape, reference.stride()))
                    self.assertTrue(torch.equal(value.view(torch.uint8), reference.view(torch.uint8)), name)
                self.assertIsNone(actual.parallel_context)
                self.assertIs(actual.valid_mask, rows[3])
                self.assertEqual(actual.positions.data_ptr(), rows[0].data_ptr())
                self.assertEqual(actual.request_ids.data_ptr(), rows[1].data_ptr())
                self.assertFalse(positions32.is_inference())
                self.assertEqual(positions32.dtype, torch.int32)
                self.assertTrue(torch.equal(positions32.view(torch.uint8), expected32.view(torch.uint8)))
                self.assertIs(state.rows(context), original_rows)
                self.assertEqual(rows[0].dtype, torch.int64)
                self.assertEqual(rows[2].dtype, torch.int32)
                results.append((actual, positions32))
            self.assertEqual(len(calls), 2)
            self.assertIsNot(results[0][0], results[1][0])
            self.assertNotEqual(results[0][0].owner_ranks.data_ptr(), results[1][0].owner_ranks.data_ptr())
            self.assertNotEqual(results[0][1].data_ptr(), results[1][1].data_ptr())
            fresh, fresh_cp, fresh_rows = self._local_metadata_fixture()
            self.assertFalse(fresh._local_metadata_entries)
            with local_metadata_cpu_stream():
                self.assertIsNot(fresh.local_prefill_positions32(fresh_cp, fresh_rows), results[0][1])

    @torch.inference_mode()
    def test_local_metadata_source_mutation_rebind_and_unknown_birth_stay_invalid(self):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            for compact in (False, True):
                for field in range(4):
                    for change in ("write_restore", "rebind", "inference", "before_first"):
                        with self.subTest(compact=compact, field=field, change=change):
                            state, cp, rows = self._local_metadata_fixture(compact)
                            with local_metadata_cpu_stream():
                                if change != "before_first":
                                    self.assertIsNotNone(state.local_prefill_step_context(cp, rows, builder=build_step_context))
                                actual = rows
                                if change in ("rebind", "inference"):
                                    with torch.inference_mode(change == "inference"):
                                        replacement = rows[field].clone()
                                    actual = tuple(replacement if i == field else value for i, value in enumerate(rows))
                                elif rows[field].dtype == torch.bool:
                                    rows[field].logical_not_().logical_not_()
                                else:
                                    rows[field].add_(1).sub_(1)
                                self.assertIsNone(state.local_prefill_positions32(cp, actual))
                                self.assertIsNone(state.local_prefill_step_context(cp, rows, builder=build_step_context))
                                self.assertTrue(state._local_metadata_invalid)
            state, cp, rows = self._local_metadata_fixture()
            state._indexer_lookup_stamp = None
            with local_metadata_cpu_stream():
                self.assertIsNone(state.local_prefill_positions32(cp, rows))

    @torch.inference_mode()
    def test_local_metadata_destination_fields_and_private_cast_reject_mutation(self):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            for field in ("positions", "valid_mask", "request_ids", "owner_ranks", "request_offsets", "positions32", "parallel_context"):
                for change in ("write_restore", "rebind", "inference"):
                    with self.subTest(field=field, change=change):
                        state, cp, rows = self._local_metadata_fixture()
                        with local_metadata_cpu_stream():
                            context = state.local_prefill_step_context(cp, rows, builder=build_step_context)
                            cast = state.local_prefill_positions32(cp, rows)
                            entry = state._local_metadata_entries[False]
                            if field == "parallel_context":
                                original = None
                                object.__setattr__(context, field, object())
                            elif field == "positions32":
                                cast.add_(1).sub_(1)
                            else:
                                original = getattr(context, field)
                                if change == "write_restore":
                                    if original.dtype == torch.bool:
                                        original.logical_not_().logical_not_()
                                    else:
                                        original.add_(1).sub_(1)
                                else:
                                    with torch.inference_mode(change == "inference"):
                                        replacement = original.clone()
                                    object.__setattr__(context, field, replacement)
                                    self.assertTrue(any(value is original for value in entry.retained),
                                                    "field replacement must retain the original in-flight allocation")
                            self.assertIsNone(state.local_prefill_positions32(cp, rows))
                            if field != "positions32":
                                object.__setattr__(context, field, original)
                            self.assertIsNone(state.local_prefill_step_context(cp, rows, builder=build_step_context))
                            self.assertIs(entry.values["step"][0], context, "invalid entries retain in-flight owners")
                            self.assertIs(entry.values["positions32"][0], cast)

    @torch.inference_mode()
    def test_local_metadata_phase_current_owner_plan_and_cache_changes_are_sticky(self):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            for change in ("current", "cache", "plan", "inputs", "attn", "prefill", "idle", "verify", "graph", "capture", "cp8", "rows_cache"):
                with self.subTest(change=change):
                    state, cp, rows = self._local_metadata_fixture(compact=True)
                    with local_metadata_cpu_stream() as stream:
                        self.assertIsNotNone(state.local_prefill_step_context(cp, rows, builder=build_step_context))
                        target, name, replacement = {
                            "current": (state.runtime, "current", None),
                            "cache": (state, "cache", NS(sharded=False)),
                            "plan": (state, "plan", replace(state.plan)),
                            "inputs": (state, "inputs", NS(**vars(state.inputs))),
                            "attn": (state, "attn", NS(**vars(state.attn))),
                            "prefill": (state, "prefill", False), "idle": (state, "idle", True),
                            "verify": (state, "verify", True), "graph": (state, "is_graph", True),
                            "capture": (stream, "capture", True), "cp8": (cp, "cp_size", 8),
                            "rows_cache": (state, "_local_prefill_rows", {}),
                        }[change]
                        missing = object()
                        old = getattr(target, name, missing)
                        setattr(target, name, replacement)
                        self.assertIsNone(state.local_prefill_positions32(cp, rows))
                        if old is missing:
                            delattr(target, name)
                        else:
                            setattr(target, name, old)
                        self.assertIsNone(state.local_prefill_step_context(cp, rows, builder=build_step_context))

    @torch.inference_mode()
    def test_local_metadata_current_stream_uses_handle_device_and_sticky_fallback(self):
        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            for change in ("handle", "device"):
                state, cp, rows = self._local_metadata_fixture()
                with local_metadata_cpu_stream() as stream:
                    first = state.local_prefill_positions32(cp, rows)
                    self.assertIsNotNone(first)
                    # The API returns a fresh wrapper each time, same native handle.
                    self.assertIs(state.local_prefill_positions32(cp, rows), first)
                    if change == "handle":
                        stream.handle += 1
                    else:
                        stream.device = torch.device("cuda:1")
                    self.assertIsNone(state.local_prefill_positions32(cp, rows))
                    stream.handle, stream.device = 19, None
                    self.assertIsNone(state.local_prefill_positions32(cp, rows))
                    self.assertIs(state._local_metadata_entries[False].values["positions32"][0], first)

    @torch.inference_mode()
    def test_local_metadata_builder_change_and_construction_failures_do_not_retry(self):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            state, cp, rows = self._local_metadata_fixture()
            with local_metadata_cpu_stream():
                self.assertIsNotNone(state.local_prefill_step_context(cp, rows, builder=build_step_context))
                self.assertIsNone(state.local_prefill_step_context(cp, rows, builder=lambda *a, **k: None))
                self.assertIsNone(state.local_prefill_step_context(cp, rows, builder=build_step_context))
            for kind in ("builder", "cast"):
                state, cp, rows = self._local_metadata_fixture()
                calls = []

                def fail(*args, **kwargs):
                    calls.append(1)
                    raise RuntimeError("single metadata construction")

                with local_metadata_cpu_stream(), ExitStack() as stack:
                    if kind == "cast":
                        stack.enter_context(patch.object(torch.Tensor, "to", new=fail))
                    with self.assertRaisesRegex(RuntimeError, "single metadata construction"):
                        if kind == "builder":
                            state.local_prefill_step_context(cp, rows, builder=fail)
                        else:
                            state.local_prefill_positions32(cp, rows)
                    self.assertEqual(calls, [1])
                    self.assertIsNone(state.local_prefill_positions32(cp, rows))
                    self.assertEqual(calls, [1])

    @torch.inference_mode()
    def test_local_metadata_change_during_constructor_returns_once_without_reuse(self):
        from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            for change in ("stream", "source"):
                state, cp, rows = self._local_metadata_fixture()
                calls = []
                with local_metadata_cpu_stream() as stream:
                    def builder(*args, **kwargs):
                        value = build_step_context(*args, **kwargs)
                        calls.append(value)
                        if change == "stream":
                            stream.handle += 1
                        else:
                            rows[2].add_(0)
                        return value

                    actual = state.local_prefill_step_context(cp, rows, builder=builder)
                    self.assertIs(actual, calls[0], "completed constructor must not trigger caller retry")
                    self.assertEqual(len(calls), 1)
                    self.assertTrue(state._local_metadata_invalid)
                    self.assertIsNone(state.local_prefill_step_context(cp, rows, builder=builder))
                    self.assertEqual(len(calls), 1)

    @torch.inference_mode()
    def test_local_metadata_original_caller_and_changed_bindings_preserve_fallback(self):
        from rtp_llm.models_py.modules.dsv41.ppu import attention_layer, ops, prefill

        environment = {
            **LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
            attention_layer.OPTIMIZED_DIAGNOSTICS_ENV: "0",
            attention_layer.OPTIMIZED_LATE_FOCUS_ENV: "0", "RTP_V41_PPU_PACKED_ATTENTION": "0",
        }
        with patch.dict(os.environ, environment):
            state, cp, rows = self._local_metadata_fixture()
            attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
            torch.nn.Module.__init__(attn)
            attn.ops = ops.PPUOps.__new__(ops.PPUOps)
            attn.runtime = state.runtime
            self.assertTrue(attention_layer._local_step_context_consumer(attn))
            with local_metadata_cpu_stream():
                first = attention_layer._local_step_context(attn, state, cp, rows)
                self.assertIs(attention_layer._local_step_context(attn, state, cp, rows), first)
                calls = []
                original_builder = prefill.build_step_context

                def custom_builder(*args, **kwargs):
                    calls.append(1)
                    return original_builder(*args, **kwargs)

                with patch.object(attention_layer, "build_step_context", new=custom_builder):
                    left = attention_layer._local_step_context(attn, state, cp, rows)
                    right = attention_layer._local_step_context(attn, state, cp, rows)
                self.assertEqual(calls, [1, 1])
                self.assertTrue(state._local_metadata_invalid)
                self.assertIsNot(left, right)
                self.assertIsNot(left, first)
                self.assertNotEqual(left.owner_ranks.data_ptr(), right.owner_ranks.data_ptr())
            forbidden = lambda *args, **kwargs: self.fail("admission executed an unknown callback")
            for owner, name in (
                (attn, "_select"), (attention_layer.SharedPPUAttention, "_project_indexer"),
                (prefill, "build_step_context"), (prefill.V41StepContext, "__init__"),
                (prefill._CPPrefillWorkspace, "__init__"), (prefill._CPPrefillWorkspace, "indices"),
                (ops, "_index_score"), (ops.PPUOps, "index_score"), (attn.ops, "index_score"),
                (attention_layer, "prepare_packed_attention"),
            ):
                with self.subTest(name=name), patch.object(owner, name, new=forbidden):
                    self.assertFalse(attention_layer._local_step_context_consumer(attn))
            for flag in ("1", "invalid"):
                with patch.dict(os.environ, {"RTP_V41_PPU_PACKED_ATTENTION": flag}):
                    self.assertFalse(attention_layer._local_step_context_consumer(attn))

    def test_local_metadata_mega_readonly_binding_rejects_unknown_and_changed_consumers(self):
        import importlib.metadata
        import tempfile
        from pathlib import Path
        from types import ModuleType

        from rtp_llm.models_py.modules.dsv41.ppu import mega_attention as mega
        from rtp_llm.tools.v41 import preflight

        # CPU binding seam only: installed-file authentication and a pybind
        # entry stand in for the pinned wheel, never execute a native kernel.
        source = (
            "def prefill(token_positions):\n"
            "    return _backend.fused_norm_rope_attn_rope_cast_fwd(token_positions)\n"
        )

        def native_entry(*args, **kwargs):
            self.fail("binding admission must not execute the native entry")

        native_entry.__name__ = "fused_norm_rope_attn_rope_cast_fwd"
        native_entry.__module__ = "flash_mla_cuda"
        native = ModuleType("flash_mla_cuda")
        native.fused_norm_rope_attn_rope_cast_fwd = native_entry
        module = ModuleType("flash_mla.fused_norm_rope_attn_rope_cast")
        module._backend = native
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "fused.py"
            wrapper.write_text(source)
            module.__file__ = str(wrapper)
            exec(compile(source, str(wrapper), "exec", dont_inherit=True), vars(module))
            operation = module.prefill
            with patch.dict(os.environ, {"RTP_V41_PPU_FLASHMLA_BUILD_RECEIPT": "fixture.json"}), \
                    patch.object(importlib.metadata, "distribution", return_value=NS()), \
                    patch.object(preflight, "flashmla_build_identity", return_value={}) as authenticate, \
                    patch.object(mega.inspect, "isbuiltin", side_effect=lambda value: value is native_entry):
                binding = mega._readonly_prefill_binding(module, operation)
                self.assertIsNotNone(binding)
                authenticate.assert_called_once()
                original_code = operation.__code__

                def changed(token_positions):
                    return token_positions.zero_()

                operation.__code__ = changed.__code__
                try:
                    self.assertIsNone(mega._readonly_prefill_binding(module, operation))
                finally:
                    operation.__code__ = original_code
                self.assertIs(operation.__code__, original_code)
                for error in (TypeError("invalid files"), AttributeError("invalid receipt")):
                    with patch.object(preflight, "flashmla_build_identity", side_effect=error):
                        self.assertIsNone(mega._readonly_prefill_binding(module, operation))
            backend = NS(prefill=operation, readonly_prefill=binding)
            consumer = mega.PPUMegaAttention.__new__(mega.PPUMegaAttention)
            consumer.backend = backend
            consumer._readonly_prefill = (backend, binding)
            forbidden = lambda *args, **kwargs: self.fail("unexpected callback/resolver")
            with patch.object(mega, "_backend", side_effect=forbidden):
                self.assertTrue(mega.readonly_prefill_positions(consumer))
                self.assertFalse(mega.readonly_prefill_positions(NS(backend=backend)))
                with patch.object(consumer, "_readonly_prefill", new=None):
                    self.assertFalse(mega.readonly_prefill_positions(consumer))
                for owner, name in (
                    (backend, "prefill"), (module, "prefill"),
                    (native, "fused_norm_rope_attn_rope_cast_fwd"),
                    (mega, "_normal"),
                    *((consumer, name) for name in (
                        "prefill", "prefill_reason", "query_reason", "_common",
                        "_query", "_current", "_indices_ok",
                    )),
                ):
                    with self.subTest(name=name), patch.object(owner, name, new=forbidden):
                        self.assertFalse(mega.readonly_prefill_positions(consumer))
                operation.__code__ = changed.__code__
                try:
                    self.assertFalse(mega.readonly_prefill_positions(consumer))
                finally:
                    operation.__code__ = original_code



if __name__ == "__main__":
    unittest.main()
