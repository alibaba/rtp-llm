"""Existing BF16 sparse-prefill provider: identity, routing and frozen math.

CPU uses meta payloads for host admission and an independent float64 oracle
for small real PPU cases. Large same-input arms retain every row and the
existing strict per-row/global SSE-energy metric; no tolerance is introduced.
"""

from contextlib import ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer, layers
from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_attention_provider as provider
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache, _StagedSwaSegments
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_native_write import fixture as native_fixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import FLAGS
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import _qualification_metric


ENV, RECEIPT = provider.ENV, provider.RECEIPT_ENV
PREFIX = 126976
_ORIGINAL_DEVICE_REASON = provider._device_reason


class _Object:
    def __init__(self, **values):
        self.__dict__.update(values)


def _fixture(device="meta", rows=4096, ratio=0, prefix=PREFIX):
    geometry = NS(head_dim=512, n_heads=64, window=128, ratio=lambda layer: ratio,
                  kv_source_for=lambda layer: 2 if ratio else None, sm_scale=lambda: 512 ** -0.5)
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache.sharded, cache._is_prefill, cache.lengths = False, True, (prefix + rows,)
    cache.geometry = geometry
    state = _Object(prefill=True, idle=False, compacted=False, context=None, cache=cache,
                    lengths=(rows,), query_prefixes=(prefix,),
                    inputs=NS(input_ids=torch.empty(rows, device=device, dtype=torch.int64),
                              combo_position_ids=torch.arange(prefix, prefix + rows)),
                    attn=NS(cu_seqlens=torch.tensor([0, rows], dtype=torch.int32), is_cuda_graph=False))
    model = NS(_is_decode_role=False, _is_speculative=False, _prefill_cp_size=1,
               parallelism_config=NS(tp_size=1))
    state.runtime = _Object(current=state, model=model, policy=SharedPrefillPolicy(mode="compact"))
    bank = rows + min(prefix, 127) + ((prefix + rows) // ratio if ratio else 0)
    tensors = (torch.empty(rows, 64, 512, device=device, dtype=torch.bfloat16),
               torch.empty(bank, 1, 512, device=device, dtype=torch.bfloat16),
               torch.empty(rows, 1, 640, device=device, dtype=torch.int32),
               torch.empty(rows, device=device, dtype=torch.int32),
               torch.empty(64, device=device, dtype=torch.float32))
    return state, geometry, tensors


def _fake_binding(callback=lambda *args, **kwargs: (None, None, None), *, filename="<candidate-test>"):
    extension = ModuleType(provider.EXTENSION_MODULE)
    extension.sparse_prefill_fwd = lambda: None
    namespace = dict(flash_mla_cuda=extension, callback=callback)
    exec(compile("def native(*args, **kwargs):\n    return callback(*args, **kwargs)\n", filename, "exec"), namespace)
    return provider._Provider(namespace["native"], extension, extension.sparse_prefill_fwd,
                              provider.candidate_receipt())


def _cpu_reference(q, kv, indices, lengths, sink):
    """Independent sparse softmax with a zero-valued natural-log sink token."""
    q, kv, indices, lengths = (value.detach().cpu() for value in (q, kv, indices, lengths))
    sink = None if sink is None else sink.detach().cpu().double()
    output = torch.zeros_like(q)
    for row in range(q.shape[0]):
        selected = indices[row, 0, :int(lengths[row])].long()
        selected = selected[(selected >= 0) & (selected < kv.shape[0])]
        if not selected.numel():
            continue
        values = kv[selected, 0].double()
        logits = (q[row].double() @ values.T) * (512 ** -0.5)
        if sink is not None:
            logits = torch.cat((logits, sink[:, None]), dim=-1)
        probabilities = torch.softmax(logits, -1)[..., :selected.numel()]
        answer = probabilities @ values
        if sink is not None:
            answer = torch.where(torch.isposinf(sink[:, None]), torch.zeros_like(answer), answer)
        output[row] = answer.to(torch.bfloat16)
    return output


def _guarded(shape, dtype, device, value=73):
    storage = torch.full((math.prod(shape) + 32,), value, dtype=dtype, device=device)
    return storage[16:-16].view(shape), storage


def _real_inputs(rows, ratio, *, prefix=PREFIX):
    bank = rows + min(prefix, 127) + ((prefix + rows) // ratio if ratio else 0)
    q, qraw = _guarded((rows, 64, 512), torch.bfloat16, "cuda:0")
    kv, kvraw = _guarded((bank, 1, 512), torch.bfloat16, "cuda:0")
    indices, iraw = _guarded((rows, 1, 640), torch.int32, "cuda:0")
    lengths, lraw = _guarded((rows,), torch.int32, "cuda:0")
    sink, sraw = _guarded((64,), torch.float32, "cuda:0")
    q.copy_(torch.randn_like(q) * 0.125)
    kv.copy_(torch.randn_like(kv) * 0.125)
    sink.copy_(torch.linspace(-2, 2, 64, device=q.device))
    indices.fill_(-1)
    row = torch.arange(rows, device=q.device)[:, None]
    global_rows = (prefix + rows) // ratio if ratio else 0
    begin = 0
    if ratio:
        visible = (prefix + row + 1) // ratio
        indices[:, 0, :512] = ((row * 37 + torch.arange(512, device=q.device) * 31) % visible).int()
        begin = 512
    indices[:, 0, begin:begin + 128] = global_rows + row + torch.arange(128, device=q.device)
    lengths.fill_(begin + 128)
    return (q, kv, indices, lengths, sink), (qraw, kvraw, iraw, lraw, sraw)


class LocalAttentionProviderCPU(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, {ENV: "1", RECEIPT: "/candidate-receipt.json"}))
        self.stack.enter_context(patch.object(layers, "is_ppu", return_value=True))
        self.stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch.object(provider, "_device_reason", new=lambda _q: None))

    def test_default_late_flag_inactive_constructor_close_and_stable_stats(self):
        with patch.dict(os.environ):
            os.environ.pop(ENV)
            inactive = provider.LocalPrefillAttentionProvider(NS())
        with patch.object(provider, "load_provider", side_effect=AssertionError("default must not import")):
            self.assertIsNone(inactive.try_run(None, 0, None, None, None, None, None, None))
        self.assertFalse(inactive.stats["enabled"])
        partial = provider.LocalPrefillAttentionProvider(NS(idle=True))
        self.assertIsNone(partial.try_run(None, 0, None, None, None, None, None, None))
        stats = partial.stats
        partial.close()
        partial.close()
        self.assertIs(stats, partial.stats)
        self.assertTrue(stats["closed"])
        with patch.dict(os.environ, {ENV: "true"}), self.assertRaisesRegex(ValueError, "explicitly 0 or 1"):
            provider.LocalPrefillAttentionProvider(NS())

    def test_receipt_schema_pins_and_explicit_configuration_errors(self):
        receipt = provider.candidate_receipt()
        self.assertIsNone(provider.receipt_reason(receipt))
        self.assertEqual(receipt["qualification"], "IDENTITY_VERIFIED_NUMERIC_PENDING")
        for name in receipt:
            with self.subTest(field=name):
                self.assertIsNotNone(provider.receipt_reason(dict(receipt, **{name: "wrong"})))
        self.assertIsNotNone(provider.receipt_reason({"format": "rtp-v41-packed-flashmla-v1",
                                                     "native_entry": "sparse_decode_fwd"}))
        self.assertIsNotNone(provider.receipt_reason([]))
        with self.assertRaises(ValueError):
            provider.load_provider("relative.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            with self.assertRaises(FileNotFoundError):
                provider.load_provider(str(path))
            path.write_text("{")
            with self.assertRaises(json.JSONDecodeError):
                provider.load_provider(str(path))
            path.write_text(json.dumps(dict(receipt, extension_sha256="0" * 64)))
            with self.assertRaisesRegex(ValueError, "extension_sha256"):
                provider.load_provider(str(path))

    def test_import_binds_real_wrapper_extension_entry_and_file_hashes(self):
        provider._import_provider.cache_clear()
        self.addCleanup(provider._import_provider.cache_clear)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            package, wrapper, binary = (Path(directory) / name for name in ("__init__.py", "interface.py", "native.so"))
            for path in (package, wrapper, binary):
                path.write_bytes(path.name.encode())
            binding = _fake_binding(filename=str(wrapper))
            module = ModuleType(provider.MODULE)
            module.__file__, module.flash_mla_sparse_fwd = str(package), binding.native
            binding.extension.__file__ = str(binary)
            for name, path in (("PACKAGE_SHA256", package), ("WRAPPER_SHA256", wrapper), ("EXTENSION_SHA256", binary)):
                stack.enter_context(patch.object(provider, name, hashlib.sha256(path.read_bytes()).hexdigest()))
            stack.enter_context(patch.object(provider.importlib, "import_module",
                                            side_effect=lambda name: module if name == provider.MODULE else binding.extension))
            got = provider._import_provider()
            self.assertIs(got.native, binding.native)
            self.assertIs(got.extension, binding.extension)
            provider._import_provider.cache_clear()
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash_mismatch:extension"):
                provider._import_provider()
            binding.native.__globals__["flash_mla_cuda"] = object()
            with self.assertRaisesRegex(ValueError, "native_binding_mismatch"):
                provider._import_provider()

    def test_phase_cache_cp_grad_graph_and_stream_gates_preserve_fallback(self):
        mutations = [lambda s, flag=flag: setattr(s, flag, True) for flag in
                     ("idle", "is_graph", "is_graph_commit", "verify", "proposal", "compacted", "restoring_aux")]
        mutations += [lambda s: setattr(s, "prefill", False), lambda s: setattr(s.cache, "sharded", True),
                      lambda s: setattr(s, "context", NS(cp_size=8, kv_cache_sharded=True)),
                      lambda s: setattr(s, "context", NS(cp_size=1, kv_cache_sharded=False, gather_restore_positions=object())),
                      lambda s: setattr(s.runtime.model, "_is_decode_role", True),
                      lambda s: setattr(s.runtime.model, "_is_speculative", True),
                      lambda s: setattr(s.runtime.model.parallelism_config, "tp_size", 2)]
        with patch.object(provider, "load_provider", side_effect=AssertionError("ineligible import")):
            for mutate in mutations:
                state, geometry, values = _fixture()
                helper = provider.LocalPrefillAttentionProvider(state)
                mutate(state)
                self.assertIsNone(helper.try_run(state, 0, geometry, *values))
            state, geometry, values = _fixture()
            helper = provider.LocalPrefillAttentionProvider(state)
            with torch.enable_grad():
                self.assertIsNone(helper.try_run(state, 0, geometry, *values))
        # Exercise the real pure-host stream predicate separately from meta payload admission.
        with patch.object(provider, "is_ppu", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(8, 9)) as arch, \
             patch("torch.cuda.current_device", return_value=0) as device, \
             patch("torch.cuda.is_current_stream_capturing", return_value=False) as capture, \
             patch("torch.cuda.current_stream", return_value=1) as stream, \
             patch("torch.cuda.default_stream", return_value=2):
            q = NS(device=torch.device("cuda:0"))
            self.assertEqual(_ORIGINAL_DEVICE_REASON(q), "stream_or_capture")
            stream.return_value = 2
            self.assertIsNone(_ORIGINAL_DEVICE_REASON(q))
            capture.return_value = True
            self.assertEqual(_ORIGINAL_DEVICE_REASON(q), "stream_or_capture")
            capture.return_value = False
            arch.return_value = (8, 0)
            self.assertEqual(_ORIGINAL_DEVICE_REASON(q), "device_arch")
            arch.return_value = (8, 9)
            device.return_value = 1
            self.assertEqual(_ORIGINAL_DEVICE_REASON(q), "stream_or_capture")

    def test_normal_tensor_shape_geometry_and_layout_decline_without_copy(self):
        state, geometry, values = _fixture()
        cases = [(20, geometry, values), (0, NS(**dict(vars(geometry), head_dim=256)), values)]
        q, kv, indices, lengths, sink = values
        cases += [(0, geometry, (q.float(), kv, indices, lengths, sink)),
                  (0, geometry, (q, kv[:-1], indices, lengths, sink)),
                  (0, geometry, (q, kv, indices[..., :128], lengths, sink)),
                  (0, geometry, (q, kv, indices, lengths.long(), sink)),
                  (0, geometry, (q, kv, indices, lengths, sink[:-1])),
                  (0, geometry, (torch._neg_view(q), kv, indices, lengths, sink)),
                  (0, geometry, (q.to(torch.complex64).conj(), kv, indices, lengths, sink)),
                  (0, geometry, (q.transpose(0, 1), kv, indices, lengths, sink)),
                  (0, geometry, (q, kv, indices, None, sink))]
        with patch.object(provider, "load_provider", side_effect=AssertionError("must decline before import")):
            for layer, g, operands in cases:
                self.assertIsNone(provider.LocalPrefillAttentionProvider(state).try_run(state, layer, g, *operands))
            for rows in (0, 140, 4095, 4098):
                other, g, operands = _fixture(rows=rows)
                self.assertIsNone(provider.LocalPrefillAttentionProvider(other).try_run(other, 0, g, *operands))

    def test_birth_metadata_and_owner_rebind_changed_back_never_remint(self):
        for kind in ("cu", "positions", "ids", "cache", "receipt", "foreign"):
            state, geometry, values = _fixture()
            helper = provider.LocalPrefillAttentionProvider(state)
            if kind in ("cu", "positions"):
                value = state.attn.cu_seqlens if kind == "cu" else state.inputs.combo_position_ids
                value.add_(1).sub_(1)
            elif kind == "ids":
                state.inputs.input_ids = state.inputs.input_ids.clone()
            elif kind == "cache":
                state.cache = _fixture()[0].cache
            elif kind == "foreign":
                state = _fixture()[0]
            with patch.dict(os.environ, {RECEIPT: "/other.json"} if kind == "receipt" else {}), \
                 patch.object(provider, "load_provider", side_effect=AssertionError("invalid ancestor import")):
                self.assertIsNone(helper.try_run(state, 0, geometry, *values), kind)
            self.assertIsNone(helper.try_run(state, 0, geometry, *values), kind)

    def test_actual_arguments_twenty_layers_stats_identity_and_native_error_no_retry(self):
        state, geometry, values = _fixture()
        seen = []
        def call(*args, **kwargs):
            seen.append((tuple(id(value) for value in args[:3]), kwargs))
            return (None, None, None)
        binding = _fake_binding(call)
        with patch.object(provider, "load_provider", return_value=binding) as load:
            helper = provider.LocalPrefillAttentionProvider(state)
            for layer in range(20):
                self.assertEqual(helper.try_run(state, layer, geometry, *values), (None, None, None))
            self.assertIsNone(helper.try_run(state, 19, geometry, *values))
        self.assertEqual(load.call_count, 1)
        self.assertEqual(helper.stats["calls"], 20)
        self.assertEqual(helper.stats["layers"], list(range(20)))
        self.assertEqual(len(helper.stats["operands"]), 20)
        self.assertEqual(helper.stats["operands"][0]["kv"]["shape"], [4223, 1, 512])
        self.assertEqual(helper.stats["identity"]["extension_sha256"], provider.EXTENSION_SHA256)
        self.assertTrue(all(ids == tuple(map(id, values[:3])) for ids, _ in seen))
        self.assertTrue(all(options["attn_sink"] is values[4] and options["topk_length"] is values[3]
                            and options["d_v"] == 512 for _, options in seen))
        def fail(*args, **kwargs):
            raise RuntimeError("native failure")
        with patch.object(provider, "load_provider", return_value=_fake_binding(fail)):
            helper = provider.LocalPrefillAttentionProvider(state)
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                helper.try_run(state, 0, geometry, *values)
        self.assertEqual(helper.stats["attempts"], 1)
        self.assertEqual(helper.stats["calls"], 0)
        self.assertEqual(helper.stats["native_errors"], 1)

    def test_missing_provider_fresh_stats_and_no_payload_state_retention(self):
        state, geometry, values = _fixture()
        def unavailable(*args, **kwargs):
            raise ImportError("unavailable")
        with patch.object(provider, "load_provider", side_effect=unavailable) as load:
            helper = provider.LocalPrefillAttentionProvider(state)
            self.assertIsNone(helper.try_run(state, 0, geometry, *values))
            self.assertIsNone(helper.try_run(state, 1, geometry, *values))
            self.assertEqual(load.call_count, 1)
        with patch.object(provider, "load_provider", return_value=_fake_binding()):
            helper = provider.LocalPrefillAttentionProvider(state)
            helper.try_run(state, 0, geometry, *values)
        stats = helper.stats
        state_ref, payload_ref, metadata_ref = weakref.ref(state), weakref.ref(values[0]), weakref.ref(state.inputs.input_ids)
        del state, values
        gc.collect()
        self.assertIsNone(state_ref())
        self.assertIsNone(payload_ref())
        self.assertIsNotNone(metadata_ref())
        helper.close()
        gc.collect()
        self.assertIsNone(metadata_ref())
        self.assertIs(stats, helper.stats)
        self.assertEqual(stats["calls"], 1)
        self.assertTrue(stats["closed"])
        newer = provider.LocalPrefillAttentionProvider(_fixture()[0])
        self.assertIsNot(stats, newer.stats)
        json.dumps(stats)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalAttentionProviderDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(provider.is_ppu(), "qualification requires actual PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.assertTrue(Path(os.environ.get(RECEIPT, "")).is_absolute(), "explicit independent candidate receipt required")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, {ENV: "1"}))
        self.binding = provider.load_provider(os.environ[RECEIPT])
        self.assertEqual(self.binding.identity["extension_sha256"], provider.EXTENSION_SHA256)
        self.saved_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", self.saved_tf32)
        self.reports = []

    def qualify(self, actual, expected, label):
        metric = _qualification_metric(torch, actual, expected)
        self.assertEqual(metric["qualified_status"], "PASS", (label, metric))
        self.reports.append(dict(label=label, metric=metric))

    def candidate(self, values):
        q, kv, indices, lengths, sink = values
        return self.binding.native(q, kv, indices, 512 ** -0.5, d_v=512,
                                   attn_sink=sink, topk_length=lengths)

    def old(self, values):
        q, kv, indices, lengths, sink = values
        return flash_mla_sparse_fwd_ppu(q, kv, indices, 512 ** -0.5,
                                        attn_sink=sink, topk_length=lengths)

    def test_small_independent_cpu_oracle_boundaries_holes_duplicates_causal_sink(self):
        rows, bank = 9, 1024
        q = torch.randn(rows, 64, 512, device="cuda:0", dtype=torch.bfloat16) * 0.125
        kv = torch.randn(bank, 1, 512, device="cuda:0", dtype=torch.bfloat16) * 0.125
        indices = torch.arange(640, device=q.device, dtype=torch.int32)[None, None].expand(rows, 1, -1).clone()
        indices[:, :, 3] = -1
        indices[:, :, 5] = bank + 19
        indices[:, :, 7] = indices[:, :, 6]  # duplicates are counted, not deduplicated
        lengths = torch.tensor([0, 1, 127, 128, 639, 640, 2, 128, 640], device=q.device, dtype=torch.int32)
        # Each row has its own causal prefix; future IDs remain -1 within
        # scanned prefixes, preserving holes rather than extending visibility.
        columns = torch.arange(640, device=q.device)
        visible = torch.tensor([0, 1, 127, 128, 639, 640, 2, 128, 640], device=q.device)[:, None, None]
        indices.masked_fill_(columns[None, None] >= visible, -1)
        for sink in (None, torch.linspace(-2, 2, 64, device=q.device),
                     torch.full((64,), float("inf"), device=q.device),
                     torch.full((64,), -float("inf"), device=q.device)):
            values = q, kv, indices, lengths, sink
            snapshots = tuple(value.clone() if value is not None else None for value in values)
            reference = _cpu_reference(*values)
            old, actual = self.old(values)[0], self.candidate(values)[0]
            self.qualify(old, reference, "small old/CPU")
            self.qualify(actual, reference, "small candidate/CPU")
            self.qualify(actual, old, "small same-input")
            for value, snapshot in zip(values, snapshots):
                if value is not None:
                    assert_bytes(self, value, snapshot)
        self.report("small-oracle")

    def test_real_full_and_lookahead_banks_all_rows_same_input_and_canaries(self):
        for rows in (4096, 4097):
            for ratio in (0, 2):
                values, raw = _real_inputs(rows, ratio)
                snapshots = tuple(value.clone() for value in raw)
                old, actual = self.old(values), self.candidate(values)
                for name, got, expected in zip(("output", "max_logits", "lse_base2"), actual, old):
                    self.qualify(got, expected, f"M{rows}.ratio{ratio}.{name}")
                    self.assertFalse(any(torch._C._is_alias_of(got, source) for source in values))
                for value, expected in zip(raw, snapshots):
                    assert_bytes(self, value, expected)
                self.assertEqual(values[1].shape[0], (4223 if rows == 4096 else 4224)
                                 + (65536 if ratio else 0))
        self.report("full-banks")

    def test_ieee_classification_signed_zero_and_ignored_padding_match_old(self):
        q = torch.zeros(8, 64, 512, device="cuda:0", dtype=torch.bfloat16)
        kv = torch.zeros(1024, 1, 512, device=q.device, dtype=q.dtype)
        q[:, 1::2].fill_(-0.0)
        kv[1::2].fill_(-0.0)
        q[1, 0, 0] = float("nan")
        q[2, 1, 0] = float("inf")
        kv[900, 0, 0], kv[901, 0, 0] = float("inf"), float("nan")
        indices = torch.arange(640, device=q.device, dtype=torch.int32)[None, None].expand(8, 1, -1).clone()
        indices[3, 0, 0], indices[4, 0, 0] = 900, 901
        indices[5, 0, -1] = 901
        lengths = torch.tensor([0, 1, 128, 128, 640, 639, 128, 640], device=q.device, dtype=torch.int32)
        sink = torch.zeros(64, device=q.device)
        sink[2], sink[3] = float("inf"), -float("inf")
        values = q, kv, indices, lengths, sink
        frozen = tuple(value.clone() for value in values)
        for got, old in zip(self.candidate(values), self.old(values)):
            for classifier in (torch.isnan, torch.isposinf, torch.isneginf, torch.isfinite):
                self.assertTrue(torch.equal(classifier(got), classifier(old)), classifier.__name__)
            zero = (got == 0) & (old == 0)
            self.assertTrue(torch.equal(torch.signbit(got)[zero], torch.signbit(old)[zero]))
        for value, old in zip(values, frozen):
            assert_bytes(self, value, old)

    def test_actual_helper_default_stream_and_unsupported_routes_fall_back(self):
        state, geometry, _ = _fixture("cuda:0")
        values, _ = _real_inputs(4096, 0)
        helper = provider.LocalPrefillAttentionProvider(state)
        self.assertIsNotNone(helper.try_run(state, 0, geometry, *values))
        self.assertEqual(helper.stats["calls"], 1)
        for layer, operands in ((20, values), (1, (values[0].float(), *values[1:])),
                                (1, (values[0], values[1], values[2][..., :128], *values[3:]))):
            self.assertIsNone(helper.try_run(state, layer, geometry, *operands))
        other = provider.LocalPrefillAttentionProvider(state)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self.assertIsNone(other.try_run(state, 0, geometry, *values))
        torch.cuda.current_stream().wait_stream(side)
        self.assertEqual(other.stats["calls"], 0)
        self.assertEqual(other.stats["fallbacks"], {"stream_or_capture": 1})
        compact, g, _ = _fixture("cuda:0", rows=140)
        self.assertIsNone(provider.LocalPrefillAttentionProvider(compact).try_run(compact, 19, g,
                                                                                *(value[:140] for value in values)))
        with patch.dict(os.environ, {ENV: "0"}):
            late = provider.LocalPrefillAttentionProvider(state)
        self.assertIsNone(late.try_run(state, 1, geometry, *values))

    def test_real_provider_profile_grid_identity_and_no_helper_copy(self):
        values, _ = _real_inputs(4096, 2)
        self.candidate(values)
        state, geometry, _ = _fixture("cuda:0", ratio=2)
        helper = provider.LocalPrefillAttentionProvider(state)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            result = helper.try_run(state, 2, geometry, *values)
            torch.cuda.synchronize()
        self.assertIsNotNone(result)
        events = _trace_events(profile, "local-attention-provider-full")
        kernels = [event for event in events if event.get("cat") == "kernel"]
        self.assertEqual(len(kernels), 1, [event["name"] for event in kernels])
        self.assertIn("flash_sparse_prefill_fwd_hs64", kernels[0]["name"])
        sms = torch.cuda.get_device_properties(values[0].device).multi_processor_count
        expected = min(math.ceil(math.ceil(4096 / 3) / sms) * sms, 4096 // 2)
        self.assertEqual(kernels[0]["args"]["grid"], [expected, 1, 1])
        self.assertEqual(kernels[0]["args"]["block"], [512, 1, 1])
        cpu_ops = [event["name"] for event in events if event.get("cat") == "cpu_op"]
        self.assertNotIn("aten::clone", cpu_ops)
        self.assertNotIn("aten::copy_", cpu_ops)
        markers = [event["name"] for event in events if event.get("cat") == "user_annotation"]
        self.assertEqual(markers.count("dsv41.ppu.local_attention_provider.L02.full"), 1)
        self.assertEqual(helper.stats["identity"]["extension_sha256"], provider.EXTENSION_SHA256)
        self.reports.append(dict(operands=helper.stats["operands"], grid=kernels[0]["args"]["grid"]))
        self.report("profile")

    def test_actual_attention_body_boundary_and_source_pool_bytes(self):
        for rows, layer in ((4096, 0), (4097, 0), (140, 21)):
            q = torch.randn(rows, 64, 512, device="cuda:0", dtype=torch.bfloat16) * 0.125
            kv = torch.randn(rows, 512, device=q.device, dtype=q.dtype) * 0.125 + 1
            hidden = torch.zeros(rows, 8, device=q.device, dtype=q.dtype)
            frozen_inputs = tuple(value.clone() for value in (q, kv, hidden))
            arms = []
            for enabled in (False, True):
                with patch.dict(os.environ, {**FLAGS, ENV: str(int(enabled)),
                    "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1", "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE_FUSED": "1",
                    "RTP_V41_PPU_FUSED_BF16_ROPE": "1", attention_layer.FUSION_ENV: "0", attention_layer.STREAMS_ENV: "0",
                    "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE": "0"}):
                    state = native_fixture("cuda:0", (PREFIX,), (rows,), layers=layer + 1)
                    geometry = NS(hidden_size=8, n_heads=64, head_dim=512, window=128,
                                  o_groups=1, o_lora_rank=8, ratio=lambda layer: 0,
                                  kv_source_for=lambda layer: None, sm_scale=lambda: 512 ** -0.5)
                    state.cache.geometry = geometry
                    model = state.runtime.model
                    model._is_decode_role, model._is_speculative, model._prefill_cp_size = False, False, 1
                    model.parallelism_config = NS(tp_size=1)
                    state.runtime.current = state
                    state.runtime.qkv_streams = None
                    frequencies = torch.ones(PREFIX + rows, 32, device=q.device, dtype=torch.complex64)
                    state.runtime.freqs_for = lambda layer: frequencies
                    state.local_attention_provider = provider.LocalPrefillAttentionProvider(state)
                    attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
                    nn.Module.__init__(attn)
                    attn.geometry, attn.layer_id, attn.compress_ratio = geometry, layer, 0
                    attn.is_kv_source = attn.is_index_source = False
                    attn._cp_ctx, attn.runtime = None, state.runtime
                    attn.ops = NS(encode_kv=encode_kv, decode_kv=decode_kv)
                    attn.project_query_kv = lambda value, **kwargs: (value, q.clone(), kv.clone(), None)
                    attn.wo_a = nn.Linear(64 * 512, 8, bias=False, device=q.device, dtype=q.dtype)
                    attn.wo_a.weight.data.fill_(1 / (64 * 512))
                    attn.wo_b, attn.attn_sink = nn.Identity(), torch.zeros(64, device=q.device)
                    snapshots = []
                    release = state.cache.release_swa
                    def save_and_release(owner):
                        saved = state.cache._fresh_swa[owner, 0]
                        if isinstance(saved, _StagedSwaSegments):
                            snapshots.append(("segments", saved.prefix, saved.tail.clone(), saved.fresh.clone()))
                        else:
                            self.assertIsInstance(saved, tuple)
                            self.assertEqual(len(saved), 2)
                            snapshots.append(("legacy", saved[0].clone(), saved[1].clone()))
                        release(owner)
                    with patch.object(attention_layer, "prepare_packed_attention", return_value=(None, "disabled")), \
                         patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False), \
                         patch.object(state.cache, "release_swa", side_effect=save_and_release), \
                         patch.object(attention_layer, "flash_mla_sparse_fwd_ppu", wraps=flash_mla_sparse_fwd_ppu) as old:
                        result = attn._body(hidden)
                    expected_new = enabled and layer < 20
                    self.assertEqual(state.local_attention_provider.stats["calls"], int(expected_new))
                    self.assertEqual(old.call_count, int(not expected_new))
                    self.assertEqual(state.shared["n5_routes"][-1]["attention_backend"],
                                     "local_bf16_sparse_prefill_provider" if expected_new else "flash_mla_sparse_fwd")
                    arms.append((result, tuple(pool.tensor.clone() for pool in state.cache.pools.values()),
                                 list(state.cache.writes), snapshots))
            self.qualify(arms[1][0], arms[0][0], f"actual body M{rows} L{layer}")
            for got, expected in zip(arms[1][1], arms[0][1]):
                assert_bytes(self, got, expected)
            self.assertEqual(arms[1][2], arms[0][2])
            self.assertEqual(len(arms[1][3]), len(arms[0][3]))
            for got, expected in zip(arms[1][3], arms[0][3]):
                self.assertEqual(got[0], expected[0])
                first_tensor = 2 if got[0] == "segments" else 1
                if first_tensor == 2:
                    self.assertEqual(got[1], expected[1])
                self.assertEqual(len(got), len(expected))
                for value, frozen in zip(got[first_tensor:], expected[first_tensor:]):
                    assert_bytes(self, value, frozen)
            for value, old in zip((q, kv, hidden), frozen_inputs):
                assert_bytes(self, value, old)
        self.report("body")

    def report(self, label):
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            (Path(directory) / f"local-attention-provider-{label}.json").write_text(json.dumps(dict(
                contract="frozen per-row/global SSE-energy strictly<5e-5", identity=self.binding.identity,
                cases=self.reports,
            ), indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
