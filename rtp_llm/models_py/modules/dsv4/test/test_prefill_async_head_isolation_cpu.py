"""CPU isolation tests for the async prefill-head builder (review P2-3).

The builder thread must never rebind state on the live attention modules: it
builds on private shadow copies snapshotted at kick time, and the live
modules' setters reject builder-thread writes. These tests drive the REAL
``AsyncHeadBuilder`` thread and ``maybe_kick_async_head`` with light stubs for
the CP context and metadata builders (no Triton, no CUDA).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from runpy import run_path
from unittest.mock import patch

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
FP8 = DSV4 / "fp8"
_PREFIX = "rtp_llm.models_py.modules.dsv4"


def _package(name, path=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_package(parent), child, mod)
    return sys.modules[name]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_package(parent), child, mod)
    return mod


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    setattr(_package(parent), child, mod)
    return mod


# ---------------------------------------------------------------------------
# Stub model: nn.Module tree mirroring attn -> compressor / indexer -> compressor
# ---------------------------------------------------------------------------

_MAIN_THREAD = threading.main_thread()


class _Tripwire:
    """Records attribute writes made off the main thread on LIVE instances."""

    off_thread_live_writes: list = []

    def __setattr__(self, name, value):
        if threading.current_thread() is not _MAIN_THREAD and not vars(self).get(
            head_prebuild._SHADOW_ATTR, False
        ):
            _Tripwire.off_thread_live_writes.append((type(self).__name__, name))
        super().__setattr__(name, value)


class _Compressor(_Tripwire, nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.register_buffer("freqs_cis", None)
        self._cp_ctx = None

    def set_cp_ctx(self, ctx):
        fp8_kv_utils.guard_live_module_mutation(self, "cp_ctx")
        self._cp_ctx = ctx


class _Indexer(_Tripwire, nn.Module):
    def __init__(self):
        super().__init__()
        self.compressor = _Compressor()
        self._cp_ctx = None

    def set_cp_ctx(self, ctx):
        fp8_kv_utils.guard_live_module_mutation(self, "cp_ctx")
        self._cp_ctx = ctx


class _Attn(_Tripwire, nn.Module):
    """``_build_shared_prefill_meta`` reads the bound state off ``self``."""

    hook = None  # test hook run inside the build, on the builder thread

    def __init__(self, ratio):
        super().__init__()
        self.compress_ratio = ratio
        self.weight = nn.Parameter(torch.arange(8.0))
        self.compressor = _Compressor() if ratio else None
        self.indexer = _Indexer() if ratio == 4 else None
        self._cp_ctx = None
        self._kv_cache = None
        self._block_tables_by_type = None

    def set_cp_ctx(self, ctx):
        fp8_kv_utils.guard_live_module_mutation(self, "cp_ctx")
        self._cp_ctx = ctx

    def _build_shared_prefill_meta(self, x, sp_int, **kwargs):
        if _Attn.hook is not None:
            _Attn.hook(self)
        # Late binding of compressor freqs, as ``_ensure_freqs_cis_bound``.
        if self.compressor is not None and self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.weight.detach()
        idx = self.indexer
        return types.SimpleNamespace(
            ratio=self.compress_ratio,
            cp_ctx=self._cp_ctx,
            kv_cache=self._kv_cache,
            tables=self._block_tables_by_type,
            compressor_ctx=getattr(self.compressor, "_cp_ctx", None),
            indexer_ctx=getattr(idx, "_cp_ctx", None),
            nested_ctx=getattr(getattr(idx, "compressor", None), "_cp_ctx", None),
            weight_ptr=self.weight.data_ptr(),
        )


@contextmanager
def _bind_attn_cache(attn, kv_cache=None, block_tables_by_type=None):
    # Same contract as production ``attention.bind_attn_cache``.
    fp8_kv_utils.guard_live_module_mutation(attn, "kv-cache binding")
    prev = (attn._kv_cache, attn._block_tables_by_type)
    if kv_cache is not None:
        attn._kv_cache = kv_cache
    if block_tables_by_type is not None:
        attn._block_tables_by_type = block_tables_by_type
    try:
        yield attn
    finally:
        attn._kv_cache, attn._block_tables_by_type = prev


class _V4:
    def __init__(self, attns):
        self.layers = [types.SimpleNamespace(attn=a) for a in attns]
        self.fp8_kv_cache = True
        self.max_seq_len = 1 << 20
        self.args = types.SimpleNamespace(is_mtp=False)


def _recipe_ctx(prefix, tag="live"):
    return types.SimpleNamespace(
        tag=tag,
        cp_size=4,
        cp_rank=1,
        chunk_length=1024,
        padded_seq_len=4096,
        seq_len_full=4096,
        kv_cache_sharded=False,
        chunk_lengths_per_req=(1024,),
        input_lengths_full_host=(4096,),
        prefix_lengths_full_host=(prefix,),
    )


def _predicted_ctx(cp_info, cp_size, cp_rank, chunk, device, *, prefix_lengths, **_):
    return types.SimpleNamespace(
        tag="predicted",
        cp_size=cp_size,
        prefix=int(prefix_lengths[0]),
        global_positions=torch.arange(1024),
        prefix_lengths=prefix_lengths,
        req_id_per_token=torch.zeros(1024, dtype=torch.int32),
    )


def setUpModule():
    global fp8_kv_utils, head_prebuild
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()
    _package("rtp_llm")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.modules")
    _package(_PREFIX, str(DSV4))
    _package(_PREFIX + ".fp8", str(FP8))
    _stub("rtp_llm.models_py.utils.arch", is_sm120=lambda device=None: False)
    _load(_PREFIX + ".kv_cache_utils", DSV4 / "kv_cache_utils.py")
    fp8_kv_utils = _load(_PREFIX + ".fp8._kv_cache_utils", FP8 / "_kv_cache_utils.py")
    _stub(
        _PREFIX + ".cp",
        build_cp_context_for_forward=_predicted_ctx,
        first_position_for_meta=lambda ctx, pos: int(pos[0]),
    )
    _stub(_PREFIX + ".fp8.attention", bind_attn_cache=_bind_attn_cache)
    _stub(_PREFIX + ".fp8.prefill_meta", _meta_shared_build_enabled=lambda: False)
    _stub(
        _PREFIX + ".fp8._compact_cp_runtime",
        _zigzag_constants=lambda: (None, torch.arange(4096)),
    )
    head_prebuild = _load(_PREFIX + ".fp8._head_prebuild", FP8 / "_head_prebuild.py")


class _Base(unittest.TestCase):
    def setUp(self):
        _Tripwire.off_thread_live_writes = []
        _Attn.hook = None
        self.addCleanup(setattr, _Attn, "hook", None)
        saved = head_prebuild._BUILDER
        head_prebuild._BUILDER = None
        self.addCleanup(setattr, head_prebuild, "_BUILDER", saved)
        enabled = patch.object(head_prebuild, "async_head_enabled", return_value=True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def _model(self):
        live_ctx = _recipe_ctx(4096)
        attns = [_Attn(0), _Attn(4), _Attn(128), _Attn(4), _Attn(128)]
        for a in attns:
            a.set_cp_ctx(live_ctx)
            if a.compressor is not None:
                a.compressor.set_cp_ctx(live_ctx)
            if a.indexer is not None:
                a.indexer.set_cp_ctx(live_ctx)
                a.indexer.compressor.set_cp_ctx(live_ctx)
        return _V4(attns), live_ctx

    def _kick(self, v4, live_ctx, kv_cache="kv-N", tables=None):
        head_prebuild.maybe_kick_async_head(
            v4,
            kv_cache,
            tables if tables is not None else {"swa": torch.ones(1, 4)},
            live_ctx,
            torch.device("cpu"),
        )

    def _join(self, key=None):
        return head_prebuild._BUILDER.consume(key)


class GuardContract(_Base):
    def test_constants_match_pool_module(self):
        self.assertEqual(
            head_prebuild._BUILDER_THREAD_ATTR,
            fp8_kv_utils.ASYNC_HEAD_BUILDER_THREAD_ATTR,
        )
        self.assertEqual(
            head_prebuild._SHADOW_ATTR, fp8_kv_utils.ASYNC_HEAD_SHADOW_ATTR
        )

    def test_guard_rejects_live_writes_only_on_builder_thread(self):
        live = fp8_kv_utils.PoolBackedModule()
        shadow = head_prebuild._shadow_module(live, {})
        live.clear_pool_context()  # serving thread: allowed
        outcome = {}

        def run():
            try:
                shadow.clear_pool_context()
                outcome["shadow"] = "ok"
                live.clear_pool_context()
                outcome["live"] = "ok"
            except RuntimeError as exc:
                outcome["live"] = str(exc)

        t = threading.Thread(target=run)
        setattr(t, fp8_kv_utils.ASYNC_HEAD_BUILDER_THREAD_ATTR, True)
        t.start()
        t.join(5)
        self.assertEqual(outcome["shadow"], "ok")
        self.assertIn("mutate pool context on live module", outcome["live"])
        # An unmarked worker thread (e.g. unrelated helpers) is unaffected.
        plain = threading.Thread(target=live.clear_pool_context)
        plain.start()
        plain.join(5)

    def test_builder_thread_is_marked(self):
        seen = {}

        def build():
            seen["marked"] = getattr(
                threading.current_thread(), head_prebuild._BUILDER_THREAD_ATTR, False
            )
            return None

        builder = head_prebuild.AsyncHeadBuilder()
        builder.kick(("k",), build)
        builder.consume(None)
        self.assertTrue(seen["marked"])

    def test_live_module_mutation_on_builder_fails_closed(self):
        v4, live_ctx = self._model()
        attn = v4.layers[1].attn
        builder = head_prebuild._builder()

        def bad_build():
            attn.set_cp_ctx("predicted")  # a regression: live write off-thread
            return "bundle"

        builder.kick(("k",), bad_build)
        self.assertIsNone(builder.consume(("k",)))
        self.assertEqual(builder._failed, 1)
        self.assertIs(attn._cp_ctx, live_ctx)


class ShadowModules(unittest.TestCase):
    def setUp(self):
        _Tripwire.off_thread_live_writes = []

    def test_shadow_writes_do_not_reach_live_tree(self):
        live = _Attn(4)
        shadow = head_prebuild._shadow_module(live, {})
        self.assertIsNot(shadow, live)
        self.assertIsNot(shadow.compressor, live.compressor)
        self.assertIsNot(shadow.indexer, live.indexer)
        self.assertIsNot(shadow.indexer.compressor, live.indexer.compressor)
        # Weights are shared, not copied.
        self.assertIs(shadow.weight, live.weight)
        self.assertIs(shadow.compressor.proj.weight, live.compressor.proj.weight)
        shadow._cp_ctx = "p"
        shadow._kv_cache = "kv"
        shadow.compressor.set_cp_ctx("p")
        shadow.indexer.compressor.set_cp_ctx("p")
        shadow.compressor.freqs_cis = torch.ones(2)  # registered buffer slot
        shadow.compressor = None  # rebinding a child module
        self.assertIsNone(live._cp_ctx)
        self.assertIsNone(live._kv_cache)
        self.assertIsNone(live.compressor._cp_ctx)
        self.assertIsNone(live.indexer.compressor._cp_ctx)
        self.assertIsNone(live.compressor.freqs_cis)
        self.assertIsNotNone(live.compressor)
        self.assertFalse(vars(live).get(head_prebuild._SHADOW_ATTR, False))

    def test_shared_children_stay_shared_within_one_snapshot(self):
        a, b = _Attn(4), _Attn(4)
        b.compressor = a.compressor
        memo = {}
        sa = head_prebuild._shadow_module(a, memo)
        sb = head_prebuild._shadow_module(b, memo)
        self.assertIs(sa.compressor, sb.compressor)
        self.assertIsNot(sa.compressor, a.compressor)


class ConcurrentServingMutation(_Base):
    def test_builder_never_mutates_live_modules(self):
        v4, live_ctx = self._model()
        self._kick(v4, live_ctx)
        self._join()
        self.assertEqual(_Tripwire.off_thread_live_writes, [])
        for layer in v4.layers:
            a = layer.attn
            self.assertIs(a._cp_ctx, live_ctx)
            self.assertIsNone(a._kv_cache)
            if a.compressor is not None:
                self.assertIs(a.compressor._cp_ctx, live_ctx)
                self.assertIsNone(a.compressor.freqs_cis)

    def test_serving_thread_rebinds_while_builder_runs(self):
        """The serving thread rebinds live modules mid-build (e.g. a forward
        that skipped the drain). The build must still see only the predicted
        state, and the builder must not clobber the serving thread's writes."""
        v4, live_ctx = self._model()
        entered, release = threading.Event(), threading.Event()
        calls = []

        def hook(attn):
            calls.append(attn)
            if len(calls) == 1:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("serving thread never released the build")

        _Attn.hook = hook
        tables = {"swa": torch.ones(1, 4)}
        self._kick(v4, live_ctx, kv_cache="kv-N", tables=tables)
        self.assertTrue(entered.wait(5))
        serving_ctx = _recipe_ctx(0, tag="serving")
        for layer in v4.layers:
            a = layer.attn
            a.set_cp_ctx(serving_ctx)
            a._kv_cache = "kv-serving"
            a._block_tables_by_type = {"swa": torch.zeros(1, 4)}
            if a.compressor is not None:
                a.compressor.set_cp_ctx(serving_ctx)
            if a.indexer is not None:
                a.indexer.set_cp_ctx(serving_ctx)
                a.indexer.compressor.set_cp_ctx(serving_ctx)
        release.set()
        key = head_prebuild.predicted_continuation_key(live_ctx, torch.device("cpu"))
        bundle = self._join(key)
        self.assertIsNotNone(bundle)
        self.assertEqual(head_prebuild._BUILDER._failed, 0)
        self.assertEqual(sorted(bundle.meta_by_ratio), [0, 4, 128])
        for r, meta in bundle.meta_by_ratio.items():
            self.assertEqual(meta.cp_ctx.tag, "predicted")
            self.assertEqual(meta.cp_ctx.prefix, 4096 + 4096)
            self.assertEqual(meta.kv_cache, "kv-N")
            self.assertIs(meta.tables, tables)
            if r:
                self.assertEqual(meta.compressor_ctx.tag, "predicted")
            if r == 4:
                self.assertEqual(meta.indexer_ctx.tag, "predicted")
                self.assertEqual(meta.nested_ctx.tag, "predicted")
            # The bundle patches against the LIVE rep (first per ratio).
            self.assertIs(bundle.rep_attns[r], v4.layers[[0, 4, 128].index(r)].attn)
            self.assertEqual(meta.weight_ptr, bundle.rep_attns[r].weight.data_ptr())
        # Serving-thread writes survive; nothing was restored over them.
        for layer in v4.layers:
            a = layer.attn
            self.assertIs(a._cp_ctx, serving_ctx)
            self.assertEqual(a._kv_cache, "kv-serving")
            if a.compressor is not None:
                self.assertIs(a.compressor._cp_ctx, serving_ctx)
        self.assertEqual(_Tripwire.off_thread_live_writes, [])
        # Each build call ran on a shadow, never a live module.
        for attn in calls:
            self.assertTrue(vars(attn).get(head_prebuild._SHADOW_ATTR, False))

    def test_shadow_snapshot_is_taken_at_kick(self):
        """State the serving thread binds after the kick is not seen."""
        v4, live_ctx = self._model()
        gate = threading.Event()
        seen = []

        def hook(attn):
            gate.wait(5)
            seen.append(attn.weight)

        _Attn.hook = hook
        self._kick(v4, live_ctx)
        new_weight = nn.Parameter(torch.zeros(8))
        v4.layers[0].attn.weight = new_weight
        gate.set()
        self._join()
        self.assertIsNot(seen[0], new_weight)


if __name__ == "__main__":
    unittest.main()
