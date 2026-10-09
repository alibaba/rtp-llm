"""CPU tests for the indexer tail stream and its serialized fallback.

A stream dependency model checks producer/consumer fences and rejects missing
waits. Kernel doubles compare inputs and outputs through the production caller.
Hardware stream ordering still requires GPU coverage."""

from __future__ import annotations

import ast
import os
import pathlib
import sys
import types
import unittest
from runpy import run_path
from typing import Any, Dict, NamedTuple, Optional
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve().parent
FP8 = HERE.parent
INDEXER_PATH = FP8 / "indexer.py"
INDEXER_SRC = INDEXER_PATH.read_text()
ATTN_PATH = FP8 / "attention.py"
ATTN_SRC = ATTN_PATH.read_text()

_PREFIX = "rtp_llm.models_py.modules.dsv4"


# ---------------------------------------------------------------------------
# Happens-before DAG model
# ---------------------------------------------------------------------------
class _DAGError(AssertionError):
    pass


class _FakeEvent:
    def __init__(self, dag):
        self._dag = dag
        self.frontier = None

    def record(self, stream=None):
        stream = stream or self._dag.current_stream()
        self.frontier = set(stream.frontier)
        self._dag.log.append(("record", stream.name, len(self.frontier)))


class _FakeStream:
    def __init__(self, dag, name):
        self._dag = dag
        self.name = name
        # Causal frontier: the set of op-ids this stream's next op happens-after.
        self.frontier = set()

    def wait_event(self, event):
        assert event.frontier is not None, "wait on a never-recorded event"
        self.frontier |= event.frontier
        self._dag.log.append(("wait_event", self.name, len(event.frontier)))

    def wait_stream(self, other):
        self.frontier |= other.frontier
        self._dag.log.append(("wait_stream", self.name, len(other.frontier)))


class _FakeStreamCtx:
    def __init__(self, dag, stream):
        self._dag = dag
        self._stream = stream

    def __enter__(self):
        self._dag.push_stream(self._stream)

    def __exit__(self, *exc):
        self._dag.pop_stream()


class _DAG:
    """Exact happens-before tracking for the fake streams.

    Every stubbed kernel calls :meth:`op` with its read/write tensors; a read
    whose writer op-id is not in the reading stream's causal frontier is an
    illegal cross-stream read (the production bug class the tail stream must not
    introduce).
    """

    def __init__(self):
        self.main = _FakeStream(self, "main")
        self._stack = [self.main]
        self.streams = {"main": self.main}
        self.log = []
        self.writer = {}  # storage data_ptr -> op-id
        self.ops = []  # (op-id, name, stream)
        self.errors = []

    @staticmethod
    def _key(t):
        # The allocator works on storages; a slice write IS a parent write.
        return t.untyped_storage().data_ptr()

    def stream(self, name):
        s = self.streams.get(name)
        if s is None:
            s = _FakeStream(self, name)
            self.streams[name] = s
        return s

    def current_stream(self):
        return self._stack[-1]

    def push_stream(self, s):
        self._stack.append(s)

    def pop_stream(self):
        self._stack.pop()

    def op(self, name, reads=(), writes=()):
        stream = self.current_stream()
        op_id = len(self.ops)
        for t in reads:
            w = self.writer.get(self._key(t))
            if w is not None and w not in stream.frontier:
                self.errors.append(
                    f"illegal read of {name}: op {op_id} on {stream.name} reads "
                    f"tensor written by op {w} without a happens-before edge"
                )
        seq = stream.frontier
        seq = set(seq)
        seq.add(op_id)
        stream.frontier = seq
        self.ops.append((op_id, name, stream.name))
        for t in writes:
            self.writer[self._key(t)] = op_id
        return op_id


def _make_fake_cuda(dag):
    """A ``torch.cuda`` shim whose streams/events feed the DAG."""

    class _CudaShim:
        @staticmethod
        def current_stream(device=None):
            return dag.current_stream()

        @staticmethod
        def Event():
            return _FakeEvent(dag)

        @staticmethod
        def stream(stream):
            return _FakeStreamCtx(dag, stream)

        @staticmethod
        def is_current_stream_capturing():
            return False

    return _CudaShim


class _TorchShim:
    """Forward everything to real torch; swap out ``cuda`` for the fake."""

    def __init__(self, cuda):
        self._cuda = cuda

    @property
    def cuda(self):
        return self._cuda

    def __getattr__(self, name):
        return getattr(torch, name)


# ---------------------------------------------------------------------------
# AST extraction of the real indexer pieces
# ---------------------------------------------------------------------------
def _extract_indexer(dag):
    tree = ast.parse(INDEXER_SRC)
    want_classes = {"_PendingIndexerTopk", "_IndexerScoreState"}
    want_funcs = {
        "_indexer_tail_stream_enabled",
        "_get_indexer_tail_stream",
    }
    want_methods = {
        "forward",
        "forward_prefill_tail_on_stream",
        "_prefill_score_prefix",
        "_prefill_score_tail",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_funcs:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in want_classes:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "IndexerFP8":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in want_methods:
                    body.append(m)
    names = {n.name for n in body}
    assert (
        want_funcs | want_classes | want_methods
    ) == names, f"missing: {(want_funcs | want_classes | want_methods) - names}"
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))

    torch_shim = _TorchShim(_make_fake_cuda(dag))

    def _record_function_range(name):
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Ctx()

    env = {
        "torch": torch_shim,
        "os": os,
        "F": torch.nn.functional,
        "Optional": Optional,
        "Any": Any,
        "Dict": Dict,
        "NamedTuple": NamedTuple,
        "record_function_range": lambda name: _record_function_range(name),
        "INDEXER_HEAD_DIM": 4,
        "_INDEXER_TAIL_STREAM_FLAG": "DSV4_FP8_INDEXER_TAIL_STREAM",
        "_INDEXER_TAIL_STREAMS": {},
        "_fp8_prefill_score_chunk_rows": lambda: 0,
        "_as_bf16_contig": lambda t: t.contiguous(),
        "_flat_1d": lambda t: t.reshape(-1).contiguous(),
        "has_fp8_mqa_logits": lambda device=None: True,
        "indexer_q_rope_fp8_quant_fold": None,  # replaced per-fixture
        "fp8_mqa_indexer_score": None,  # replaced per-fixture
        "_run_prefill_topk": None,  # replaced per-fixture
        "_PendingIndexerTopk": None,  # filled by exec
    }
    exec(compile(mod, str(INDEXER_PATH), "exec"), env)
    return env, torch_shim


# ---------------------------------------------------------------------------
# Stub indexer + deterministic CPU "kernels"
# ---------------------------------------------------------------------------
class _StubCompressor:
    def __init__(self, dag):
        self.freqs_cis = None
        self._dag = dag

    def __call__(self, x, sp, meta=None, workspace=None):
        self._dag.op("nested_compressor", reads=[x], writes=[])


class _StubIndexer:
    """Carries exactly what the extracted indexer methods read off ``self``."""

    def __init__(self, dag, *, pool_bound=True, T=6, M=5, K=3):
        self._dag = dag
        self.index_topk = K
        self.compress_ratio = 4
        self.n_heads = 2
        self.head_dim = 4
        self.rope_head_dim = 2
        self._T = T
        self._M = M
        self._kv_block_table = (
            torch.zeros((1, 2), dtype=torch.int32) if pool_bound else None
        )
        self._kv_pool_view = (
            torch.zeros((2, 2, 132), dtype=torch.uint8) if pool_bound else None
        )
        self._kv_eb = 2 if pool_bound else 0
        self.freqs_cis = torch.zeros((64, 8), dtype=torch.float32)
        self.compressor = _StubCompressor(dag)
        self.weights_proj = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 100.0

    # --- helpers used by the extracted prefix ---
    def _propagate_pool_to_nested(self):
        return None

    def _clear_nested_pool(self):
        return None

    def _discard_prefill_k_cache_gather(self, pending):
        assert pending is None

    def _wait_prefill_k_cache_gather(self, pending):
        raise AssertionError("no pending gather in the unsharded fixture")

    def _compute_indexer_q(self, qr, freqs, batched_rope=False, apply_rope=True):
        # Deterministic stand-in: [M, n_heads*head_dim]
        M = int(qr.shape[0])
        out = (qr.sum(dim=-1, keepdim=True) + 1.0).repeat(
            1, self.n_heads * self.head_dim
        )
        self._dag.op("compute_q", reads=[qr], writes=[out])
        return out

    def _gather_prefill_k_cache(
        self, attention_inputs, k_quant_flat, k_scale_buf, **kw
    ):
        # The vendored gather writes both buffers from the pool: model it as a
        # deterministic write so the tail reads real bytes. (fill_ is safe for
        # the fp8/uint8 CPU fixtures; arange+copy_ would trip CPU dtype gaps.)
        k_quant_flat.fill_(0.25)
        k_scale_buf.fill_(1)
        self._dag.op("gather_k", reads=[], writes=[k_quant_flat, k_scale_buf])
        return None

    # --- extracted real methods, bound per-fixture in the test ---


def _bind_indexer(stub, env):
    for name in (
        "forward",
        "forward_prefill_tail_on_stream",
        "_prefill_score_prefix",
        "_prefill_score_tail",
    ):
        setattr(stub, name, types.MethodType(env[name], stub))
    return stub


def _wire_heavy_ops(env, dag):
    """Deterministic CPU stand-ins for the quant fold, DeepGEMM score, topk."""

    def quant_fold(q, w, freqs, rope_dim):
        # q: [1, M, H*D] -> q_fp8 [1, M, H, D]; w: [1, M, H'] -> w_fold
        qf = (q * 0.5).reshape(1, q.shape[1], 2, 4)
        wf = w.squeeze(0) if w.dim() == 3 else w
        dag.op("quant_fold", reads=[q, w], writes=[qf, wf])
        return qf, wf

    def score(q_score, w_score, k_quant, k_scale, ks, ke, clean_logits=False):
        rows = q_score.shape[0]
        T = k_quant.shape[0]
        base = torch.arange(rows * T, dtype=torch.float32).reshape(rows, T) / 11.0
        # Touch every input deterministically (dtype-safe reductions only).
        base += float(q_score.to(torch.float32).sum()) * 0.0
        base += float(w_score.to(torch.float32).sum()) * 0.0
        base += float(k_quant.to(torch.float32).sum()) * 0.001
        base += float(k_scale.to(torch.float32).sum()) * 0.0
        base += float(ks.to(torch.float32).sum()) * 0.0
        base += float(ke.to(torch.float32).sum()) * 0.0
        dag.op(
            "score",
            reads=[q_score, w_score, k_quant, k_scale],
            writes=[base],
        )
        return base

    def topk(logits, row_starts, row_ends, out, topk_k, compress_ratio):
        dag.op("topk", reads=[logits, row_starts, row_ends], writes=[out])
        k = min(int(topk_k), logits.shape[1])
        vals = logits.topk(k, dim=-1).indices.to(torch.int32)
        out[:, :k].copy_(vals)
        if k < out.shape[1]:
            out[:, k:].fill_(-1)

    env["indexer_q_rope_fp8_quant_fold"] = quant_fold
    env["fp8_mqa_indexer_score"] = score
    env["_run_prefill_topk"] = topk


class _Meta(NamedTuple):
    M: int
    T: int
    sp_int: int
    ks: torch.Tensor
    ke: torch.Tensor
    freqs_cis_slice: torch.Tensor
    compressor_meta: Any = None


def _make_meta(M, T):
    return _Meta(
        M=M,
        T=T,
        sp_int=0,
        ks=torch.zeros(M, dtype=torch.int32),
        ke=torch.full((M,), T, dtype=torch.int32),
        freqs_cis_slice=torch.zeros((M, 8), dtype=torch.float32),
    )


class IndexerTailStreamValueAndOrder(unittest.TestCase):
    def _run_pair(self, M, T, K, flag):
        """Legacy serialized forward vs the side-stream tail, same fixture."""
        dag = _DAG()
        env, torch_shim = _extract_indexer(dag)
        _wire_heavy_ops(env, dag)
        x = torch.arange(M * 12, dtype=torch.float32).reshape(M, 12) / 7.0
        qr = torch.arange(M * 6, dtype=torch.float32).reshape(M, 6) / 5.0
        meta = _make_meta(M, T)
        stub = _bind_indexer(_StubIndexer(dag, pool_bound=True, T=T, M=M, K=K), env)
        with patch.dict(os.environ, {"DSV4_FP8_INDEXER_TAIL_STREAM": flag}):
            legacy = stub.forward(x, qr, meta, workspace=None)
        legacy_ops = [n for _, n, _ in dag.ops]
        self.assertEqual(dag.errors, [])

        # Candidate: side-stream tail + resolve on the main stream.
        dag2 = _DAG()
        env2, torch_shim2 = _extract_indexer(dag2)
        _wire_heavy_ops(env2, dag2)
        stub2 = _bind_indexer(_StubIndexer(dag2, pool_bound=True, T=T, M=M, K=K), env2)
        tail_stream = dag2.stream("indexer_tail")
        # The call-site gates on the env flag via _indexer_tail_stream_enabled;
        # drive the new method directly (the call-site test below covers the
        # gate).
        pending = stub2.forward_prefill_tail_on_stream(
            x, qr, meta, workspace=None, tail_stream=tail_stream
        )
        self.assertIsInstance(pending, env2["_PendingIndexerTopk"])
        # Consumer side: some main-stream prep, then the resolve + read.
        prep = torch.zeros(2, 2)
        dag2.op("main_prep", reads=[], writes=[prep])
        got = pending.resolve(dag2.current_stream())
        dag2.op("combine_topk", reads=[got], writes=[])
        self.assertEqual(dag2.errors, [])
        self.assertTrue(torch.equal(legacy, got))
        self.assertEqual(legacy.dtype, torch.int32)
        # Same kernel sequence for the indexer chain (the tail's ops are
        # identical, only placed on the side stream); the trailing
        # main_prep/combine_topk are the consumer-side ops added by this test.
        cand_ops = [n for _, n, _ in dag2.ops][: len(legacy_ops)]
        self.assertEqual(legacy_ops, cand_ops)
        # The tail ran on the side stream; the prefix on main.
        tail_ops = [(n, s) for _, n, s in dag2.ops if n in ("score", "topk")]
        self.assertTrue(tail_ops and all(s == "indexer_tail" for _, s in tail_ops))
        prefix_ops = [
            (n, s)
            for _, n, s in dag2.ops
            if n in ("compute_q", "nested_compressor", "quant_fold", "gather_k")
        ]
        self.assertTrue(prefix_ops and all(s == "main" for _, s in prefix_ops))
        # Event structure: one wait on the tail stream (the in-fence), one
        # wait on the main stream (the join), records on both.
        waits = [(k, s) for k, s, _ in dag2.log if k in ("wait_event",)]
        self.assertIn(("wait_event", "indexer_tail"), waits)
        self.assertIn(("wait_event", "main"), waits)
        return legacy, got

    def test_byte_identity_matrix(self):
        for M, T, K in ((5, 6, 3), (1, 1, 1), (16, 40, 8), (7, 3, 4)):
            with self.subTest(M=M, T=T, K=K):
                self._run_pair(M, T, K, "1")

    def test_pending_pins_prefix_products_until_resolve(self):
        dag = _DAG()
        env, _ = _extract_indexer(dag)
        _wire_heavy_ops(env, dag)
        M, T, K = 5, 6, 3
        x = torch.arange(M * 12, dtype=torch.float32).reshape(M, 12)
        qr = torch.arange(M * 6, dtype=torch.float32).reshape(M, 6)
        stub = _bind_indexer(_StubIndexer(dag, pool_bound=True, T=T, M=M, K=K), env)
        pending = stub.forward_prefill_tail_on_stream(
            x, qr, _make_meta(M, T), workspace=None, tail_stream=dag.stream("tail")
        )
        # The handle keeps the prefix products alive: dropping every other
        # reference must not free them for reuse before resolve.
        state = pending.state
        self.assertIsNotNone(state.q_fp8)
        self.assertIsNotNone(state.w_fold)
        self.assertIsNotNone(state.k_quant_flat)
        self.assertIsNotNone(state.k_scale_buf)
        out = pending.resolve(dag.current_stream())
        self.assertEqual(out.shape, (M, K))

    def test_dropped_consumer_wait_mutant_caught(self):
        """Mutant: resolve() that never waits the done event must be caught."""
        dag = _DAG()
        env, _ = _extract_indexer(dag)
        _wire_heavy_ops(env, dag)
        M, T, K = 5, 6, 3
        x = torch.arange(M * 12, dtype=torch.float32).reshape(M, 12)
        qr = torch.arange(M * 6, dtype=torch.float32).reshape(M, 6)
        stub = _bind_indexer(_StubIndexer(dag, pool_bound=True, T=T, M=M, K=K), env)
        pending = stub.forward_prefill_tail_on_stream(
            x, qr, _make_meta(M, T), workspace=None, tail_stream=dag.stream("tail")
        )

        # Mutant resolve: no wait_event.
        out = pending.out_buf
        dag.op("combine_topk", reads=[out], writes=[])
        self.assertTrue(
            dag.errors,
            "the DAG model must flag a consumer reading the tail output "
            "without the join wait",
        )

    def test_dropped_producer_fence_mutant_caught(self):
        """Mutant: tail stream that never waits the in-fence must be caught."""
        dag = _DAG()
        env, _ = _extract_indexer(dag)
        _wire_heavy_ops(env, dag)
        M, T, K = 5, 6, 3
        x = torch.arange(M * 12, dtype=torch.float32).reshape(M, 12)
        qr = torch.arange(M * 6, dtype=torch.float32).reshape(M, 6)
        stub = _bind_indexer(_StubIndexer(dag, pool_bound=True, T=T, M=M, K=K), env)
        state = stub._prefill_score_prefix(x, qr, _make_meta(M, T), workspace=None)
        tail_stream = dag.stream("tail")
        # Mutant: skip tail_stream.wait_event(in_event) — tail reads prefix
        # products without a happens-before edge.
        fake_cuda = _make_fake_cuda(dag)
        with fake_cuda.stream(tail_stream):
            stub._prefill_score_tail(state, _make_meta(M, T))
        self.assertTrue(
            dag.errors,
            "the DAG model must flag the tail reading prefix products without "
            "the producer fence",
        )

    def test_early_exits_return_plain_tensors(self):
        dag = _DAG()
        env, _ = _extract_indexer(dag)
        _wire_heavy_ops(env, dag)
        M, K = 4, 3
        x = torch.arange(M * 12, dtype=torch.float32).reshape(M, 12)
        qr = torch.arange(M * 6, dtype=torch.float32).reshape(M, 6)
        # Warmup: pool unbound.
        stub = _bind_indexer(_StubIndexer(dag, pool_bound=False, T=0, M=M, K=K), env)
        out = stub.forward_prefill_tail_on_stream(
            x, qr, _make_meta(M, 0), workspace=None, tail_stream=dag.stream("tail")
        )
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (M, 0))
        # No events, no fences, no ops — the early return never engages.
        self.assertEqual(dag.log, [])
        # Cold start: pool bound but T == 0.
        stub2 = _bind_indexer(_StubIndexer(dag, pool_bound=True, T=0, M=M, K=K), env)
        out2 = stub2.forward_prefill_tail_on_stream(
            x, qr, _make_meta(M, 0), workspace=None, tail_stream=dag.stream("tail2")
        )
        self.assertIsInstance(out2, torch.Tensor)
        self.assertEqual(out2.shape, (M, 0))
        self.assertEqual(dag.errors, [])


class IndexerTailFlagAndCallSite(unittest.TestCase):
    def test_flag_fail_closed_parse(self):
        env, _ = _extract_indexer(_DAG())
        parser = env["_indexer_tail_stream_enabled"]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_FP8_INDEXER_TAIL_STREAM", None)
            self.assertTrue(parser())  # default ON
        with patch.dict(os.environ, {"DSV4_FP8_INDEXER_TAIL_STREAM": "0"}):
            self.assertFalse(parser())
        with patch.dict(os.environ, {"DSV4_FP8_INDEXER_TAIL_STREAM": "1"}):
            self.assertTrue(parser())
        with patch.dict(os.environ, {"DSV4_FP8_INDEXER_TAIL_STREAM": "yes"}):
            with self.assertRaises(ValueError):
                parser()


# ---------------------------------------------------------------------------
# Call-site wiring: the REAL ``_forward_prefill_csa`` + ``_forward_prefill_compressed``
# ---------------------------------------------------------------------------
def _extract_callsite(dag):
    tree = ast.parse(ATTN_SRC)
    want = {"_forward_prefill_csa", "_forward_prefill_compressed"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AttentionFP8":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in want:
                    body.append(m)
    names = {n.name for n in body}
    assert names == want, f"missing: {want - names}"
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    torch_shim = _TorchShim(_make_fake_cuda(dag))

    env = {
        "torch": torch_shim,
        "Optional": Optional,
        "Any": Any,
        "record_function_range": lambda name: _null_ctx(),
        "_indexer_tail_stream_enabled": None,  # per-fixture
        "_get_indexer_tail_stream": None,  # per-fixture
        "_PendingIndexerTopk": None,  # per-fixture
        "IndexerFP8": None,  # per-fixture
    }
    exec(compile(mod, str(ATTN_PATH), "exec"), env)
    return env, torch_shim


def _null_ctx():
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Ctx()


def _install_indexer_stub_module(indexer_cls):
    """``_forward_prefill_csa`` does a function-level ``from ... import
    IndexerFP8``; give that import a stub package chain resolving to the
    fixture's indexer class."""
    chain = [
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
        "rtp_llm.models_py.modules.dsv4.fp8",
        "rtp_llm.models_py.modules.dsv4.fp8.indexer",
    ]
    saved = {}
    for name in chain:
        saved[name] = sys.modules.get(name)
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    sys.modules[chain[-1]].IndexerFP8 = indexer_cls
    return saved


def _restore_modules(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


class _CallSiteIndexer:
    """Records which entry point the CSA path used and returns a marker."""

    def __init__(self, pending_marker, plain_marker):
        self._pending_marker = pending_marker
        self._plain_marker = plain_marker
        self.calls = []

    def forward(self, x, qr, meta, *, workspace):
        self.calls.append(("forward", workspace))
        return self._plain_marker

    # nn.Module-style: the production call site does ``self.indexer(...)``.
    def __call__(self, x, qr, meta, *, workspace):
        return self.forward(x, qr, meta, workspace=workspace)

    def forward_prefill_tail_on_stream(self, x, qr, meta, *, workspace, tail_stream):
        self.calls.append(("tail_on_stream", workspace, tail_stream))
        return self._pending_marker


class _CallSiteAttn:
    compress_ratio = 4

    def __init__(self, env, indexer, capturing):
        self.indexer = indexer
        self._capturing = capturing
        self.compressor = None  # compressed path records the write
        self._resolved_at = None
        self._env = env

    # the real extracted methods are bound in the test
    def _materialize_prefill_q(self, qkv, common):
        return qkv

    def _attn_via_workspace(self, qkv, common, workspace_meta, cmp_topk_runtime):
        return ("workspace_out", cmp_topk_runtime)

    def _attn_fp8_swa_via_kv_full(self, qkv, common):
        return ("warmup_out",)


class _CallSiteCompressor:
    def __init__(self):
        self.calls = []

    def __call__(self, x, sp, meta=None, workspace=None):
        self.calls.append(("compressor", meta))


def _run_callsite(dag, *, flag, capturing):
    """Drive the real _forward_prefill_csa; return the recorded facts."""
    env, torch_shim = _extract_callsite(dag)
    pending_marker = object()
    plain_marker = torch.zeros(2, 2, dtype=torch.int32)
    indexer = _CallSiteIndexer(pending_marker, plain_marker)

    # env wiring
    real_pending_cls = _extract_indexer(_DAG())[0]["_PendingIndexerTopk"]
    env["_indexer_tail_stream_enabled"] = lambda: flag
    tail_stream = dag.stream("tail")
    env["_get_indexer_tail_stream"] = lambda device: tail_stream
    env["_PendingIndexerTopk"] = real_pending_cls
    env["IndexerFP8"] = _CallSiteIndexer

    attn = _CallSiteAttn(env, indexer, capturing)
    attn.compressor = _CallSiteCompressor()
    attn._forward_prefill_csa = types.MethodType(env["_forward_prefill_csa"], attn)
    attn._forward_prefill_compressed = types.MethodType(
        env["_forward_prefill_compressed"], attn
    )

    # capture state for the call-site gate
    cap = capturing
    orig_cuda = torch_shim.cuda

    class _CapCuda(_make_fake_cuda(dag)):
        @staticmethod
        def is_current_stream_capturing():
            return cap

    torch_shim._cuda = _CapCuda

    class _Common:
        csa_meta = types.SimpleNamespace(
            indexer_meta=object(), compressor_meta=object(), workspace_meta=object()
        )
        workspace = object()
        sp_int = 0

    class _FakeX:
        # The production gate reads ``x.is_cuda``; a CPU test tensor would
        # always take the fallback, so the fixture fakes the CUDA tag.
        is_cuda = True
        device = "cuda:0"

    saved = _install_indexer_stub_module(_CallSiteIndexer)
    try:
        x = _FakeX()
        qkv = types.SimpleNamespace(qr=torch.zeros(3, 6))
        out = attn._forward_prefill_csa(x, qkv, _Common())
    finally:
        _restore_modules(saved)
    return indexer, out, pending_marker, plain_marker


class CallSiteWiring(unittest.TestCase):
    def test_flag_on_engages_side_stream_and_resolves(self):
        dag = _DAG()
        indexer, out, pending_marker, plain_marker = _run_callsite(
            dag, flag=True, capturing=False
        )
        self.assertEqual(indexer.calls[0][0], "tail_on_stream")
        # The pending marker is NOT a tensor; _forward_prefill_compressed must
        # have tried to resolve it — since our marker is a plain object the
        # isinstance check must simply pass it through. (The real pending is
        # exercised in the value tests above.)
        self.assertEqual(out[0], "workspace_out")

    def test_flag_off_uses_plain_forward(self):
        dag = _DAG()
        indexer, out, pending_marker, plain_marker = _run_callsite(
            dag, flag=False, capturing=False
        )
        self.assertEqual(indexer.calls[0][0], "forward")
        self.assertEqual(out[0], "workspace_out")
        self.assertIs(out[1], plain_marker)

    def test_capture_active_falls_back_to_serialized(self):
        dag = _DAG()
        indexer, out, pending_marker, plain_marker = _run_callsite(
            dag, flag=True, capturing=True
        )
        self.assertEqual(indexer.calls[0][0], "forward")
        self.assertIs(out[1], plain_marker)

    def test_real_pending_resolved_before_workspace_consumer(self):
        """The pending object must be event-joined before ``_attn_via_workspace``."""
        dag = _DAG()
        env, torch_shim = _extract_callsite(dag)
        indexer_env, _ = _extract_indexer(dag)
        pending_cls = indexer_env["_PendingIndexerTopk"]
        state_cls = indexer_env["_IndexerScoreState"]

        tail_stream = dag.stream("tail")
        out_buf = torch.arange(6, dtype=torch.int32).reshape(2, 3)
        # Model the tail having "run" on the side stream: write + done event.
        dag.push_stream(tail_stream)
        dag.op("topk", reads=[], writes=[out_buf])
        done = _FakeEvent(dag)
        done.record(tail_stream)
        dag.pop_stream()
        state = state_cls(
            q_fp8=torch.zeros(1),
            w_fold=torch.zeros(1),
            k_quant_flat=torch.zeros(1),
            k_scale_buf=torch.zeros(1),
            out_shape=(2, 3),
        )
        pending = pending_cls(out_buf, done, tail_stream, state)

        indexer = _CallSiteIndexer(pending, plain_marker=torch.zeros(2, 2))
        env["_indexer_tail_stream_enabled"] = lambda: True
        env["_get_indexer_tail_stream"] = lambda device: tail_stream
        env["_PendingIndexerTopk"] = pending_cls
        env["IndexerFP8"] = _CallSiteIndexer

        attn = _CallSiteAttn(env, indexer, capturing=False)
        attn.compressor = _CallSiteCompressor()
        attn._forward_prefill_csa = types.MethodType(env["_forward_prefill_csa"], attn)
        attn._forward_prefill_compressed = types.MethodType(
            env["_forward_prefill_compressed"], attn
        )

        class _FakeX:
            is_cuda = True
            device = "cuda:0"

        class _Common:
            csa_meta = types.SimpleNamespace(
                indexer_meta=object(), compressor_meta=object(), workspace_meta=object()
            )
            workspace = object()
            sp_int = 0

        saved = _install_indexer_stub_module(_CallSiteIndexer)
        try:
            out = attn._forward_prefill_csa(
                _FakeX(), types.SimpleNamespace(qr=torch.zeros(3, 6)), _Common()
            )
        finally:
            _restore_modules(saved)
        # The workspace consumer received the resolved plain tensor.
        self.assertIs(out[1], out_buf)
        self.assertEqual(dag.errors, [])
        waits = [(k, s) for k, s, _ in dag.log if k == "wait_event"]
        self.assertIn(("wait_event", "main"), waits)


def setUpModule():
    run_path(str(HERE.parents[1] / "test" / "cpu_test_utils.py"))[
        "isolate_cpu_test_module"
    ]()


if __name__ == "__main__":
    unittest.main()
