"""Execute the changed forward/MoE/count callers with CPU tensor/collective leaves.

No CUDA, model weights, NCCL, JIT or numerical qualification is implied.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only: hide CUDA"
torch.cuda._lazy_init = lambda: (_ for _ in ()).throw(AssertionError("CUDA forbidden"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "rtp_llm.models_py.modules.dsv4.moe.forward_ep_plan"
spec = importlib.util.spec_from_file_location(NAME, ROOT / "moe/forward_ep_plan.py")
plan = importlib.util.module_from_spec(spec)
sys.modules[NAME] = plan
spec.loader.exec_module(plan)


def extract(path, names, env, cls=None):
    tree = ast.parse(path.read_text())
    body = (
        tree.body
        if cls is None
        else next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    )
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(mod, str(path), "exec"), env)
    return {n.name: env[n.name] for n in nodes}


def combine(routed, shared, dtype, out=None):
    result = routed if shared is None else routed + shared
    if out is None:
        return result.to(dtype)
    out.copy_(result)
    return out


rt = types.SimpleNamespace(ENABLED=False, should_record_layer=lambda _: False)
# Only the debug import within the actual MoE.forward is doubled.
package = types.ModuleType("rtp_llm.models_py.modules.dsv4")
package.__path__ = [str(ROOT)]
package._record_tensor = rt
sys.modules[package.__name__] = package
for name in ("rtp_llm", "rtp_llm.models_py", "rtp_llm.models_py.modules"):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod

STRATEGY_ENV = dict(
    torch=torch,
    current_scope=plan.current_scope,
    nullcontext=nullcontext,
    _count_fuse_enabled=lambda: True,
)
strategy_methods = extract(
    ROOT / "moe/strategies/nccl_ep_mxfp8.py",
    {
        "synchronized_chunk_extent",
        "_counts_for_forward",
        "_take_pending_counts",
        "forward_subchunk_scope",
    },
    STRATEGY_ENV,
    "NcclEpMxfp8Strategy",
)

strategy_methods.update(
    extract(
        ROOT / "moe/strategies/base.py",
        {"prepare_dispatch", "run_dispatch_prepared"},
        {},
        "RoutedExpertsStrategy",
    )
)

MOE_ENV = dict(
    torch=torch,
    os=os,
    logging=types.SimpleNamespace(info=lambda *a, **k: None),
    record_function_range=lambda *a: nullcontext(),
    chunked_moe_enabled=lambda: True,
    _CHUNKED_MOE_LOGGED=False,
    combine_routed_and_shared=combine,
    _get_or_create_final_out=lambda n, d, t, dev: torch.empty(
        (n, d), dtype=t, device=dev
    ),
)
moe_methods = extract(
    ROOT / "moe/moe_layer.py",
    {"forward", "_should_chunk", "_run_chunk", "_forward_chunked"},
    MOE_ENV,
    "MoE",
)


class Strategy:
    name = "fork_nccl_mxfp8"
    requires_synchronized_chunk_schedule = True

    def __init__(self, ctx, counts, ledger):
        self.cfg = types.SimpleNamespace(
            stage_context=ctx, ep_size=4, ep_rank=ctx.group_rank
        )
        self.counts = tuple(counts)
        self.ledger = ledger
        self._pending_counts = None
        self.fail = False

    def _stage(self):
        c = self.cfg.stage_context
        return c.process_group, c.group_size, c.group_rank

    def _gather_counts(self, n, group, world, device):
        assert n == self.counts[self.cfg.ep_rank]
        self.ledger.append(("gather", self.cfg.stage_context.pp_rank, n))
        return list(self.counts)

    _exchange_counts = _gather_counts

    def __call__(self, x, weights, indices):
        pending = self._take_pending_counts()
        group, world, rank = self._stage()
        counts = self._counts_for_forward(x.shape[0], group, world, x.device, pending)
        self.ledger.append(("payload", tuple(counts), rank))
        if self.fail:
            raise RuntimeError("injected routed failure")
        return x.clone()


for name, method in strategy_methods.items():
    setattr(Strategy, name, method)


class Shared:
    def start(self, module, x):
        self.value = torch.zeros_like(x)

    def finish(self):
        return self.value


def make_model(counts=(8, 8, 8, 8), rank=0, stage=0, layers=3, width=12288, fast=False):
    ledger = []
    ctx = types.SimpleNamespace(
        process_group=object(),
        world_ranks=tuple(range(4 * stage, 4 * stage + 4)),
        group_rank=rank,
        group_size=4,
        pp_rank=stage,
        generation=7,
    )
    blocks = []
    for i in range(layers):
        strategy = Strategy(ctx, counts, ledger)
        moe = types.SimpleNamespace(
            _strategy=strategy,
            dim=2,
            max_tokens_per_rank=width,
            layer_id=i,
            _is_decode_role=False,
            _gate_pack_static=False,
            _routed_includes_shared=False,
            shared_experts=None,
            _shared_executor=Shared(),
            gate=lambda x, ids: (
                torch.ones((len(x), 1)),
                torch.zeros((len(x), 1), dtype=torch.int64),
            ),
        )
        for name, method in moe_methods.items():
            setattr(moe, name, types.MethodType(method, moe))

        class Block:
            def __init__(self, ffn):
                self.ffn = ffn

            def __call__(self, h, ids, *a, **kw):
                ledger.append(("normal", id(plan.current_scope())))
                return self.ffn.forward(h, ids)

            def fast_call(self, h, ids, *a, **kw):
                ledger.append(("fast", id(plan.current_scope())))
                return self.ffn.forward(h, ids)

        blocks.append(Block(moe))
    n = counts[rank]
    cp = types.SimpleNamespace(
        cp_size=4, cp_rank=rank, chunk_length=n, global_positions=torch.arange(n)
    )
    model = types.SimpleNamespace(
        layers=blocks,
        commit_only=False,
        _cp_info=object(),
        _cp_size=4,
        _cp_rank=rank,
        _kv_cache_sharded=False,
        _propagate_cp_ctx=lambda x: None,
        cp=cp,
        fast=fast,
        embed_full=lambda ids: ids.float().unsqueeze(-1).expand(-1, 2),
        hc_mult=1,
        capture_aux_hidden_layer_ids=(),
        fp8_kv_cache=False,
        _mtp_hidden_buffer=None,
        norm=None,
    )
    return model, cp, ledger


def coordinator(model, cp, rows, device):
    scope = plan.make_prefill_scope(model, cp, rows, device, requested=plan.enabled())
    return plan.activate_scope(scope)


FWD_ENV = dict(
    torch=torch,
    os=os,
    nullcontext=nullcontext,
    _FWD_STATS=False,
    _FWD_GPU=False,
    _FWD_PROFILE=False,
    _rt=rt,
    build_cp_context_for_forward=lambda *a, **kw: FWD_ENV["model"].cp,
    _prefill_fast_path_layer_calls=lambda v: tuple(b.fast_call for b in v.layers),
    _prefill_fast_path_enabled=lambda v, *a: v.fast,
    _profiler=types.SimpleNamespace(disable_record_function_ranges=nullcontext),
    prefill_forward_scope=coordinator,
)
forward_layers = extract(ROOT / "prefill/forward.py", {"forward_layers"}, FWD_ENV)[
    "forward_layers"
]


def run_model(model):
    FWD_ENV["model"] = model
    n = model.cp.chunk_length
    ids = torch.arange(n)
    result = forward_layers(model, None, ids, ids, torch.tensor([0, n]), None)
    torch.testing.assert_close(result, ids.float().reshape(n, 1, 1).expand(n, 1, 2))
    return result


class ForwardPlanTests(unittest.TestCase):
    def test_real_default_prepared_dispatch_contract(self):
        strategy = make_model()[0].layers[0].ffn._strategy
        self.assertIsNone(strategy.prepare_dispatch(None, None, None))
        with self.assertRaisesRegex(NotImplementedError, "run_dispatch_prepared"):
            strategy.run_dispatch_prepared({})

    def setUp(self):
        self.env = patch.dict(
            os.environ, {"DSV4_MOE_FORWARD_COUNT_PLAN": "1"}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.assertIsNone(plan.current_scope())

    def test_both_actual_coordinator_chains_all_stage_ranks(self):
        for stage, layers in ((0, 22), (1, 21)):
            for rank in range(4):
                for fast in (False, True):
                    with self.subTest(stage=stage, rank=rank, fast=fast):
                        m, c, log = make_model(
                            rank=rank, stage=stage, layers=layers, fast=fast
                        )
                        run_model(m)
                        self.assertEqual(sum(x[0] == "gather" for x in log), 1)
                        self.assertEqual(sum(x[0] == "payload" for x in log), layers)
                        self.assertEqual(
                            sum(x[0] == ("fast" if fast else "normal") for x in log),
                            layers,
                        )
                        self.assertIsNone(plan.current_scope())

    def test_flag_off_legacy_count_fuse_unchanged(self):
        os.environ["DSV4_MOE_FORWARD_COUNT_PLAN"] = "0"
        m, c, log = make_model(layers=22)
        run_model(m)
        self.assertEqual(sum(x[0] == "gather" for x in log), 22)

    def test_actual_chunked_caller_ragged_zero_peer_and_tails(self):
        for counts, width in [
            ((0, 3, 1, 5), 2),
            ((904,) * 4, 12288),
            ((1024,) * 4, 12288),
            ((0, 0, 0, 0), 12288),
            ((3, 3, 3, 3), 2),
        ]:
            for rank in range(4):
                m, c, log = make_model(counts=counts, rank=rank, width=width, layers=2)
                run_model(m)
                self.assertEqual(sum(x[0] == "gather" for x in log), 1)
                expected = (
                    [counts]
                    if max(counts) <= width
                    else [
                        tuple(max(0, min(width, n - s)) for n in counts)
                        for s in range(0, max(counts), width)
                    ]
                )
                self.assertEqual([x[1] for x in log if x[0] == "payload"], expected * 2)

    def test_next_forward_same_local_different_peers_regathers(self):
        m, c, log = make_model(counts=(3, 3, 3, 3), layers=2)
        run_model(m)
        for layer in m.layers:
            layer.ffn._strategy.counts = (3, 2, 1, 0)
        run_model(m)
        self.assertEqual(sum(x[0] == "gather" for x in log), 2)
        self.assertEqual(
            [x[1] for x in log if x[0] == "payload"][-2:], [(3, 2, 1, 0)] * 2
        )

    def test_layer_exception_scope_cleanup_and_recovery(self):
        m, c, log = make_model()
        m.layers[1].ffn._strategy.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            run_model(m)
        self.assertIsNone(plan.current_scope())
        m.layers[1].ffn._strategy.fail = False
        run_model(m)
        self.assertEqual(sum(x[0] == "gather" for x in log), 2)

    def test_nested_forward_masks_and_restores_scope(self):
        m, c, _ = make_model()
        a = plan.make_prefill_scope(m, c, 8, torch.device("cpu"), requested=True)
        other, cc, _ = make_model(stage=1)
        with plan.activate_scope(a):
            outer = plan.current_scope()
            run_model(other)
            self.assertIs(plan.current_scope(), outer)
            with plan.activate_scope(None):
                self.assertIsNone(plan.current_scope())
            self.assertIs(plan.current_scope(), outer)
        self.assertIsNone(plan.current_scope())
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            with plan.activate_scope(a):
                pass
        self.assertIsNone(plan.current_scope())

    def test_wrong_strategy_group_device_and_rows_fail_without_collective(self):
        for defect in ("foreign", "group", "device", "rows", "rank"):
            m, c, log = make_model()
            scope = plan.make_prefill_scope(
                m, c, 8, torch.device("cpu"), requested=True
            )
            st = m.layers[0].ffn._strategy
            group, world, _ = st._stage()
            dev = torch.device("cpu")
            rows = 8
            if defect == "foreign":
                st = Strategy(st.cfg.stage_context, (8,) * 4, log)
            if defect == "group":
                group = object()
            if defect == "device":
                dev = torch.device("cuda:1")
            if defect == "rows":
                rows = 7
            if defect == "rank":
                st.cfg.ep_rank = 3
            with plan.activate_scope(scope), self.assertRaises(RuntimeError):
                scope.get_counts(
                    st,
                    rows,
                    group,
                    world,
                    dev,
                    lambda: (_ for _ in ()).throw(AssertionError("no gather")),
                )
            self.assertFalse(log)

    def test_invalid_gather_rejected_and_no_mutable_alias(self):
        for bad in ([8, 8, 8], [8, -1, 8, 8], [8, True, 8, 8], [7, 8, 8, 8]):
            m, c, _ = make_model()
            scope = plan.make_prefill_scope(
                m, c, 8, torch.device("cpu"), requested=True
            )
            st = m.layers[0].ffn._strategy
            with plan.activate_scope(scope), self.assertRaises(RuntimeError):
                scope.get_counts(
                    st, 8, *st._stage()[:2], torch.device("cpu"), lambda: bad
                )
        m, c, _ = make_model()
        scope = plan.make_prefill_scope(m, c, 8, torch.device("cpu"), requested=True)
        st = m.layers[0].ffn._strategy
        counts = [8] * 4
        with plan.activate_scope(scope):
            scope.get_counts(
                st, 8, *st._stage()[:2], torch.device("cpu"), lambda: counts
            )
            counts[1] = 99
            self.assertEqual(scope.plan.physical_rows, (8,) * 4)
            with self.assertRaises(FrozenInstanceError):
                scope.plan.physical_rows = (1,) * 4

    def test_bad_subchunk_identity_and_schedule_rejected(self):
        m, c, _ = make_model(counts=(5,) * 4, width=2)
        scope = plan.make_prefill_scope(m, c, 5, torch.device("cpu"), requested=True)
        st = m.layers[0].ffn._strategy
        with plan.activate_scope(scope):
            st.synchronized_chunk_extent(5, torch.device("cpu"))
            for start, width in ((1, 2), (6, 2), (0, 3), (-1, 2)):
                with self.assertRaises(ValueError):
                    with st.forward_subchunk_scope(start, width, 5):
                        pass
            with st.forward_subchunk_scope(0, 2, 5):
                with self.assertRaises(RuntimeError):
                    st.synchronized_chunk_extent(5, torch.device("cpu"))
                with self.assertRaises(RuntimeError):
                    with st.forward_subchunk_scope(2, 2, 5):
                        pass
            self.assertEqual(
                st._counts_for_forward(5, *st._stage()[:2], torch.device("cpu")),
                [5] * 4,
            )

    def test_unsupported_modes_and_bad_topology(self):
        m, c, _ = make_model()
        for kw in ({"capturing": True}, {"warming": True}):
            self.assertIsNone(
                plan.make_prefill_scope(
                    m, c, 8, torch.device("cpu"), requested=True, **kw
                )
            )
        self.assertIsNone(
            plan.make_prefill_scope(m, None, 8, torch.device("cpu"), requested=True)
        )
        m.layers[0].ffn._is_decode_role = True
        self.assertIsNone(
            plan.make_prefill_scope(m, c, 8, torch.device("cpu"), requested=True)
        )
        m.layers[0].ffn._is_decode_role = False
        m.layers[0].ffn._strategy.cfg.stage_context.world_ranks = (4, 5, 6, 7)
        with self.assertRaises(RuntimeError):
            plan.make_prefill_scope(m, c, 8, torch.device("cpu"), requested=True)

    def test_no_scope_legacy_ragged_and_subchunk(self):
        os.environ["DSV4_MOE_FORWARD_COUNT_PLAN"] = "0"
        for counts in ((0, 3, 1, 5), (0, 0, 0, 0)):
            m, c, log = make_model(counts=counts, rank=0, width=2, layers=2)
            # The legacy gather leaf must produce the actual subchunk count vector.
            # This test checks absent-scope behavior directly, not a fake ragged gather.
            st = m.layers[0].ffn._strategy
            self.assertEqual(
                st.synchronized_chunk_extent(counts[0], torch.device("cpu")),
                max(counts),
            )
            with st.forward_subchunk_scope(0, 2, counts[0]):
                self.assertIsNone(plan.current_scope())

    def test_flag_parser_rejects_typo(self):
        os.environ["DSV4_MOE_FORWARD_COUNT_PLAN"] = "yes"
        with self.assertRaises(ValueError):
            plan.enabled()


if __name__ == "__main__":
    unittest.main(verbosity=2)
