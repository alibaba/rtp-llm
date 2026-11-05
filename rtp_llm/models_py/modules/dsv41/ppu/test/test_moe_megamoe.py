"""CPU contracts for the opt-in native PPU MegaMoE adapter.

Only device validation, CUDA lifecycle calls, distributed collectives and the
four external native APIs are replaced. Production forward, route padding,
weight-cache selection and capacity calculations execute unchanged. The native
double uses analytic experts f_e(x)=(e+1)*x to expose extra route weighting;
these tests do not establish MXFP4 kernel accuracy or real EP communication.
"""

import gc
import os
import sys
import types
import unittest
import weakref
from contextlib import ExitStack, contextmanager
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.ppu import moe as moe_module
from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import (
    MOE_BACKEND_ENV,
    MOE_PRECISION_ENV,
    moe_backend,
    moe_kernel_name,
    moe_precision,
)


@contextmanager
def _selection(backend=None, precision=None):
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(MOE_BACKEND_ENV, None)
        os.environ.pop(MOE_PRECISION_ENV, None)
        if backend is not None:
            os.environ[MOE_BACKEND_ENV] = backend
        if precision is not None:
            os.environ[MOE_PRECISION_ENV] = precision
        yield


class _Group:
    def __init__(self, size=2, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _weights(seed=901):
    generator = torch.Generator().manual_seed(seed)
    w13 = torch.randint(256, (4, 256, 256), generator=generator, dtype=torch.uint8)
    w2 = torch.randint(256, (4, 512, 64), generator=generator, dtype=torch.uint8)
    s13 = torch.randint(112, 118, (4, 256, 16), generator=generator, dtype=torch.uint8)
    s2 = torch.randint(130, 135, (4, 512, 4), generator=generator, dtype=torch.uint8)
    return (
        (w13.view(torch.int8), s13.view(torch.float8_e8m0fnu)),
        (w2.view(torch.int8), s2.view(torch.int8)),
    )


def _inputs(rows):
    x = ((torch.arange(rows * 512).reshape(rows, 512) % 11 - 5) / 8).bfloat16()
    ids = torch.tensor([6, 2, 4, 1, 7, 0], dtype=torch.int32).expand(rows, 6)
    weights = torch.tensor([0.25, 0.0, 0.375, 0.5, 0.125, 0.25]).expand(rows, 6)
    return x, ids, weights


class _SymmetricBuffer:
    def __init__(self, owner, group, experts, capacity, topk, hidden, intermediate):
        self.owner, self.group = owner, group
        self.num_experts, self.num_max_tokens_per_rank = experts, capacity
        self.num_topk, self.hidden, self.intermediate_hidden = (
            topk,
            hidden,
            intermediate,
        )
        self.mma_type, self.activation = "fp4xfp4", "swiglu"
        self.buffer = torch.zeros(1, dtype=torch.uint8)
        self.x = torch.empty(capacity, hidden // 2, dtype=torch.uint8)
        self.x_sf = torch.empty(capacity, hidden // 32, dtype=torch.uint8)
        self.topk_idx = torch.empty(capacity, topk, dtype=torch.int64)
        self.topk_weights = torch.empty(capacity, topk, dtype=torch.float32)
        self.source = None
        self.destroyed = False

    def destroy(self):
        self.owner.case.assertFalse(self.destroyed, "workspace destroyed twice")
        self.owner.events.append(("destroy", self.num_max_tokens_per_rank))
        self.destroyed = True
        self.buffer = self.x = self.x_sf = self.topk_idx = self.topk_weights = None


class _NativeMega(types.ModuleType):
    """Strict signatures from DeepGEMM-for-sail 07674e6, not deep_gemm_mega."""

    def __init__(self, case, events):
        super().__init__("deep_gemm")
        self.case, self.events = case, events
        self.transforms, self.buffers, self.dispatches, self.completes = [], [], [], []
        self.expected_clamp = 10.0

    def transform_weights_for_mega_moe(
        self, l1_weights, l2_weights, activation="swiglu"
    ):
        self.case.assertEqual(activation, "swiglu")
        for pair, weight_shape, scale_shape in (
            (l1_weights, (4, 256, 256), (4, 256, 16)),
            (l2_weights, (4, 512, 64), (4, 512, 4)),
        ):
            self.case.assertEqual(tuple(pair[0].shape), weight_shape)
            self.case.assertEqual(tuple(pair[1].shape), scale_shape)
            for tensor in pair:
                self.case.assertEqual(tensor.dtype, torch.uint8)
                self.case.assertTrue(tensor.is_contiguous())
        # Distinct output storage catches accidental use of raw/grouped layouts.
        l1 = (
            l1_weights[0].flip(1).contiguous(),
            torch.full((4, 2048), 17, dtype=torch.uint16),
        )
        l2_scale = torch.full((4, 2, 512), 0x4243, dtype=torch.uint16).transpose(1, 2)
        l2 = (l2_weights[0], l2_scale)
        self.transforms.append(
            {
                "raw": tuple(
                    tensor.clone()
                    for pair in (l1_weights, l2_weights)
                    for tensor in pair
                ),
                "l1": l1,
                "l2": l2,
            }
        )
        return l1, l2

    def get_symm_buffer_for_mega_moe(
        self,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts=0,
        use_fp8_dispatch=None,
        mma_type="fp4xfp4",
        activation="swiglu",
    ):
        self.case.assertEqual(
            (num_experts, num_topk, hidden, intermediate_hidden), (8, 8, 512, 128)
        )
        self.case.assertEqual(num_shared_experts, 0)
        self.case.assertIn(use_fp8_dispatch, (None, False))
        self.case.assertEqual((mma_type, activation), ("fp4xfp4", "swiglu"))
        self.case.assertGreaterEqual(num_max_tokens_per_rank, 128)
        self.case.assertEqual(
            num_max_tokens_per_rank & (num_max_tokens_per_rank - 1), 0
        )
        result = _SymmetricBuffer(
            self,
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        self.buffers.append(result)
        self.events.append(("allocate", num_max_tokens_per_rank))
        return result

    def mega_moe_pre_dispatch(
        self, hidden_states, topk_idx, topk_weights, sym_buffer, num_tokens=None
    ):
        count = hidden_states.shape[0] if num_tokens is None else num_tokens
        self.case.assertFalse(sym_buffer.destroyed)
        self.case.assertLessEqual(count, sym_buffer.num_max_tokens_per_rank)
        for tensor, dtype in (
            (hidden_states, torch.bfloat16),
            (topk_idx, torch.int64),
            (topk_weights, torch.float32),
        ):
            self.case.assertEqual(tensor.dtype, dtype)
            self.case.assertTrue(tensor.is_contiguous())
        self.case.assertEqual(tuple(topk_idx.shape), (count, 8))
        self.case.assertEqual(tuple(topk_weights.shape), (count, 8))
        sym_buffer.source = hidden_states[:count].clone()
        sym_buffer.topk_idx[:count].copy_(topk_idx[:count])
        sym_buffer.topk_weights[:count].copy_(topk_weights[:count])
        self.dispatches.append(
            (hidden_states.clone(), topk_idx.clone(), topk_weights.clone())
        )
        self.events.append(("pre_dispatch", count))

    def fp4_fp4_mega_moe(
        self,
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        cumulative_local_expert_recv_stats=None,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=None,
        fast_math=True,
        *,
        l1_weights_replicas=None,
        l2_weights_replicas=None,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    ):
        self.case.assertFalse(sym_buffer.destroyed)
        self.case.assertEqual(y.dtype, torch.bfloat16)
        self.case.assertEqual((recipe, activation), ((1, 1, 32), "swiglu"))
        self.case.assertEqual(activation_clamp, self.expected_clamp)
        self.case.assertIs(fast_math, True)
        self.case.assertIsNone(l1_weights_replicas)
        self.case.assertIsNone(l2_weights_replicas)
        self.case.assertEqual((situ_beta, situ_linear_beta), (4.0, 25.0))
        record = next(
            record for record in self.transforms if record["l1"][0] is l1_weights[0]
        )
        for supplied, transformed in zip(
            (*l1_weights, *l2_weights), (*record["l1"], *record["l2"])
        ):
            self.case.assertIs(supplied, transformed)
        self.case.assertTrue(l1_weights[1].is_contiguous())
        self.case.assertTrue(l2_weights[1].transpose(1, 2).is_contiguous())
        stats = cumulative_local_expert_recv_stats
        self.case.assertIsNotNone(stats)
        self.case.assertEqual((stats.dtype, tuple(stats.shape)), (torch.int32, (4,)))
        count = y.shape[0]
        self.case.assertEqual(tuple(sym_buffer.source.shape), tuple(y.shape))
        output = torch.zeros(y.shape, dtype=torch.float32)
        start = sym_buffer.group.rank() * 4
        for token in range(count):
            for slot in range(8):
                expert = int(sym_buffer.topk_idx[token, slot])
                if expert < 0:
                    continue
                weight = float(sym_buffer.topk_weights[token, slot])
                output[token] += (
                    sym_buffer.source[token].float() * (expert + 1) * weight
                )
                if start <= expert < start + 4:
                    stats[expert - start] += 1
        y.copy_(output.to(y.dtype))
        self.completes.append({"rows": count, "buffer": sym_buffer, "stats": stats})
        self.events.append(("complete", count))


class _CPUHarness:
    def __init__(self, case, global_rows=()):
        self.case, self.global_rows = case, list(global_rows)
        self.events, self.reductions, self.runners = [], [], []
        self.group = _Group()
        self.native = _NativeMega(case, self.events)
        self.capturing = False
        self.stack = ExitStack()

    def __enter__(self):
        enter = self.stack.enter_context
        enter(mock.patch.dict(sys.modules, {"deep_gemm": self.native}))
        enter(
            mock.patch.object(
                moe_module.PPUMoE, "_validate_megamoe_input", return_value=None
            )
        )
        enter(
            mock.patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=lambda: self.capturing,
            )
        )
        enter(
            mock.patch.object(
                torch.cuda,
                "synchronize",
                side_effect=lambda *args, **kwargs: self.events.append(("sync",)),
            )
        )
        enter(
            mock.patch.object(
                torch.distributed, "group", types.SimpleNamespace(WORLD=self.group)
            )
        )
        enter(mock.patch.object(torch.distributed, "is_initialized", return_value=True))
        enter(
            mock.patch.object(
                torch.distributed, "all_reduce", side_effect=self.all_reduce
            )
        )
        # These branches must remain untouched, even with an empty local batch.
        for name in (
            "_use_deepep_dispatch",
            "_forward_deepep",
            "_forward_local",
            "_local_compute_grouped",
        ):
            enter(
                mock.patch.object(
                    moe_module.PPUMoE,
                    name,
                    side_effect=AssertionError("MegaMoE entered " + name),
                )
            )
        enter(
            mock.patch.object(
                moe_module,
                "ep_gather",
                side_effect=AssertionError("extra weighted gather"),
            )
        )
        enter(
            mock.patch.object(
                moe_module,
                "routed_swiglu_fp8",
                side_effect=AssertionError("W4A8 preweighting in MegaMoE"),
            )
        )
        return self

    def __exit__(self, kind, value, traceback):
        try:
            for runner in self.runners:
                runner.close()
        finally:
            self.stack.close()

    def all_reduce(self, rows, op, group=None):
        self.case.assertEqual(op, torch.distributed.ReduceOp.MAX)
        self.case.assertEqual(rows.dtype, torch.int64)
        self.case.assertEqual(rows.numel(), 1)
        local = int(rows.item())
        global_count = self.global_rows.pop(0) if self.global_rows else local
        self.case.assertGreaterEqual(global_count, local)
        self.reductions.append((local, global_count, group))
        self.events.append(("all_reduce", local, global_count))
        rows.fill_(global_count)

    def runner(self, *, group=None, rank=0, limit=10.0, bind=True):
        with _selection("megamoe"):
            runner = moe_module.PPUMoE(
                num_routed_experts=8,
                num_local_experts=4,
                ep_size=2,
                ep_rank=rank,
                hidden_size=512,
                inter_dim=128,
                swiglu_limit=limit,
                group=group,
            )
        self.runners.append(runner)
        if bind:
            runner.bind_expert_weights(7, *_weights())
        return runner


class MegaMoESelectionTest(unittest.TestCase):
    def test_provider_requires_only_four_native_mega_apis_and_rejects_conflict(self):
        with _selection():
            from rtp_llm.models_py.modules.dsv41.ppu import backend as provider

        env = {MOE_BACKEND_ENV: "megamoe"}
        required = dict(provider.kernel_symbols(env))["deep_gemm"]
        self.assertEqual(
            set(required),
            {
                "fp4_fp4_mega_moe",
                "get_symm_buffer_for_mega_moe",
                "transform_weights_for_mega_moe",
                "mega_moe_pre_dispatch",
            },
        )
        self.assertEqual(len(required), 4)
        # Mega uses raw MXFP4 transform, without W4A8 preprocessing or enum.
        provider.check_moe_preprocess(types.SimpleNamespace(), env)
        with self.assertRaisesRegex(ValueError, "W4A4"):
            provider.kernel_symbols({**env, MOE_PRECISION_ENV: "w4a8"})

    def test_backend_precision_combinations_and_kernel_names(self):
        cases = (
            ({}, "grouped", "w4a8"),
            ({MOE_PRECISION_ENV: "w4a4"}, "grouped", "w4a4"),
            ({MOE_BACKEND_ENV: "megamoe"}, "megamoe", "w4a4"),
            (
                {MOE_BACKEND_ENV: " MegaMoE ", MOE_PRECISION_ENV: " W4A4 "},
                "megamoe",
                "w4a4",
            ),
        )
        for env, backend, precision in cases:
            with self.subTest(env=env):
                self.assertEqual(moe_backend(env), backend)
                self.assertEqual(moe_precision(env), precision)
                expected = (
                    "fp4_fp4_mega_moe"
                    if backend == "megamoe"
                    else (
                        "m_grouped_gemm_{}_fp4_bf16_nt_nopad".format(
                            "fp8" if precision == "w4a8" else "fp4"
                        )
                    )
                )
                self.assertEqual(moe_kernel_name(precision, backend), expected)

    def test_invalid_backend_and_explicit_w4a8_conflict_are_rejected(self):
        for value in ("", "mega", "deep_gemm_mega", "auto"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, MOE_BACKEND_ENV):
                    moe_backend({MOE_BACKEND_ENV: value})
        env = {MOE_BACKEND_ENV: "megamoe", MOE_PRECISION_ENV: "w4a8"}
        with self.assertRaisesRegex(ValueError, "W4A4"):
            moe_precision(env)
        with _selection("megamoe", "w4a8"):
            with self.assertRaisesRegex(ValueError, "W4A4"):
                moe_module.PPUMoE(num_routed_experts=8, num_local_experts=4, ep_size=2)
        with self.assertRaisesRegex(ValueError, "W4A4"):
            moe_kernel_name("w4a8", "megamoe")

    def test_instance_freezes_backend_precision_and_real_forward_route(self):
        with _CPUHarness(self) as harness:
            runner = harness.runner()
            x, ids, weights = _inputs(2)
            with _selection("grouped", "w4a8"):
                actual = runner.forward(x, ids, weights, layer_id=7)
            self.assertEqual((runner.backend, runner.precision), ("megamoe", "w4a4"))
            self.assertEqual(tuple(actual.shape), tuple(x.shape))
            self.assertEqual(len(harness.native.completes), 1)


class MegaMoENativeBoundaryTest(unittest.TestCase):
    def test_strict_native_abi_global_routes_padding_and_single_weight_application(
        self,
    ):
        with _CPUHarness(self) as harness:
            runner = harness.runner()
            x, ids, weights = _inputs(3)
            actual = runner(x, ids, weights, layer_id=7)
            staged_x, staged_ids, staged_weights = harness.native.dispatches[0]
            torch.testing.assert_close(staged_x, x, rtol=0, atol=0)
            self.assertTrue(torch.equal(staged_ids[:, :6], ids.long()))
            self.assertTrue(torch.equal(staged_weights[:, :6], weights.float()))
            self.assertTrue((staged_ids[:, 6:] == -1).all())
            self.assertTrue((staged_weights[:, 6:] == 0).all())
            coefficients = ((ids.float() + 1) * weights.float()).sum(-1)
            expected = (x.float() * coefficients[:, None]).bfloat16()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            stats = runner.execution_stats()
            self.assertEqual(stats["backend"], "megamoe")
            self.assertEqual(stats["precision"], "w4a4")
            self.assertEqual(stats["kernel"], "fp4_fp4_mega_moe")
            self.assertEqual(stats["mega_moe_calls"], 1)
            self.assertEqual(stats["grouped_gemm_calls"], 0)
            self.assertEqual(stats["route_weight_stage"], "after_gemm2")

    def test_missing_native_apis_fail_clearly_on_every_attempt_including_empty_rank(
        self,
    ):
        names = (
            "get_symm_buffer_for_mega_moe",
            "transform_weights_for_mega_moe",
            "mega_moe_pre_dispatch",
            "fp4_fp4_mega_moe",
        )
        for name in names:
            with self.subTest(api=name), _CPUHarness(self) as harness:
                runner = harness.runner()
                setattr(harness.native, name, None)
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, name):
                        runner(*_inputs(0), layer_id=7)
                self.assertEqual(harness.native.buffers, [])

    def test_transform_receives_raw_bytes_and_cache_rebinding_is_layer_specific(self):
        with _CPUHarness(self) as harness:
            runner = harness.runner(bind=False)
            original = _weights()
            runner.bind_expert_weights(7, *original)
            runner.bind_expert_weights(8, *_weights(seed=902))
            device = torch.device("cpu")
            before = runner._processed_weights(7, device)
            self.assertEqual(len(before), 6)
            self.assertEqual(before[4:], (1.0, 1.0))
            self.assertIs(runner._processed_weights(7, device), before)
            self.assertEqual(len(harness.native.transforms), 1)
            for observed, raw in zip(
                harness.native.transforms[0]["raw"],
                (t for pair in original for t in pair),
            ):
                self.assertTrue(
                    torch.equal(observed, raw.contiguous().view(torch.uint8))
                )
            other = runner._processed_weights(8, device)
            runner._processed[(7, torch.device("cuda:7"))] = before
            replacement = _weights(seed=903)
            runner.bind_expert_weights(7, *replacement)
            self.assertFalse(any(key[0] == 7 for key in runner._processed))
            self.assertIs(runner._processed_weights(8, device), other)
            after = runner._processed_weights(7, device)
            self.assertIsNot(after, before)
            self.assertEqual(len(harness.native.transforms), 3)
            self.assertTrue(
                torch.equal(
                    harness.native.transforms[-1]["raw"][0],
                    replacement[0][0].view(torch.uint8),
                )
            )
            runner(*_inputs(1), layer_id=7)
            self.assertEqual(len(harness.native.transforms), 3)

    def test_disabled_clamp_is_forwarded_as_none(self):
        with _CPUHarness(self) as harness:
            harness.native.expected_clamp = None
            runner = harness.runner(limit=0.0)
            runner(*_inputs(1), layer_id=7)
            self.assertEqual(len(harness.native.completes), 1)


class MegaMoECollectiveLifetimeTest(unittest.TestCase):
    def test_global_max_drives_equal_capacity_and_idle_rank_growth(self):
        with _CPUHarness(self, global_rows=(129, 65, 257, 3)) as harness:
            runner = harness.runner()
            for local_rows in (3, 0, 0, 1):
                result = runner(*_inputs(local_rows), layer_id=7)
                self.assertEqual(tuple(result.shape), (local_rows, 512))
            self.assertEqual(
                [(local, count) for local, count, _ in harness.reductions],
                [(3, 129), (0, 65), (0, 257), (1, 3)],
            )
            self.assertTrue(
                all(group is harness.group for _, _, group in harness.reductions)
            )
            self.assertEqual(
                [buffer.num_max_tokens_per_rank for buffer in harness.native.buffers],
                [256, 512],
            )
            self.assertEqual(
                [call["rows"] for call in harness.native.completes], [3, 0, 0, 1]
            )
            old, current = harness.native.buffers
            self.assertTrue(old.destroyed)
            self.assertFalse(current.destroyed)
            destroy = harness.events.index(("destroy", 256))
            self.assertEqual(harness.events[destroy - 1], ("sync",))
            self.assertEqual(harness.events[destroy + 1], ("allocate", 512))
            stats = harness.native.completes[0]["stats"]
            self.assertTrue(
                all(call["stats"] is stats for call in harness.native.completes)
            )
            # Global IDs 0,1,2 are local to rank 0, including the zero-weight route.
            self.assertEqual(int(stats.sum()), 12)
            self.assertEqual(runner.execution_stats()["mega_moe_calls"], 4)

    def test_all_empty_world_still_allocates_minimum_and_completes(self):
        with _CPUHarness(self, global_rows=(0,)) as harness:
            runner = harness.runner()
            result = runner(*_inputs(0), layer_id=7)
            self.assertEqual(tuple(result.shape), (0, 512))
            self.assertEqual(harness.native.buffers[0].num_max_tokens_per_rank, 128)
            self.assertEqual(len(harness.native.dispatches), 1)
            self.assertEqual(len(harness.native.completes), 1)

    def test_explicit_group_is_used_and_rank_size_mismatches_fail_before_allocation(
        self,
    ):
        with _CPUHarness(self) as harness:
            explicit = _Group(size=2, rank=1)
            runner = harness.runner(group=explicit, rank=1)
            runner(*_inputs(1), layer_id=7)
            self.assertIs(harness.reductions[0][2], explicit)
            self.assertIs(harness.native.buffers[0].group, explicit)
        for group in (_Group(size=1), _Group(size=2, rank=1)):
            with self.subTest(size=group.size(), rank=group.rank()), _CPUHarness(
                self
            ) as harness:
                runner = harness.runner(group=group)
                with self.assertRaisesRegex(RuntimeError, "EP size and rank"):
                    runner(*_inputs(0), layer_id=7)
                self.assertEqual(harness.native.buffers, [])
                self.assertEqual(harness.reductions, [])

    def test_missing_world_group_fails_before_collective_allocation(self):
        with _CPUHarness(self) as harness:
            runner = harness.runner()
            with mock.patch.object(
                torch.distributed, "is_initialized", return_value=False
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "initialized EP process group"
                ):
                    runner(*_inputs(0), layer_id=7)
            self.assertEqual(harness.native.buffers, [])
            self.assertEqual(harness.reductions, [])

    def test_capture_requires_warmup_and_reuses_stable_capacity_without_collective(
        self,
    ):
        with _CPUHarness(self) as harness:
            runner = harness.runner()
            harness.capturing = True
            with self.assertRaisesRegex(RuntimeError, "warmup"):
                runner(*_inputs(0), layer_id=7)
            self.assertEqual(harness.reductions, [])
            harness.capturing = False
            runner(*_inputs(3), layer_id=7)
            buffer = harness.native.buffers[0]
            harness.capturing = True
            runner(*_inputs(2), layer_id=7)
            self.assertEqual(len(harness.reductions), 1)
            self.assertIs(harness.native.completes[-1]["buffer"], buffer)
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                runner(*_inputs(129), layer_id=7)
            self.assertEqual(len(harness.native.buffers), 1)
            self.assertFalse(buffer.destroyed)

    def test_close_drains_before_destroy_and_is_idempotent(self):
        with _CPUHarness(self) as harness:
            runner = harness.runner()
            runner(*_inputs(1), layer_id=7)
            harness.events.clear()
            runner.close()
            self.assertEqual(harness.events, [("sync",), ("destroy", 128)])
            runner.close()
            self.assertEqual(harness.events, [("sync",), ("destroy", 128)])

    def test_class_shutdown_hook_closes_live_workspaces_without_retaining_engines(self):
        with _CPUHarness(self) as harness:
            left, right = harness.runner(), harness.runner()
            left(*_inputs(1), layer_id=7)
            right(*_inputs(1), layer_id=7)
            moe_module.PPUMoE.close_mega_resources()
            self.assertTrue(all(buffer.destroyed for buffer in harness.native.buffers))
            destroyed = sum(event[0] == "destroy" for event in harness.events)
            moe_module.PPUMoE.close_mega_resources()
            self.assertEqual(
                sum(event[0] == "destroy" for event in harness.events), destroyed
            )
            reference = weakref.ref(left)
            harness.runners.remove(left)
            del left
            gc.collect()
            self.assertIsNone(
                reference(), "native shutdown registry must use weak references"
            )


if __name__ == "__main__":
    unittest.main()
