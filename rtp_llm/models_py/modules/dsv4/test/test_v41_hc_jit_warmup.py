"""CPU contracts for V4.1 delayed-HC startup compilation; no GPU launches."""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4.hc import v41_jit_warmup as warmup
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit


def unit(*, eps=1e-6, cls=DelayedHCUnit):
    return cls(
        torch.zeros(24, 20480),
        torch.zeros(24),
        torch.ones(3),
        dim=5120,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=eps,
        hc_eps=eps,
    )


def block(*, eps=1e-6):
    result = torch.nn.Module()
    result.attn_hc, result.ffn_hc = unit(eps=eps), unit(eps=eps)
    result.ffn_hc.set_previous(result.attn_hc)
    result.ffn_norm = RMSNorm(torch.ones(5120, dtype=torch.bfloat16), eps)
    return result


def model(*blocks):
    result = torch.nn.Module()
    result.layers = torch.nn.ModuleList(blocks)
    return result


class V41HCJitWarmupCPU(unittest.TestCase):
    def setUp(self):
        warmup._WARMED_KEYS.clear()
        self.addCleanup(warmup._WARMED_KEYS.clear)

    def _mock_device(self, stack):
        def run(_label, _description, launch, *, device):
            return launch()

        stack.enter_context(
            patch.object(warmup.common, "model_warm_up_enabled", return_value=True)
        )
        stack.enter_context(
            patch.object(warmup.common, "_is_cuda_device", return_value=True)
        )
        stack.enter_context(patch.object(warmup.common, "_assert_not_capturing"))
        stack.enter_context(
            patch.object(warmup.common, "_get_deep_gemm_num_sms", return_value=148)
        )
        stack.enter_context(patch.object(warmup.common, "_sync_cuda"))
        stack.enter_context(patch.object(warmup.common, "_release_cuda_cache"))
        stack.enter_context(
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3))
        )
        stack.enter_context(
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=7),
            )
        )
        stack.enter_context(
            patch.object(
                torch.cuda,
                "device",
                side_effect=lambda device: contextlib.nullcontext(),
            )
        )
        stack.enter_context(
            patch.object(warmup.v41_prenorm, "_has_prenorm_gemm", return_value=True)
        )
        stack.enter_context(
            patch.object(warmup.v41_mega_mhc, "_get_mega_mhc", return_value=Mock())
        )
        stack.enter_context(
            patch.dict(
                os.environ, {"DSV41_FUSED_MHC_PRENORM": "1", "DSV41_MEGA_MHC": "1"}
            )
        )
        for name in (
            "_run_deepgemm_warmup_launch_with_retry",
            "_run_tilelang_warmup_launch_with_retry",
            "_run_triton_warmup_launch_with_retry",
        ):
            stack.enter_context(patch.object(warmup.common, name, side_effect=run))
        return stack.enter_context(
            patch.object(
                warmup.common,
                "_run_deepgemm_warmup_launches_serialized",
                side_effect=lambda label, launch: launch(),
            )
        )

    def test_collects_real_units_and_only_correct_in_block_seams(self):
        first, same, other = block(), block(), block(eps=2e-5)
        units, pairs = warmup._collect_v41_hc_configs(model(first, same, other))
        self.assertEqual(len(units), 2)
        self.assertEqual(len(pairs), 2)
        self.assertIs(next(iter(units.values())), first.attn_hc)
        self.assertIs(next(iter(pairs.values()))[0], first.attn_hc)
        same.ffn_hc.set_previous(first.attn_hc)
        self.assertEqual(len(warmup._collect_v41_hc_configs(model(same))[1]), 0)
        same.ffn_hc.set_previous(same.attn_hc)
        same.ffn_norm = torch.nn.Identity()
        self.assertEqual(len(warmup._collect_v41_hc_configs(model(same))[1]), 0)

    def test_subclasses_and_invalid_weight_layout_are_not_guessed(self):
        class ChangedUnit(DelayedHCUnit):
            pass

        self.assertEqual(
            warmup._collect_v41_hc_configs(model(unit(cls=ChangedUnit))), ({}, {})
        )
        owner = unit()
        owner.fn = owner.fn.t().contiguous().t()
        self.assertIsNone(warmup._unit_key(owner))

    def test_small_prenorm_covers_every_reduction_constexpr(self):
        self.assertEqual(warmup._small_prenorm_ms(0), ())
        self.assertEqual(warmup._small_prenorm_ms(3), (1, 2, 3))
        self.assertEqual(warmup._small_prenorm_ms(131072), tuple(range(1, 65)))

    def test_mega_representatives_cover_pinned_dg_heuristic_with_bounded_memory(self):
        for sms in (1, 16, 132, 148):
            for maximum in (1, 64, 65, 128, 1024, 131072):
                reps = warmup._mega_representative_ms(maximum, sms)

                # Independent translation of pinned get_num_splits: K=80 blocks.
                def split(m):
                    cap = min(64, max(16, sms // ((m + 63) // 64)))
                    return (80 + ((80 + cap - 1) // cap) - 1) // ((80 + cap - 1) // cap)

                self.assertEqual(
                    {split(m) for m in reps}, {split(m) for m in range(1, maximum + 1)}
                )
                self.assertEqual(len(reps), len({split(m) for m in reps}))
                self.assertTrue(all(1 <= m <= maximum for m in reps))
                self.assertLessEqual(max(reps), sms * 4 + 1)
        self.assertEqual(warmup._mega_representative_ms(0, 148), ())
        self.assertEqual(
            warmup._mega_representative_ms(131072, 148), (1, 193, 321, 449)
        )
        with self.assertRaises(ValueError):
            warmup._mega_representative_ms(1, 0)

    def test_global_switch_and_cpu_skip_without_backend_or_allocation(self):
        with patch.object(
            warmup.common, "model_warm_up_enabled", return_value=False
        ), patch.object(warmup, "_collect_v41_hc_configs") as collect:
            warmup.warmup_v41_hc_jit(None, max_m=131072, device="cuda")
            collect.assert_not_called()
        with patch.object(
            warmup.common, "model_warm_up_enabled", return_value=True
        ), patch.object(warmup, "_collect_v41_hc_configs") as collect:
            warmup.warmup_v41_hc_jit(None, max_m=131072, device="cpu")
            collect.assert_not_called()
        self.assertFalse(warmup._WARMED_KEYS)

    def test_wrong_architecture_and_capture_skip_or_fail_before_clone(self):
        with contextlib.ExitStack() as stack:
            self._mock_device(stack)
            clone = stack.enter_context(patch.object(warmup, "_clone_unit"))
            with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                warmup.warmup_v41_hc_jit(None, max_m=1024, device="cpu")
            clone.assert_not_called()
            with patch.object(
                warmup.common,
                "_assert_not_capturing",
                side_effect=RuntimeError("capture"),
            ):
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    warmup.warmup_v41_hc_jit(None, max_m=1024, device="cpu")
            clone.assert_not_called()

    def test_runtime_wrappers_receive_private_weights_and_live_state_is_unchanged(self):
        source = block()
        source.attn_hc.pre_mix_out = torch.full((2, 4), 13.0)
        source.ffn_hc.pre_mix_out = torch.full((2, 4), 17.0)
        previous_pre, next_pre = source.attn_hc.pre_mix_out, source.ffn_hc.pre_mix_out
        pair = warmup._clone_pair(
            (source.attn_hc, source.ffn_hc, source.ffn_norm), "cpu"
        )
        self.assertIs(pair[1]._previous_ref(), pair[0])
        for got, original in zip(pair[:2], (source.attn_hc, source.ffn_hc)):
            self.assertNotEqual(got.fn.data_ptr(), original.fn.data_ptr())
            self.assertNotEqual(got.base.data_ptr(), original.base.data_ptr())
            self.assertNotEqual(got.scale.data_ptr(), original.scale.data_ptr())
        self.assertNotEqual(
            pair[2].weight.data_ptr(), source.ffn_norm.weight.data_ptr()
        )

        def prenorm(residual, fn, eps):
            self.assertEqual(tuple(residual.shape), (1, 6, 4, 5120))
            fn._dsv41_prenorm_tf32_cache = "dummy-cache"
            return torch.zeros(1, 6, 24)

        def mega(x, residual, post, comb, previous, following, norm):
            self.assertIs(previous, pair[0])
            self.assertIs(following, pair[1])
            self.assertEqual(tuple(previous.pre_mix_out.shape), (6, 4))
            following.pre_mix_out = torch.zeros(6, 4)
            return residual, x, post, comb

        with patch.object(warmup.v41_prenorm, "prenorm", side_effect=prenorm):
            warmup._launch_small_prenorm(pair[1], 6, "cpu")
        with patch.object(warmup.v41_mega_mhc, "try_fused_post_pre", side_effect=mega):
            warmup._launch_mega(pair, 6, "cpu")
        self.assertIs(source.attn_hc.pre_mix_out, previous_pre)
        self.assertIs(source.ffn_hc.pre_mix_out, next_pre)
        self.assertIs(source.ffn_hc._previous_ref(), source.attn_hc)
        self.assertFalse(hasattr(source.ffn_hc.fn, "_dsv41_prenorm_tf32_cache"))
        torch.testing.assert_close(previous_pre, torch.full((2, 4), 13.0))
        torch.testing.assert_close(next_pre, torch.full((2, 4), 17.0))

    def test_tilelang_chain_covers_norm_split_sinkhorn_apply_and_inplace_post(self):
        source = unit()
        calls = []
        ops = SimpleNamespace()

        def norm(residual, fn, weight, eps, n_splits):
            calls.append("norm")
            self.assertEqual(n_splits, 1)
            return torch.zeros(1, 65, 24)

        def split(mixes, scale, base, hc, post_scale, eps):
            calls.append("split")
            return (
                torch.ones(1, 65, 4, 1),
                torch.ones(1, 65, 4, 1),
                torch.ones(1, 65, 4, 4),
            )

        def sinkhorn(comb, *, repeat, eps):
            calls.append("sinkhorn")
            self.assertEqual(repeat, 20)
            return comb

        def apply(residual, pre):
            calls.append("apply")
            return residual[:, :, 0, :].contiguous()

        def post(hidden, residual, post, comb, *, out):
            calls.append("post")
            self.assertIs(out, residual)
            return out.fill_(7)

        ops.mhc_pre_norm_fn, ops.mhc_pre_split_mixes = norm, split
        ops.sinkhorn_normalize, ops.mhc_pre_apply_mix, ops.mhc_post = (
            sinkhorn,
            apply,
            post,
        )
        with patch.object(warmup, "_tile_ops", return_value=ops):
            warmup._launch_tilelang_chain(source, 65, "cpu", include_norm=True)
        self.assertEqual(calls, ["norm", "split", "sinkhorn", "apply", "post"])
        self.assertIsNone(source.pre_mix_out)

    def test_entrypoint_dedupes_configs_and_successful_repeated_calls(self):
        root = model(block(), block())
        with contextlib.ExitStack() as stack:
            serialized = self._mock_device(stack)
            small = stack.enter_context(patch.object(warmup, "_launch_small_prenorm"))
            tile = stack.enter_context(patch.object(warmup, "_launch_tilelang_chain"))
            mega = stack.enter_context(patch.object(warmup, "_launch_mega"))
            for _ in range(2):
                warmup.warmup_v41_hc_jit(root, max_m=131072, device="cpu")
            self.assertEqual(small.call_count, 64)
            self.assertEqual(
                [call.args[1] for call in small.call_args_list], list(range(1, 65))
            )
            tile.assert_called_once()
            self.assertEqual(tile.call_args.args[1], 65)
            self.assertTrue(tile.call_args.kwargs["include_norm"])
            self.assertEqual(
                [call.args[1] for call in mega.call_args_list], [1, 193, 321, 449]
            )
            serialized.assert_called_once()
        self.assertEqual(len(warmup._WARMED_KEYS), 1)

    def test_feature_disables_compile_reachable_tilelang_fallback_only(self):
        with contextlib.ExitStack() as stack:
            self._mock_device(stack)
            stack.enter_context(
                patch.dict(
                    os.environ, {"DSV41_FUSED_MHC_PRENORM": "0", "DSV41_MEGA_MHC": "0"}
                )
            )
            small = stack.enter_context(patch.object(warmup, "_launch_small_prenorm"))
            mega = stack.enter_context(patch.object(warmup, "_launch_mega"))
            tile = stack.enter_context(patch.object(warmup, "_launch_tilelang_chain"))
            warmup.warmup_v41_hc_jit(model(block()), max_m=4, device="cpu")
            small.assert_not_called()
            mega.assert_not_called()
            self.assertEqual(tile.call_args.args[1], 4)
            self.assertTrue(tile.call_args.kwargs["include_norm"])

    def test_failure_is_not_marked_warm_and_same_configuration_can_retry(self):
        root = model(block())
        with contextlib.ExitStack() as stack:
            self._mock_device(stack)
            stack.enter_context(patch.object(warmup, "_launch_small_prenorm"))
            stack.enter_context(patch.object(warmup, "_launch_mega"))
            tile = stack.enter_context(
                patch.object(
                    warmup,
                    "_launch_tilelang_chain",
                    side_effect=RuntimeError("bad shape"),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "bad shape"):
                warmup.warmup_v41_hc_jit(root, max_m=4, device="cpu")
            self.assertFalse(warmup._WARMED_KEYS)
            tile.side_effect = None
            warmup.warmup_v41_hc_jit(root, max_m=4, device="cpu")
            self.assertFalse(tile.call_args.kwargs["include_norm"])
            self.assertEqual(len(warmup._WARMED_KEYS), 1)

    def test_reachable_wrappers_returning_none_fail_loudly(self):
        dummy = unit()
        with patch.object(warmup.v41_prenorm, "prenorm", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "small prenorm"):
                warmup._launch_small_prenorm(dummy, 1, "cpu")
        owner = block()
        pair = warmup._clone_pair((owner.attn_hc, owner.ffn_hc, owner.ffn_norm), "cpu")
        with patch.object(warmup.v41_mega_mhc, "try_fused_post_pre", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "mega_mhc"):
                warmup._launch_mega(pair, 1, "cpu")


if __name__ == "__main__":
    unittest.main()
