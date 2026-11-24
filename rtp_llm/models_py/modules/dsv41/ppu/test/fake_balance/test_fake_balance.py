"""Route invariants and dispatch integration, independent of native kernels."""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import moe
from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE, _DPForwardMegaBuffer


def engine(rank=0, enabled=True, backend="grouped"):
    with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": backend}):
        result = PPUMoE(ep_size=8, ep_rank=rank)
    result.configure_fake_balance(enabled, top_k=6, dp_rank=rank, dp_size=8)
    return result


def routes(rows=8):
    ids = torch.full((rows, 8), -1, dtype=torch.int64)
    ids[:, :6] = torch.arange(6)
    weights = torch.zeros((rows, 8), dtype=torch.float32)
    weights[:, :6] = torch.arange(1, 7).float() / 20
    return ids, weights


def prepare_double(value, capacity=128, *, generation=1, epoch=1):
    value._mega_buffer = NS(destroy=Mock())
    value._mega_device = torch.device('cpu')
    value._mega_capacity, value._mega_generation, value._dp_prepare_epoch = capacity, generation, epoch
    receipt = _DPForwardMegaBuffer(value, value._mega_buffer, capacity, capacity, generation, epoch)
    value._dp_active_preparation = receipt
    return receipt


class _MaskCPU:
    """Mask/padding transport double; actual Triton/native parity is in device.py."""
    def __getitem__(self, grid):
        def run(ids, pattern, output_ids, output_weights, rows, **kwargs):
            original = ids[:, :6]
            output_ids[:, :6].copy_(torch.where(original >= 0, pattern[:rows], original))
            output_weights[:, :6].copy_((original >= 0).float() / 6)
            output_ids[:, 6:].fill_(-1)
            output_weights[:, 6:].zero_()
        return run


def native_double(value):
    reference = engine(value._fake_balance_config[2])

    def run(ids, weights):
        source = torch.zeros_like(ids)
        expected = reference._apply_fake_balance(source, torch.zeros_like(weights))
        ids.copy_(expected[0])
        # The existing native op writes ones; normalization belongs to PPUMoE.
        weights.copy_(expected[1] * reference._fake_balance_config[1])
    return Mock(side_effect=run)


def kernel_double():
    return NS(mask_fake_routes=_MaskCPU(), triton=NS(cdiv=lambda a, b: (a + b - 1) // b))


class FakeBalanceTest(unittest.TestCase):
    def test_default_and_explicit_off_preserve_objects(self):
        for value in (PPUMoE(ep_size=8), engine(enabled=False)):
            ids, weights = routes()
            actual = value._apply_fake_balance(ids, weights)
            self.assertIs(actual[0], ids)
            self.assertIs(actual[1], weights)

    def test_global_load_and_distinct_experts_for_decode_batches(self):
        for rows in (1, 3, 8, 16, 32, 64, 128, 256):
            counts = torch.zeros(384, dtype=torch.int64)
            for rank in range(8):
                ids, weights = engine(rank)._apply_fake_balance(*routes(rows))
                self.assertTrue(torch.all(ids[:, 6:] == -1))
                self.assertTrue(torch.all(weights[:, 6:] == 0))
                self.assertTrue(torch.all(weights[:, :6] == 1 / 6))
                torch.testing.assert_close(weights.sum(-1), torch.ones(rows))
                for row in ids[:, :6]:
                    self.assertEqual(len(set(row.tolist())), 6)
                counts += torch.bincount(ids[:, :6].flatten(), minlength=384)
            self.assertEqual(int(counts.sum()), 8 * rows * 6)
            # Each EP rank receives the same number of valid assignments.
            self.assertEqual(counts.reshape(8, 48).sum(1).tolist(), [rows * 6] * 8)
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_masks_noncontiguous_storage_and_input_preservation(self):
        for dtype in (torch.int32, torch.int64):
            source_ids, source_weights = routes(16)
            source_ids = source_ids.to(dtype)
            source_ids[4:8] = -1
            source_weights[4:8] = 0
            ids, weights = source_ids[::2], source_weights[::2]
            before = ids.clone(), weights.clone()
            actual_ids, actual_weights = engine(3)._apply_fake_balance(ids, weights)
            self.assertEqual(actual_ids.dtype, dtype)
            self.assertTrue(torch.equal(actual_ids[2:4], before[0][2:4]))
            self.assertTrue(torch.all(actual_weights[2:4] == 0))
            self.assertTrue(torch.equal(ids, before[0]))
            self.assertTrue(torch.equal(weights, before[1]))

    def test_empty_and_idle_owners_keep_zero_valid_routes(self):
        for rows in (0, 1, 8):
            ids, weights = routes(rows)
            ids.fill_(-1)
            weights.zero_()
            actual_ids, actual_weights = engine()._apply_fake_balance(ids, weights)
            self.assertTrue(torch.all(actual_ids == -1))
            self.assertTrue(torch.all(actual_weights == 0))

    def test_override_reaches_both_backend_boundaries(self):
        for backend, method in (
            ("megamoe", "_forward_megamoe"),
            ("grouped", "_forward_deepep"),
        ):
            value = engine(backend=backend)
            ids, weights = routes()
            expected = value._apply_fake_balance(ids, weights)
            with patch.object(
                value, "_use_deepep_dispatch", return_value=True
            ), patch.object(value, method, return_value="result") as dispatch:
                self.assertEqual(
                    value.forward(torch.zeros(8, 5120), ids, weights, layer_id=0),
                    "result",
                )
            torch.testing.assert_close(dispatch.call_args.args[1], expected[0])
            torch.testing.assert_close(dispatch.call_args.args[2], expected[1])

    def test_configuration_is_explicit_validated_and_frozen(self):
        value = engine()
        value.configure_fake_balance(True, top_k=6, dp_rank=0, dp_size=8)
        with self.assertRaisesRegex(ValueError, "frozen"):
            value.configure_fake_balance(False, top_k=6, dp_rank=0, dp_size=8)
        for enabled, top_k, rank, size in (
            (1, 6, 0, 8),
            (True, 0, 0, 8),
            (True, 6, 8, 8),
        ):
            with self.assertRaises(ValueError):
                PPUMoE(ep_size=8).configure_fake_balance(
                    enabled, top_k=top_k, dp_rank=rank, dp_size=size
                )

    def test_shared_graph_overrides_logical_routes_before_physical_mapping(self):
        for enabled in (False, True):
            with patch.dict(os.environ, {"RTP_V41_PPU_MOE_PRECISION": "w4a4"}):
                value = engine(3, enabled=enabled, backend="megamoe")
            x = torch.zeros(8, 5120, dtype=torch.bfloat16)
            value._processed[(0, x.device)] = object()
            value._mega_buffer, value._mega_capacity = object(), 8
            ids, weights = routes()
            ids[2:4] = -1
            expected = value._apply_fake_balance(ids, weights)
            original = ids.clone(), weights.clone()
            with patch.object(value, "_map_expert_ids", side_effect=lambda ids, weights, layer: ids + 1000) as mapping, patch.object(
                value, "_forward_megamoe", return_value="native_result",
            ) as dispatch:
                self.assertEqual(value.forward_shared_graph(x, ids, weights, layer_id=0), "native_result")
            torch.testing.assert_close(mapping.call_args.args[0], expected[0])
            torch.testing.assert_close(mapping.call_args.args[1], expected[1])
            torch.testing.assert_close(dispatch.call_args.args[1], expected[0] + 1000)
            torch.testing.assert_close(dispatch.call_args.args[2], expected[1])
            self.assertIs(dispatch.call_args.kwargs["prepared_buffer"], value._mega_buffer)
            self.assertTrue(torch.equal(ids, original[0]))
            self.assertTrue(torch.equal(weights, original[1]))

    def test_shared_graph_prepares_target_and_draft_override_before_capture(self):
        for experts, top_k in ((384, 6), (128, 3)):
            with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": "megamoe",
                                        "RTP_V41_PPU_MOE_PRECISION": "w4a4"}):
                value = PPUMoE(ep_size=8, num_routed_experts=experts,
                               num_local_experts=experts // 8, top_k=top_k)
            value.configure_fake_balance(True, top_k=top_k, dp_rank=0, dp_size=8)
            x = torch.zeros(8, 5120, dtype=torch.bfloat16)
            with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), patch.object(
                value, "_validate_megamoe_input",
            ), patch.object(value, "_ensure_mega_buffer"), patch.object(
                value, "_processed_weights",
            ) as processed, patch.object(value, "_apply_fake_balance", wraps=value._apply_fake_balance) as override:
                value.prepare_shared_graph(x, (0, 1))
            self.assertEqual(override.call_count, 1)
            ids, weights = override.call_args.args
            self.assertEqual(ids.shape, (1, top_k))
            self.assertTrue(torch.all(ids[:, :top_k] == 0))
            self.assertTrue(torch.all(ids[:, top_k:] == -1))
            self.assertEqual(processed.call_count, 2)

    def test_gc_dispatch_receives_balanced_target_and_draft_routes(self):
        for experts, top_k in ((384, 6), (128, 3)):
            with self.subTest(experts=experts), patch.dict(os.environ, {
                "RTP_V41_PPU_MOE_BACKEND": "megamoe",
                "RTP_V41_PPU_MEGAMOE_FAKE_BALANCE_POLICY": "sail_round_robin",
            }):
                engine = PPUMoE(
                    ep_size=8, ep_rank=2, num_routed_experts=experts,
                    num_local_experts=experts // 8, top_k=top_k,
                )
                engine.configure_fake_balance(True, top_k=top_k, dp_rank=2, dp_size=8)
                dispatch = Mock()
                engine.gc_decode = NS(forward=dispatch)
                ids, weights = routes(4)
                ids[1] = -1
                valid = torch.tensor([True, False, True, True])
                count = torch.empty((), dtype=torch.int64)
                layer = 1
                expected = engine._apply_fake_balance(ids, weights, layer_id=layer)
                with patch.object(engine, "_apply_fake_balance", wraps=engine._apply_fake_balance) as balance:
                    engine.forward_gc_decode(
                        object(), ids, weights, layer_id=layer, expert_load=None,
                        valid=valid, shared_weights=(), routed_count=count, route_valid_mask=valid,
                    )
                self.assertEqual(balance.call_args.kwargs["layer_id"], layer)
                self.assertIs(balance.call_args.kwargs["routed_count"], count)
                self.assertIs(balance.call_args.kwargs["route_valid_mask"], valid)
                self.assertIs(dispatch.call_args.kwargs["valid"], valid)
                torch.testing.assert_close(dispatch.call_args.args[1], expected[0])
                torch.testing.assert_close(dispatch.call_args.args[2], expected[1])
                self.assertTrue(torch.all(dispatch.call_args.args[1][1] == -1))
                self.assertTrue(torch.all(dispatch.call_args.args[2][1] == 0))

    def test_compact_target_and_draft_match_legacy_mapping_and_keep_masks(self):
        for experts, top_k in ((384, 6), (128, 3)):
            for rank in range(8):
                with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": "megamoe"}):
                    value = PPUMoE(ep_size=8, ep_rank=rank, num_routed_experts=experts,
                                   num_local_experts=experts // 8, top_k=top_k)
                value.configure_fake_balance(True, top_k=top_k, dp_rank=rank, dp_size=8)
                with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": "grouped"}):
                    reference = PPUMoE(ep_size=8, ep_rank=rank, num_routed_experts=experts,
                                       num_local_experts=experts // 8, top_k=top_k)
                reference.configure_fake_balance(True, top_k=top_k, dp_rank=rank, dp_size=8)
                ids, weights = routes(16)
                ids[2, 1], ids[4:8] = -7, -1
                weights.fill_(float("nan"))
                expected = reference._apply_fake_balance(ids, weights)
                actual = value._apply_fake_balance(ids, weights)
                for new, old in zip(actual, expected):
                    self.assertEqual(new.shape, (16, top_k))
                    torch.testing.assert_close(new, old[:, :top_k])

    def test_sail_round_robin_matches_logical_layer_and_shard_formula(self):
        for experts, top_k in ((384, 6), (128, 3)):
            for rank in range(8):
                with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": "megamoe",
                    "RTP_V41_PPU_MEGAMOE_FAKE_BALANCE_POLICY": "sail_round_robin"}):
                    value = PPUMoE(ep_size=8, ep_rank=rank, num_routed_experts=experts,
                                   num_local_experts=experts // 8, top_k=top_k)
                value.configure_fake_balance(True, top_k=top_k, dp_rank=rank, dp_size=8)
                for layer in (0, 1, 17, 39):
                    ids, weights = routes(32)
                    ids[3, 1], ids[8:10] = -7, -1
                    before = ids.clone(), weights.clone()
                    actual = value._apply_fake_balance(ids, weights, layer_id=layer)
                    deal = torch.arange(32 * top_k).reshape(32, top_k) + layer + rank + 8 * rank
                    expected = (deal % 8) * (experts // 8) + (deal // 8) % (experts // 8)
                    valid = ids[:, :top_k] >= 0
                    torch.testing.assert_close(actual[0], torch.where(valid, expected, ids[:, :top_k]))
                    torch.testing.assert_close(actual[1], valid.float() / top_k)
                    self.assertTrue(torch.equal(ids, before[0]))
                    self.assertTrue(torch.equal(weights, before[1]))
                self.assertEqual(value.execution_stats()["fake_balance_policy"], "sail_round_robin")

    def test_sail_policy_declines_native_prefill_cache_and_forwards_layer(self):
        with patch.dict(os.environ, {"RTP_V41_PPU_MEGAMOE_FAKE_BALANCE_POLICY": "sail_round_robin"}):
            value = engine(3, backend="megamoe")
        ids, weights = routes()
        receipt = prepare_double(value)
        with patch.object(value, "_prefill_fake_device_eligible", side_effect=AssertionError("native cache lookup")):
            self.assertIsNone(value._try_prefill_fake_balance(ids, weights, receipt))
        expected = value._apply_fake_balance(ids, weights, layer_id=17)
        with patch.object(value, "_forward_megamoe", return_value="done") as dispatch:
            self.assertEqual(value.forward(torch.zeros(8, 5120), ids, weights, layer_id=17), "done")
        torch.testing.assert_close(dispatch.call_args.args[1], expected[0])
        value._processed[(17, torch.device("cpu"))] = object()
        value._mega_capacity = 8
        with patch.object(value, "_forward_megamoe", return_value="done") as dispatch:
            value.forward_shared_graph(torch.zeros(8, 5120), ids, weights, layer_id=17)
        torch.testing.assert_close(dispatch.call_args.args[1], expected[0])

    @torch.no_grad()
    def test_prefill_pattern_prefix_masks_padding_and_private_outputs(self):
        for rank in range(8):
            value = engine(rank, backend='megamoe')
            receipt = prepare_double(value)
            value._fake_balance_op = native_double(value)
            with patch.object(value, '_prefill_fake_device_eligible', return_value=True), patch.object(
                moe, '_get_triton_kernels', return_value=kernel_double(),
            ):
                for rows in (1, 7, 127, 128, 3):
                    ids, weights = routes(rows)
                    ids[::2, 1] = -7
                    ids[0, 3] = torch.iinfo(torch.int64).min
                    ids[:, 6:] = 93  # The old chain discards even nonnegative padding.
                    weights[:, :6] = float('nan')
                    before = ids.clone(), weights.clone()
                    actual = value._try_prefill_fake_balance(ids, weights, receipt)
                    expected = engine(rank)._apply_fake_balance(ids, weights)
                    for new, old in zip(actual, expected):
                        self.assertTrue(torch.equal(new.view(torch.uint8), old.view(torch.uint8)))
                        self.assertFalse(torch._C._is_alias_of(new, ids))
                        self.assertFalse(torch._C._is_alias_of(new, weights))
                    self.assertTrue(torch.equal(ids, before[0]))
                    self.assertTrue(torch.equal(weights.view(torch.uint8), before[1].view(torch.uint8)))
                self.assertEqual(value._fake_balance_op.call_count, 1)
                self.assertEqual(value._prefill_fake_pattern_builds, 1)
                self.assertEqual(value._prefill_fake_pattern_calls, 5)
                # A new forward reuses only the pattern; input masks are read again.
                value.finish_dp_prefill_forward(receipt)
                value._dp_prepare_epoch += 1
                receipt = replace(receipt, epoch=value._dp_prepare_epoch)
                value._dp_active_preparation = receipt
                ids, weights = routes(3)
                ids.fill_(-9)
                actual = value._try_prefill_fake_balance(ids, weights, receipt)
                self.assertTrue(torch.all(actual[0][:, :6] == -9))
                self.assertTrue(torch.all(actual[1] == 0))
                self.assertEqual(value._fake_balance_op.call_count, 1)

    @torch.inference_mode()
    def test_prefill_pattern_versions_growth_close_and_expired_receipt(self):
        value = engine(backend='megamoe')
        receipt = prepare_double(value)
        value._fake_balance_op = native_double(value)
        ids, weights = routes()
        with patch.object(value, '_prefill_fake_device_eligible', return_value=True), patch.object(
            moe, '_get_triton_kernels', return_value=kernel_double(),
        ):
            value._try_prefill_fake_balance(ids, weights, receipt)
            pattern = value._prefill_fake_pattern
            self.assertFalse(pattern.ids.is_inference())
            self.assertFalse(pattern.weights.is_inference())
            pattern.ids.add_(1).sub_(1)
            self.assertIsNone(value._try_prefill_fake_balance(ids, weights, receipt))
            self.assertEqual(value._fake_balance_op.call_count, 1, 'a mutated generation must not remint')
            with patch('torch.cuda.synchronize'):
                value.close()
            self.assertIsNone(value._prefill_fake_pattern)
            receipt = prepare_double(value, 256, generation=2, epoch=2)
            value._try_prefill_fake_balance(ids, weights, receipt)
            self.assertIsNot(value._prefill_fake_pattern, pattern)
            self.assertEqual(value._prefill_fake_pattern_builds, 2)
            current = value._prefill_fake_pattern
            value._prefill_fake_pattern = replace(current, ids=current.ids.clone())
            self.assertIsNone(value._try_prefill_fake_balance(ids, weights, receipt))
            self.assertEqual(value._prefill_fake_pattern_builds, 2)
            value._prefill_fake_pattern = None
            self.assertIsNone(value._try_prefill_fake_balance(ids, weights, receipt))
            self.assertEqual(value._prefill_fake_pattern_builds, 2)
            value.finish_dp_prefill_forward(receipt)
            with self.assertRaisesRegex(ValueError, 'stale or insufficient'):
                value._try_prefill_fake_balance(ids, weights, receipt)

    @torch.no_grad()
    def test_prefill_admission_declines_before_native_and_launch_errors_do_not_retry(self):
        value = engine(backend='megamoe')
        receipt = prepare_double(value)
        ids, weights = routes()
        for variant in ('no-receipt', 'noncontiguous', 'int32', 'empty', 'off', 'device', 'grad'):
            value._fake_balance_config = (variant != 'off', 6, 0, 8)
            test_ids, test_weights = (ids[::2], weights[::2]) if variant == 'noncontiguous' else (ids, weights)
            if variant == 'int32':
                test_ids = ids.int()
            elif variant == 'empty':
                test_ids, test_weights = ids[:0], weights[:0]
            with self.subTest(variant=variant), patch.object(
                value, '_prefill_fake_device_eligible', return_value=variant != 'device',
            ), patch('torch.is_grad_enabled', return_value=variant == 'grad'), patch.object(
                moe, '_get_triton_kernels', side_effect=AssertionError('unsupported provider lookup'),
            ):
                self.assertIsNone(value._try_prefill_fake_balance(
                    test_ids, test_weights, None if variant == 'no-receipt' else receipt,
                ))
        value._fake_balance_config = (True, 6, 0, 8)
        value._fake_balance_op = native_double(value)
        kernels = kernel_double()
        kernels.mask_fake_routes = Mock()
        kernels.mask_fake_routes.__getitem__ = Mock(return_value=Mock(side_effect=RuntimeError('pack launch failed')))
        with patch.object(value, '_prefill_fake_device_eligible', return_value=True), patch.object(
            moe, '_get_triton_kernels', return_value=kernels,
        ), patch.object(value, '_apply_fake_balance', side_effect=AssertionError('retry after launch')):
            with self.assertRaisesRegex(RuntimeError, 'pack launch failed'):
                value.forward(torch.zeros(8, 5120), ids, weights, prepared_buffer=receipt)

    @torch.no_grad()
    def test_prefill_fused_result_still_precedes_eplb_and_native_dispatch(self):
        value = engine(backend='megamoe')
        receipt = prepare_double(value)
        ids, weights = routes()
        balanced = engine()._apply_fake_balance(ids, weights)
        physical = balanced[0] + 384
        with patch.object(value, '_try_prefill_fake_balance', return_value=balanced), patch.object(
            value, '_apply_fake_balance', side_effect=AssertionError('duplicate override'),
        ), patch.object(value, '_map_expert_ids', return_value=physical) as remap, patch.object(
            value, '_forward_megamoe', return_value='done',
        ) as dispatch:
            self.assertEqual(value.forward(torch.zeros(8, 5120), ids, weights, layer_id=3,
                                           prepared_buffer=receipt), 'done')
        self.assertIs(remap.call_args.args[0], balanced[0])
        self.assertIs(remap.call_args.args[1], balanced[1])
        self.assertIs(dispatch.call_args.args[1], physical)
        self.assertIs(dispatch.call_args.args[2], balanced[1])
        self.assertIs(dispatch.call_args.kwargs['prepared_buffer'], receipt.buffer)


if __name__ == "__main__":
    unittest.main()
