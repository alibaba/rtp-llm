"""CPU-only contracts for the actual compact Mega mHC wrapper.

Only native-dependent unit imports and backend operations are stubbed. Fake
CUDA tensors exercise metadata gates without initializing a CUDA context.
These tests prove dispatch/storage contracts, not GPU arithmetic equivalence.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


class DelayedHCUnit:
    pass


class RMSNorm:
    pass


RMSNorm.__module__ = "rtp_llm.models_py.modules.base.cuda.norm"


def _load():
    delayed = ModuleType("rtp_llm.models_py.modules.dsv4.hc.delayed")
    delayed.DelayedHCUnit = DelayedHCUnit
    prenorm = ModuleType("rtp_llm.models_py.modules.dsv4.hc.v41_prenorm")
    prenorm.prepare_tf32_weight = lambda weight: weight
    path = Path(__file__).resolve().parents[1] / "hc/v41_mega_mhc.py"
    spec = importlib.util.spec_from_file_location("compact_mega_mhc_contract", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules, {delayed.__name__: delayed, prenorm.__name__: prenorm}
    ):
        spec.loader.exec_module(module)
    return module


op = _load()


def _units(device="cpu"):
    previous, following = DelayedHCUnit(), DelayedHCUnit()
    for unit in (previous, following):
        unit.dim, unit.hc_mult = 5120, 4
        unit.hc_sinkhorn_iters = 20
        unit.norm_eps = unit.hc_eps = 1e-6
        unit.pre_mix_out = None
    following.fn = torch.empty(24, 20480, device=device)
    following.scale = torch.ones(3, device=device)
    following.base = torch.zeros(24, device=device)
    following._previous_ref = lambda: previous
    norm = RMSNorm()
    norm.weight = torch.ones(5120, dtype=torch.bfloat16, device=device)
    norm.variance_epsilon = 1e-6
    return previous, following, norm


def _case(leading):
    previous, following, norm = _units()
    tokens = math.prod(leading)
    previous.pre_mix_out = torch.arange(tokens * 4).float().view(*leading, 4)
    return dict(
        attn_out=torch.randn(*leading, 5120, dtype=torch.bfloat16),
        residual=torch.randn(*leading, 4, 5120, dtype=torch.bfloat16),
        post=torch.randn(*leading, 4, 1),
        comb=torch.randn(*leading, 4, 4),
        previous=previous,
        next_hc=following,
        norm=norm,
    )


class CompactMegaMHCContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Torch's test machinery before temporary sys.modules patches.
        with FakeTensorMode():
            pass

    def setUp(self):
        self.backend = SimpleNamespace(
            get_num_sms=Mock(return_value=152),
            set_num_sms=Mock(side_effect=AssertionError("global policy mutation")),
            use_deterministic_algorithms=Mock(
                side_effect=AssertionError("global policy mutation")
            ),
        )
        norm_module = ModuleType(RMSNorm.__module__)
        norm_module.RMSNorm = RMSNorm
        self.modules = patch.dict(
            sys.modules, {"deep_gemm": self.backend, RMSNorm.__module__: norm_module}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.next_hc = SimpleNamespace(norm_eps=1e-6, hc_eps=1e-6)
        self.norm = SimpleNamespace(variance_epsilon=1e-6)

    def rows(self, original, compact):
        return op._compact_execution_tokens(original, compact, self.next_hc, self.norm)

    def test_152_sm_split_regimes_and_no_padding_when_splits_match(self):
        for original, compact, expected in (
            (34, 32, 32),
            (128, 32, 32),
            (192, 32, 32),
            (193, 32, 256),
            (256, 32, 256),
            (320, 32, 256),
            (321, 32, 384),
            (384, 32, 384),
            (448, 32, 384),
            (449, 32, 512),
            (4096, 32, 512),
            (32768, 32, 512),
            (1 << 20, 32, 512),
            (4096, 1024, 1024),
            (1024, 1024, 1024),
        ):
            with self.subTest(original=original, compact=compact):
                self.assertEqual(self.rows(original, compact), expected)

    def test_live_sm_count_and_bounded_representatives_match_source_heuristic(self):
        def splits(rows, sms):
            maximum = min(64, max(16, sms // math.ceil(rows / 64)))
            return math.ceil(80 / math.ceil(80 / maximum))

        for sms in (16, 80, 132, 148, 152, 160, 192, 1024):
            self.backend.get_num_sms.return_value = sms
            for original in range(32, 2049):
                actual = self.rows(original, 32)
                if actual is not None:
                    self.assertLessEqual(actual, 512)
                    self.assertGreaterEqual(actual, 32)
                    self.assertEqual(splits(actual, sms), splits(original, sms))
                else:
                    self.assertTrue(
                        all(
                            splits(rows, sms) != splits(original, sms)
                            for rows in (32, *range(64, 513, 64))
                        )
                    )
        self.backend.get_num_sms.return_value = 1024
        self.assertIsNone(self.rows(32768, 32))
        self.backend.get_num_sms.return_value = 16
        self.assertEqual(self.rows(32768, 32), 32)

    def test_invalid_counts_or_unavailable_sms_fall_back(self):
        for original, compact in (
            (0, 0),
            (31, 32),
            (True, 1),
            (33, 0),
            (34.0, 32),
            ((1 << 20) + 1, 32),
        ):
            with self.subTest(original=original, compact=compact):
                self.assertIsNone(self.rows(original, compact))
        for sms in (0, -1, None, True):
            self.backend.get_num_sms.return_value = sms
            self.assertIsNone(self.rows(4096, 32))
        self.backend.get_num_sms = None
        self.assertIsNone(self.rows(4096, 32))

    def test_zero_epsilon_only_rejects_new_zero_padding(self):
        for owner, attr in (
            (self.next_hc, "norm_eps"),
            (self.next_hc, "hc_eps"),
            (self.norm, "variance_epsilon"),
        ):
            with self.subTest(attr=attr), patch.object(owner, attr, 0.0):
                self.assertIsNone(self.rows(4096, 32))
                self.assertEqual(self.rows(34, 32), 32)

    def test_eligibility_before_pre_mix_exists_and_capture_stream_gate(self):
        with FakeTensorMode(), torch.inference_mode():
            previous, following, norm = _units("cuda:0")
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ), patch.object(torch.cuda, "device"), patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ) as capturing, patch.object(
                op, "_get_mega_mhc", return_value=Mock()
            ), patch.object(
                op, "_stream_key", return_value=(0, 123)
            ), patch.object(
                op, "_WARMED_STREAMS", set()
            ):
                args = (previous, following, norm)
                self.assertTrue(
                    op.can_preserve_compact_mhc(
                        *args, original_tokens=4096, compact_tokens=32
                    )
                )
                self.assertIsNone(previous.pre_mix_out)
                self.assertIsNone(following.pre_mix_out)
                capturing.return_value = True
                self.assertFalse(
                    op.can_preserve_compact_mhc(
                        *args, original_tokens=4096, compact_tokens=32
                    )
                )
                op._WARMED_STREAMS.add((0, 123))
                self.assertTrue(
                    op.can_preserve_compact_mhc(
                        *args, original_tokens=4096, compact_tokens=32
                    )
                )

    def test_eligibility_rejects_unfused_contracts_without_allocation(self):
        with FakeTensorMode(), torch.inference_mode():
            previous, following, norm = _units("cuda:0")
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ), patch.object(torch.cuda, "device"), patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ), patch.object(
                op, "_get_mega_mhc", return_value=Mock()
            ) as backend:

                def eligible():
                    return op.can_preserve_compact_mhc(
                        previous,
                        following,
                        norm,
                        original_tokens=4096,
                        compact_tokens=32,
                    )

                self.assertTrue(eligible())
                for owner, attr, value in (
                    (following, "_previous_ref", None),
                    (following, "fn", None),
                    (previous, "dim", 4096),
                    (following, "hc_mult", 8),
                    (following, "hc_sinkhorn_iters", 0),
                    (norm, "variance_epsilon", -1),
                    (following, "norm_eps", 0),
                    (following, "scale", following.scale.to(torch.bfloat16)),
                ):
                    with self.subTest(attr=attr), patch.object(owner, attr, value):
                        self.assertFalse(eligible())
                backend.return_value = None
                self.assertFalse(eligible())

    @torch.inference_mode()
    def test_padding_zero_initialization_and_output_storage_without_base(self):
        for original, rows, leading in (
            (34, 32, (32,)),
            (193, 256, (4, 8)),
            (321, 384, (32,)),
            (4096, 512, (32,)),
        ):
            with self.subTest(original=original):
                case = _case(leading)
                previous_pre = case["previous"].pre_mix_out
                source = [
                    case[name] for name in ("attn_out", "residual", "post", "comb")
                ] + [previous_pre]
                saved = [tensor.clone() for tensor in source]
                output_buffers = []

                def operation(**kwargs):
                    self.assertIs(case["previous"].pre_mix_out, previous_pre)
                    self.assertEqual(kwargs["x"].shape[0], rows)
                    for key, value in zip(
                        (
                            "x",
                            "residual",
                            "post_mix",
                            "comb_res_mix",
                            "shifted_prev_mix",
                        ),
                        source,
                    ):
                        actual = kwargs[key]
                        self.assertTrue(
                            torch.equal(actual[:32].flatten(), value.flatten())
                        )
                        self.assertEqual(torch.count_nonzero(actual[32:]).item(), 0)
                        if rows == 32:
                            self.assertEqual(actual.data_ptr(), value.data_ptr())
                    for dest, src in (
                        ("new_residual", "residual"),
                        ("y_bf16", "x"),
                        ("new_post_mix", "post_mix"),
                        ("new_comb_res_mix", "comb_res_mix"),
                        ("new_prev_mix", "shifted_prev_mix"),
                    ):
                        kwargs[dest].copy_(kwargs[src])
                        output_buffers.append(kwargs[dest])

                with patch.object(op, "is_supported", return_value=True), patch.object(
                    op, "_get_mega_mhc", return_value=operation
                ), patch.object(torch.cuda, "device"), patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ), patch.object(
                    op, "_stream_key", return_value=(0, 123)
                ), patch.object(
                    op, "_WARMED_STREAMS", set()
                ):
                    result = op.try_fused_post_pre(**case, original_tokens=original)
                actual = (*result, case["next_hc"].pre_mix_out)
                expected = (
                    case["residual"],
                    case["attn_out"],
                    case["post"],
                    case["comb"],
                    previous_pre,
                )
                for got, ref, buffer in zip(actual, expected, output_buffers):
                    self.assertEqual(got.shape, ref.shape)
                    self.assertTrue(torch.equal(got, ref))
                    self.assertIsNone(got._base)
                    self.assertEqual(
                        got.untyped_storage().data_ptr(),
                        buffer.untyped_storage().data_ptr(),
                    )
                    self.assertEqual(
                        got.untyped_storage().nbytes(),
                        buffer.untyped_storage().nbytes(),
                    )
                self.assertIs(case["previous"].pre_mix_out, previous_pre)
                for value, before in zip(source, saved):
                    self.assertTrue(torch.equal(value, before))

    def test_failure_and_cold_fallback_do_not_publish_mixes(self):
        case = _case((32,))
        previous_pre = case["previous"].pre_mix_out
        sentinel = object()
        case["next_hc"].pre_mix_out = sentinel
        with patch.object(op, "is_supported", return_value=True), patch.object(
            op,
            "_get_mega_mhc",
            return_value=Mock(side_effect=RuntimeError("backend failure")),
        ) as backend, patch.object(torch.cuda, "device"), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "backend failure"):
                op.try_fused_post_pre(**case, original_tokens=4096)
            self.assertIs(case["previous"].pre_mix_out, previous_pre)
            self.assertIs(case["next_hc"].pre_mix_out, sentinel)
            backend.return_value.reset_mock()
            with patch.object(
                torch, "empty_like", side_effect=AssertionError("allocation")
            ):
                self.assertIsNone(op.try_fused_post_pre(**case, original_tokens=31))
            backend.return_value.assert_not_called()
            self.assertIs(case["next_hc"].pre_mix_out, sentinel)

    def test_default_call_does_not_query_split_policy(self):
        case = _case((32,))
        with patch.object(op, "is_supported", return_value=True), patch.object(
            op, "_get_mega_mhc", return_value=Mock()
        ), patch.object(torch.cuda, "device"), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ), patch.object(
            op, "_stream_key", return_value=(0, 123)
        ), patch.object(
            op, "_WARMED_STREAMS", set()
        ):
            self.assertIsNotNone(op.try_fused_post_pre(**case))
        self.backend.get_num_sms.assert_not_called()
        self.backend.set_num_sms.assert_not_called()
        self.backend.use_deterministic_algorithms.assert_not_called()


if __name__ == "__main__":
    unittest.main()
