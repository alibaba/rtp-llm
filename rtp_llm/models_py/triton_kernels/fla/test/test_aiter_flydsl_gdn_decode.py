import inspect
import itertools
import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.fla import aiter_flydsl_decode as adapter
from rtp_llm.models_py.triton_kernels.fla import utils as fla_utils
from rtp_llm.models_py.triton_kernels.fla.aiter_flydsl_decode import (
    AiterFlydslGdnDecodeStateMetadata,
    aiter_flydsl_gdn_decode,
    copy_aiter_flydsl_gdn_decode_state_at_block_boundary,
    is_aiter_flydsl_gdn_decode_supported,
    prepare_aiter_flydsl_gdn_decode_state_indices,
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    cal_block_idx,
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating


def _make_state_metadata(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
    *,
    host_sequence_lengths: torch.Tensor | None = None,
    state_pool_size: int,
) -> AiterFlydslGdnDecodeStateMetadata:
    return AiterFlydslGdnDecodeStateMetadata(
        block_map=block_map,
        sequence_lengths_plus_1=sequence_lengths_plus_1,
        seq_size_per_block=seq_size_per_block,
        host_sequence_lengths=host_sequence_lengths,
        state_pool_size=state_pool_size,
    )


_prepare_decode_state_indices = prepare_aiter_flydsl_gdn_decode_state_indices


def _prepare_decode_state_indices_from_tensors(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
    host_sequence_lengths: torch.Tensor | None = None,
    *,
    state_pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    read_indices, write_indices, _ = _prepare_decode_state_indices(
        _make_state_metadata(
            block_map,
            sequence_lengths_plus_1,
            seq_size_per_block,
            host_sequence_lengths=host_sequence_lengths,
            state_pool_size=state_pool_size,
        )
    )
    return read_indices, write_indices


# Keep tensor-based test setup concise while exercising the single production
# metadata entry point above.
prepare_aiter_flydsl_gdn_decode_state_indices = (
    _prepare_decode_state_indices_from_tensors
)


def _mock_flydsl_decode():
    return mock.create_autospec(adapter._flydsl_gdr_decode_contract)


def _reset_adapter_process_state(test_case: unittest.TestCase) -> None:
    cached_functions = (
        adapter._is_aiter_flydsl_gdn_decode_disabled,
        adapter._warn_host_validation_unavailable_once,
        adapter._get_aiter_flydsl_gdn_decode,
    )
    for cached_function in cached_functions:
        cached_function.cache_clear()
        test_case.addCleanup(cached_function.cache_clear)
    adapter._LOGGED_BACKEND_DECISIONS.clear()
    test_case.addCleanup(adapter._LOGGED_BACKEND_DECISIONS.clear)
    adapter._WARMED_DECODE_SIGNATURES.clear()
    test_case.addCleanup(adapter._WARMED_DECODE_SIGNATURES.clear)


def _make_decode_inputs(
    *,
    batch: int = 2,
    query_length: int = 1,
    key_heads: int = 2,
    value_heads: int = 8,
    dim: int = 128,
    state_dtype: torch.dtype = torch.float32,
):
    # Match the production layout: q/k/v are cross-strided views of one
    # mixed_qkv allocation split along the head dimension.
    mixed_qkv = torch.randn(
        batch,
        query_length,
        key_heads * 2 + value_heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = torch.split(mixed_qkv, [key_heads, key_heads, value_heads], dim=2)
    mixed_ba = torch.randn(
        batch * query_length,
        value_heads * 2,
        device="cuda",
        dtype=torch.bfloat16,
    )
    a, b = torch.split(mixed_ba, value_heads, dim=-1)
    state = torch.randn(
        batch * 3 + 1,
        value_heads,
        dim,
        dim,
        device="cuda",
        dtype=state_dtype,
    )
    return q, k, v, a, b, state


class FlaEnvironmentFlagTest(unittest.TestCase):
    def test_env_flag_parses_true_false_and_default_values(self):
        for value in ("1", "true", "TRUE", "on", "yes"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"TEST_FLA_FLAG": value}
            ):
                self.assertTrue(fla_utils.env_flag("TEST_FLA_FLAG"))

        for value in ("0", "off", "false", "", "2"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"TEST_FLA_FLAG": value}
            ):
                self.assertFalse(fla_utils.env_flag("TEST_FLA_FLAG"))

        with mock.patch.dict(os.environ):
            os.environ.pop("TEST_FLA_FLAG", None)
            self.assertFalse(fla_utils.env_flag("TEST_FLA_FLAG"))
            self.assertTrue(fla_utils.env_flag("TEST_FLA_FLAG", default="true"))


@unittest.skipUnless(torch.cuda.is_available(), "A CUDA/HIP device is required")
class AiterFlydslGdnDecodeCommonTest(unittest.TestCase):
    def setUp(self):
        _reset_adapter_process_state(self)

    def test_prepare_decode_indices_honors_noncontiguous_block_map_stride(self):
        padded_block_map = torch.tensor(
            [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]],
            device="cuda",
            dtype=torch.int32,
        )
        block_map = padded_block_map[:, :3]
        self.assertEqual(block_map.stride(), (5, 1))
        sequence_lengths_plus_1 = torch.tensor(
            [1002, 1025], device="cuda", dtype=torch.int32
        )

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, sequence_lengths_plus_1, 1024, state_pool_size=100
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, 4])
        self.assertEqual(write_indices.cpu().tolist(), [1, 5])

    def test_prepare_decode_indices_maps_padding_and_out_of_range_rows_to_dummy(self):
        block_map = torch.tensor(
            [[1, 2], [0, 0], [3, 4]],
            device="cuda",
            dtype=torch.int32,
        )
        sequence_lengths_plus_1 = torch.tensor(
            [1025, 0, 4097], device="cuda", dtype=torch.int32
        )

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, sequence_lengths_plus_1, 1024, state_pool_size=100
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, 0, 0])
        self.assertEqual(write_indices.cpu().tolist(), [2, 0, 0])

    def test_prepare_decode_indices_flags_real_errors_but_not_padding(self):
        metadata = _make_state_metadata(
            torch.tensor(
                [[1, 2], [0, 0], [3, 4]],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor([1025, 0, 4097], device="cuda", dtype=torch.int32),
            1024,
            state_pool_size=100,
        )

        _, _, invalid_row_flags = _prepare_decode_state_indices(metadata)
        torch.cuda.synchronize()

        self.assertEqual(invalid_row_flags.cpu().tolist(), [0, 0, 1])

    def test_prepare_decode_indices_maps_dummy_read_and_real_write_to_dummy(self):
        block_map = torch.tensor([[0, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1025], device="cuda", dtype=torch.int32)

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024, state_pool_size=3
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.item(), 0)
        self.assertEqual(write_indices.item(), 0)

    def test_prepare_decode_indices_maps_state_pool_overflow_to_dummy(self):
        block_map = torch.tensor([[1, 7]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1025], device="cuda", dtype=torch.int32)

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024, state_pool_size=3
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.item(), 0)
        self.assertEqual(write_indices.item(), 0)

    def test_prepare_decode_indices_matches_recurrent_block_formula(self):
        block_size = 1024
        lengths = torch.tensor(
            [1, 2, 1023, 1024, 1025, 2048, 2049],
            device="cuda",
            dtype=torch.int32,
        )
        block_map = torch.arange(
            1,
            lengths.numel() * 3 + 1,
            device="cuda",
            dtype=torch.int32,
        ).reshape(lengths.numel(), 3)

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, block_size, state_pool_size=100
        )
        torch.cuda.synchronize()

        expected_read = []
        expected_write = []
        for row, length in enumerate(lengths.cpu().tolist()):
            if length < 2:
                expected_read.append(0)
                expected_write.append(0)
                continue
            # Call the production Triton helper's Python body so this
            # equivalence test cannot silently drift from the reference path.
            read_pos = cal_block_idx.fn(length - 1, block_size)
            write_pos = cal_block_idx.fn(length, block_size)
            expected_read.append(block_map[row, read_pos].item())
            expected_write.append(block_map[row, write_pos].item())
        self.assertEqual(read_indices.cpu().tolist(), expected_read)
        self.assertEqual(write_indices.cpu().tolist(), expected_write)

    def test_prepare_decode_indices_accepts_empty_batch(self):
        block_map = torch.empty((0, 1), device="cuda", dtype=torch.int32)
        lengths = torch.empty((0,), device="cuda", dtype=torch.int32)

        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024, state_pool_size=1
        )

        self.assertEqual(read_indices.numel(), 0)
        self.assertEqual(write_indices.numel(), 0)

    def test_prepare_decode_indices_rejects_real_out_of_range_request(self):
        block_map = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([4097], device="cuda", dtype=torch.int32)
        host_lengths = torch.tensor([4096], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "exceeds block-map width"):
            prepare_aiter_flydsl_gdn_decode_state_indices(
                block_map,
                lengths,
                1024,
                host_sequence_lengths=host_lengths,
                state_pool_size=3,
            )

    def test_prepare_decode_indices_warns_when_host_validation_is_unavailable(self):
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([2], device="cuda", dtype=torch.int32)

        with self.assertLogs(adapter._LOGGER.name, level="WARNING") as logs:
            prepare_aiter_flydsl_gdn_decode_state_indices(
                block_map, lengths, 1024, state_pool_size=2
            )

        self.assertIn("validation is disabled", "\n".join(logs.output))

    def test_prepare_decode_indices_requires_state_pool_size(self):
        block_map = torch.ones((1, 1), device="cuda", dtype=torch.int32)
        lengths = torch.ones((1,), device="cuda", dtype=torch.int32)

        with self.assertRaisesRegex(TypeError, "state_pool_size"):
            prepare_aiter_flydsl_gdn_decode_state_indices(block_map, lengths, 1024)

    def test_prepare_decode_indices_validates_arguments(self):
        valid_map = torch.ones((2, 2), device="cuda", dtype=torch.int32)
        valid_lengths = torch.ones((2,), device="cuda", dtype=torch.int32)
        strided_lengths = torch.ones((4,), device="cuda", dtype=torch.int32)[::2]
        cases = (
            (
                valid_map.reshape(-1),
                valid_lengths,
                1024,
                "block_map must be 2D",
            ),
            (
                torch.empty((2, 0), device="cuda", dtype=torch.int32),
                valid_lengths,
                1024,
                "at least one block column",
            ),
            (
                valid_map.to(torch.int64),
                valid_lengths,
                1024,
                "block_map must be int32",
            ),
            (
                valid_map,
                valid_lengths.reshape(1, 2),
                1024,
                "sequence_lengths_plus_1 must be 1D",
            ),
            (
                valid_map,
                valid_lengths.to(torch.int64),
                1024,
                "sequence_lengths_plus_1 must be int32",
            ),
            (
                valid_map,
                strided_lengths,
                1024,
                "sequence_lengths_plus_1 must be contiguous",
            ),
            (
                valid_map.cpu(),
                valid_lengths,
                1024,
                "must be on the same device",
            ),
            (
                valid_map,
                valid_lengths[:1],
                1024,
                "count must equal",
            ),
            (
                valid_map,
                valid_lengths,
                0,
                "seq_size_per_block must be positive",
            ),
            (
                valid_map[:, ::2],
                valid_lengths,
                1024,
                "columns must be contiguous",
            ),
        )
        for block_map, lengths, block_size, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                prepare_aiter_flydsl_gdn_decode_state_indices(
                    block_map, lengths, block_size, state_pool_size=100
                )
        with self.assertRaisesRegex(ValueError, "state_pool_size must be positive"):
            prepare_aiter_flydsl_gdn_decode_state_indices(
                valid_map, valid_lengths, 1024, state_pool_size=0
            )

    def test_aiter_symbol_resolver_falls_back_on_runtime_import_failure(self):
        original_import = __import__

        def import_with_target_failure(name, *args, **kwargs):
            if name == "aiter.ops.flydsl.linear_attention_kernels":
                raise RuntimeError("extension load failure")
            return original_import(name, *args, **kwargs)

        with self.assertLogs(adapter._LOGGER.name, level="WARNING") as logs, mock.patch(
            "builtins.__import__", side_effect=import_with_target_failure
        ):
            self.assertIsNone(adapter._get_aiter_flydsl_gdn_decode())

        self.assertIn("extension load failure", "\n".join(logs.output))

    def test_aiter_signature_accepts_optional_extension_and_rejects_required_one(self):
        expected = inspect.signature(adapter._flydsl_gdr_decode_contract)
        optional_extension = expected.replace(
            parameters=[
                *expected.parameters.values(),
                inspect.Parameter(
                    "future_option",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                ),
            ]
        )
        required_extension = expected.replace(
            parameters=[
                *expected.parameters.values(),
                inspect.Parameter(
                    "future_required",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                ),
            ]
        )

        self.assertIsNone(
            adapter._callable_signature_incompatibility(expected, optional_extension)
        )
        self.assertIn(
            "unexpected required parameter",
            adapter._callable_signature_incompatibility(expected, required_extension),
        )

    def test_shape_gate_accepts_production_stride_and_rejects_invalid_inputs(self):
        valid = _make_decode_inputs()
        q, k, v, a, b, state = valid
        A_log = torch.randn(v.shape[2], device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v.shape[2], device="cuda", dtype=torch.bfloat16)
        self.assertFalse(q.is_contiguous())
        self.assertFalse(k.is_contiguous())
        self.assertFalse(v.is_contiguous())

        with (
            mock.patch.object(adapter, "is_amd_cdna3", True),
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
        ):
            self.assertTrue(
                is_aiter_flydsl_gdn_decode_supported(
                    *valid, A_log=A_log, dt_bias=dt_bias
                )
            )

            unaligned_storage = torch.empty(
                state.numel() + 1, device="cuda", dtype=state.dtype
            )
            unaligned_state = unaligned_storage[1:].view_as(state)
            row_elements = state[0].numel()
            row_strided_storage = torch.empty(
                (state.shape[0], row_elements + 1),
                device="cuda",
                dtype=state.dtype,
            )
            row_strided_state = row_strided_storage[:, :row_elements].view_as(state)
            cases = (
                ("q-rank", (q.squeeze(1), k, v, a, b, state)),
                (
                    "multi-token",
                    _make_decode_inputs(query_length=2),
                ),
                (
                    "shape-not-whitelisted",
                    _make_decode_inputs(key_heads=4, value_heads=8),
                ),
                ("q-fp16", (q.to(torch.float16), k, v, a, b, state)),
                ("k-shape", (q, k[:, :, :1], v, a, b, state)),
                ("v-shape", (q, k, v[:, :0], a, b, state)),
                ("state-fp16", (q, k, v, a, b, state.to(torch.float16))),
                ("state-unaligned", (q, k, v, a, b, unaligned_state)),
                ("state-row-stride", (q, k, v, a, b, row_strided_state)),
                (
                    "state-inner-stride",
                    (q, k, v, a, b, state.transpose(-1, -2)),
                ),
                ("a-shape", (q, k, v, a.reshape(1, -1), b, state)),
            )
            for name, inputs in cases:
                with self.subTest(name=name):
                    self.assertFalse(
                        is_aiter_flydsl_gdn_decode_supported(
                            *inputs, A_log=A_log, dt_bias=dt_bias
                        )
                    )

            auxiliary_cases = (
                ("A-log-dtype", A_log.to(torch.float16), dt_bias, None),
                ("A-log-count", A_log[:1], dt_bias, None),
                ("A-log-rank", A_log.reshape(1, -1), dt_bias, None),
                ("dt-bias-dtype", A_log, dt_bias.to(torch.float32), None),
                ("dt-bias-count", A_log, dt_bias[:1], None),
                ("dt-bias-rank", A_log, dt_bias.reshape(1, -1), None),
                ("scale", A_log, dt_bias, 1.0),
            )
            for name, case_A_log, case_dt_bias, scale in auxiliary_cases:
                with self.subTest(name=name):
                    self.assertFalse(
                        is_aiter_flydsl_gdn_decode_supported(
                            *valid,
                            A_log=case_A_log,
                            dt_bias=case_dt_bias,
                            scale=scale,
                        )
                    )

            with mock.patch.object(
                adapter, "_get_aiter_flydsl_gdn_decode", return_value=None
            ):
                self.assertFalse(
                    is_aiter_flydsl_gdn_decode_supported(
                        *valid, A_log=A_log, dt_bias=dt_bias
                    )
                )

    def test_shape_gate_rejects_non_cdna3_empty_batch_and_kill_switch(self):
        valid = _make_decode_inputs()
        v = valid[2]
        A_log = torch.randn(v.shape[2], device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v.shape[2], device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(
            adapter, "_get_aiter_flydsl_gdn_decode", return_value=_mock_flydsl_decode()
        ):
            with mock.patch.object(adapter, "is_amd_cdna3", False):
                self.assertFalse(
                    is_aiter_flydsl_gdn_decode_supported(
                        *valid, A_log=A_log, dt_bias=dt_bias
                    )
                )
                with self.assertRaisesRegex(ValueError, "not AMD CDNA3"):
                    aiter_flydsl_gdn_decode(
                        A_log=A_log,
                        a=valid[3],
                        dt_bias=dt_bias,
                        q=valid[0],
                        k=valid[1],
                        v=valid[2],
                        b=valid[4],
                        state=valid[5],
                        read_indices=torch.ones(
                            valid[0].shape[0], device="cuda", dtype=torch.int32
                        ),
                        write_indices=torch.ones(
                            valid[0].shape[0], device="cuda", dtype=torch.int32
                        ),
                    )

            with mock.patch.object(adapter, "is_amd_cdna3", True):
                empty = _make_decode_inputs(batch=0)
                reason = adapter._aiter_flydsl_gdn_decode_unsupported_reason(
                    *empty, A_log=A_log, dt_bias=dt_bias, scale=None
                )
                self.assertIn("batch>=1", reason)

                with mock.patch.dict(
                    os.environ, {"DISABLE_AITER_FLYDSL_GDN_DECODE": "1"}
                ):
                    adapter._is_aiter_flydsl_gdn_decode_disabled.cache_clear()
                    reason = adapter._aiter_flydsl_gdn_decode_unsupported_reason(
                        *valid, A_log=A_log, dt_bias=dt_bias, scale=None
                    )
                    self.assertIn("DISABLE_AITER_FLYDSL_GDN_DECODE", reason)

    @unittest.skipIf(torch.version.hip is not None, "NVIDIA-only dispatch guard")
    def test_real_nvidia_device_rejects_aiter_dispatch(self):
        valid = _make_decode_inputs()
        value_heads = valid[2].shape[2]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)

        self.assertFalse(
            is_aiter_flydsl_gdn_decode_supported(*valid, A_log=A_log, dt_bias=dt_bias)
        )
        reason = adapter._aiter_flydsl_gdn_decode_unsupported_reason(
            *valid, A_log=A_log, dt_bias=dt_bias, scale=None
        )
        self.assertIn("not AMD CDNA3", reason)

    def test_shape_gate_rejects_invalid_decode_state_metadata(self):
        valid = _make_decode_inputs()
        value_heads = valid[2].shape[2]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        block_map = torch.ones((2, 2), device="cuda", dtype=torch.int64)
        lengths = torch.ones((2,), device="cuda", dtype=torch.int32)

        with (
            mock.patch.object(adapter, "is_amd_cdna3", True),
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
        ):
            self.assertFalse(
                is_aiter_flydsl_gdn_decode_supported(
                    *valid,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    block_map=block_map,
                    sequence_lengths_plus_1=lengths,
                    seq_size_per_block=1024,
                    state_pool_size=valid[5].shape[0],
                )
            )

    def test_optional_state_metadata_and_batch_mismatch(self):
        valid = _make_decode_inputs()
        value_heads = valid[2].shape[2]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        block_map = torch.ones((3, 2), device="cuda", dtype=torch.int32)
        lengths = torch.ones((3,), device="cuda", dtype=torch.int32)

        with (
            mock.patch.object(adapter, "is_amd_cdna3", True),
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
        ):
            self.assertTrue(
                is_aiter_flydsl_gdn_decode_supported(
                    *valid,
                    A_log=A_log,
                    dt_bias=dt_bias,
                )
            )
            self.assertFalse(
                is_aiter_flydsl_gdn_decode_supported(
                    *valid,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    block_map=block_map,
                )
            )
            with self.assertRaisesRegex(TypeError, "sequence_lengths_plus_1"):
                AiterFlydslGdnDecodeStateMetadata(block_map=block_map)

            # Decision logging is intentionally bounded to one selected and
            # one fallback message per process. Expose this fallback slot so
            # the test can assert the specific diagnostic below.
            adapter._LOGGED_BACKEND_DECISIONS.discard(False)
            with self.assertLogs(adapter._LOGGER.name, level="INFO") as logs:
                self.assertFalse(
                    is_aiter_flydsl_gdn_decode_supported(
                        *valid,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        block_map=block_map,
                        sequence_lengths_plus_1=lengths,
                        seq_size_per_block=1024,
                        state_pool_size=valid[5].shape[0],
                    )
                )
            self.assertIn(
                "block-map batch differs from decode input batch",
                "\n".join(logs.output),
            )

    def test_state_copy_launch_policy_is_derived_from_metadata(self):
        cases = (
            ("capture", torch.tensor([1001]), 1024, True, True),
            ("non-boundary", torch.tensor([1001, 1500]), 1024, False, False),
            ("boundary", torch.tensor([1024, 1500]), 1024, False, True),
            ("unknown", None, 1024, False, True),
            ("empty", torch.empty(0, dtype=torch.int32), 1024, False, True),
            ("invalid-block-size", torch.tensor([1001]), 0, False, True),
        )
        for name, host_lengths, block_size, is_capturing, expected in cases:
            with self.subTest(name=name):
                metadata = _make_state_metadata(
                    torch.ones((1, 1), device="cuda", dtype=torch.int32),
                    torch.ones(1, device="cuda", dtype=torch.int32),
                    block_size,
                    host_sequence_lengths=host_lengths,
                    state_pool_size=2,
                )
                self.assertEqual(
                    metadata.should_copy_state(is_capturing=is_capturing),
                    expected,
                )

    def test_copy_state_validates_inner_layout_and_index_shapes(self):
        state = torch.randn(3, 2, 8, 8, device="cuda", dtype=torch.float32)
        indices = torch.tensor([1], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "state must be 4D"):
            copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                state[0], indices, indices
            )
        with self.assertRaisesRegex(ValueError, "inner dimensions must be contiguous"):
            copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                state.transpose(-1, -2), indices, indices
            )
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                state, indices, torch.tensor([1, 2], device="cuda", dtype=torch.int32)
            )
        invalid_index_cases = (
            (indices.reshape(1, 1), "contiguous 1D int32"),
            (indices.to(torch.int64), "contiguous 1D int32"),
            (
                torch.tensor([1, 0], device="cuda", dtype=torch.int32)[::2],
                "contiguous 1D int32",
            ),
            (indices.cpu(), "same device"),
        )
        for invalid_indices, message in invalid_index_cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                    state, invalid_indices, invalid_indices
                )

    def test_copy_state_device_guards_leave_state_unchanged(self):
        initial = torch.randn(3, 2, 8, 8, device="cuda", dtype=torch.float32)
        cases = (
            (
                "out-of-range",
                torch.tensor([1], device="cuda", dtype=torch.int32),
                torch.tensor([8], device="cuda", dtype=torch.int32),
            ),
            (
                "same-block",
                torch.tensor([2], device="cuda", dtype=torch.int32),
                torch.tensor([2], device="cuda", dtype=torch.int32),
            ),
            (
                "empty",
                torch.empty(0, device="cuda", dtype=torch.int32),
                torch.empty(0, device="cuda", dtype=torch.int32),
            ),
        )
        for name, read_indices, write_indices in cases:
            with self.subTest(name=name):
                state = initial.clone()
                copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                    state, read_indices, write_indices
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(state, initial, rtol=0, atol=0)

    def test_copy_state_copies_all_heads_and_non_aligned_tail(self):
        state = torch.randn(4, 2, 129, 128, device="cuda", dtype=torch.float32)
        initial = state.clone()
        state[1, 0].fill_(11)
        state[1, 1].fill_(22)
        expected_source = state[1].clone()

        copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
            state,
            torch.tensor([1], device="cuda", dtype=torch.int32),
            torch.tensor([2], device="cuda", dtype=torch.int32),
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(state[2], expected_source, rtol=0, atol=0)
        torch.testing.assert_close(state[0], initial[0], rtol=0, atol=0)
        torch.testing.assert_close(state[3], initial[3], rtol=0, atol=0)

    def test_decode_entry_validates_auxiliary_arguments(self):
        q, k, v, a, b, state = _make_decode_inputs()
        value_heads = v.shape[2]
        valid_kwargs = {
            "A_log": torch.randn(value_heads, device="cuda", dtype=torch.float32),
            "a": a,
            "dt_bias": torch.randn(value_heads, device="cuda", dtype=torch.bfloat16),
            "q": q,
            "k": k,
            "v": v,
            "b": b,
            "state": state,
            "read_indices": torch.tensor([1, 4], device="cuda", dtype=torch.int32),
            "write_indices": torch.tensor([1, 4], device="cuda", dtype=torch.int32),
        }
        cases = (
            (
                {"A_log": valid_kwargs["A_log"].to(torch.float16)},
                "A_log layout or dtype is unsupported",
            ),
            (
                {"A_log": valid_kwargs["A_log"][:1]},
                "A_log must contain",
            ),
            (
                {"dt_bias": valid_kwargs["dt_bias"].to(torch.float32)},
                "dt_bias must be",
            ),
            (
                {"dt_bias": valid_kwargs["dt_bias"][:1]},
                "dt_bias must be",
            ),
            (
                {"read_indices": valid_kwargs["read_indices"].to(torch.int64)},
                "contiguous 1D int32 tensors",
            ),
            (
                {
                    "read_indices": torch.tensor(
                        [1, 0, 4, 0], device="cuda", dtype=torch.int32
                    )[::2]
                },
                "contiguous 1D int32 tensors",
            ),
            ({"scale": 1.0}, "scale must be"),
        )
        with (
            mock.patch.object(adapter, "is_amd_cdna3", True),
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
        ):
            for changes, message in cases:
                with self.subTest(message=message):
                    kwargs = valid_kwargs | changes
                    with self.assertRaisesRegex(ValueError, message):
                        aiter_flydsl_gdn_decode(**kwargs)

    def test_decode_entry_forwards_exact_aiter_kwargs(self):
        q, k, v, a, b, state = _make_decode_inputs()
        read_indices = torch.tensor([1, 4], device="cuda", dtype=torch.int32)
        write_indices = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
        A_log = torch.randn(v.shape[2], device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v.shape[2], device="cuda", dtype=torch.bfloat16)
        flydsl_decode = _mock_flydsl_decode()
        self.assertFalse(a.is_contiguous())
        self.assertFalse(b.is_contiguous())
        self.assertEqual(a.stride(), (v.shape[2] * 2, 1))
        self.assertEqual(b.stride(), (v.shape[2] * 2, 1))

        with (
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=flydsl_decode,
            ),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            output = aiter_flydsl_gdn_decode(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                state=state,
                read_indices=read_indices,
                write_indices=write_indices,
                already_validated=True,
            )

        kwargs = flydsl_decode.call_args.kwargs
        self.assertTrue(kwargs["query"].is_contiguous())
        self.assertTrue(kwargs["key"].is_contiguous())
        self.assertTrue(kwargs["value"].is_contiguous())
        self.assertEqual(kwargs["a"].shape, (q.shape[0], 1, v.shape[2]))
        self.assertEqual(kwargs["b"].shape, (q.shape[0], 1, v.shape[2]))
        self.assertTrue(kwargs["a"].is_contiguous())
        self.assertTrue(kwargs["b"].is_contiguous())
        self.assertIs(kwargs["dt_bias"], dt_bias)
        self.assertIs(kwargs["A_log"], A_log)
        self.assertIs(kwargs["indices"], write_indices)
        self.assertIs(kwargs["state"], state)
        self.assertIs(kwargs["out"], output)
        self.assertEqual(output.shape, v.shape)
        self.assertFalse(kwargs["need_shuffle_state"])

    def test_non_boundary_eager_warms_copy_before_capture(self):
        q, k, v, a, b, state = _make_decode_inputs(batch=1)
        state[1].normal_()
        state[2].fill_(-7)
        target_before = state[2].clone()
        read_indices = torch.tensor([1], device="cuda", dtype=torch.int32)
        write_indices = torch.tensor([2], device="cuda", dtype=torch.int32)
        kwargs = {
            "A_log": torch.randn(v.shape[2], device="cuda", dtype=torch.float32),
            "a": a,
            "dt_bias": torch.randn(v.shape[2], device="cuda", dtype=torch.bfloat16),
            "q": q,
            "k": k,
            "v": v,
            "b": b,
            "state": state,
            "read_indices": read_indices,
            "write_indices": write_indices,
            "already_validated": True,
            "copy_state": False,
        }

        with (
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            aiter_flydsl_gdn_decode(**kwargs)
        torch.cuda.synchronize()

        # The no-op warmup compiles the copy kernel without applying the real
        # cross-block read/write indices.
        torch.testing.assert_close(state[2], target_before, rtol=0, atol=0)
        self.assertTrue(adapter._WARMED_DECODE_SIGNATURES)

        # Once the signature is warm, entering capture must not fail and must
        # force the real copy launch even though the eager step is not a block
        # boundary.
        with (
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            aiter_flydsl_gdn_decode(**kwargs)
        torch.cuda.synchronize()
        torch.testing.assert_close(state[2], state[1], rtol=0, atol=0)

    def test_decode_entry_fails_fast_if_capture_was_not_warmed(self):
        q, k, v, a, b, state = _make_decode_inputs()
        kwargs = {
            "A_log": torch.randn(v.shape[2], device="cuda", dtype=torch.float32),
            "a": a,
            "dt_bias": torch.randn(v.shape[2], device="cuda", dtype=torch.bfloat16),
            "q": q,
            "k": k,
            "v": v,
            "b": b,
            "state": state,
            "read_indices": torch.tensor([1, 4], device="cuda", dtype=torch.int32),
            "write_indices": torch.tensor([1, 4], device="cuda", dtype=torch.int32),
            "already_validated": True,
        }
        with (
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "must run once eagerly"),
        ):
            aiter_flydsl_gdn_decode(**kwargs)

    def test_graph_replay_records_copy_for_a_later_block_boundary(self):
        block_map = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1002], device="cuda", dtype=torch.int32)
        state = torch.zeros((3, 1, 8, 8), device="cuda", dtype=torch.float32)
        state[1].normal_()
        state[2].fill_(-7)

        # Warm Triton compilation before capture.
        warm_read, warm_write = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024, state_pool_size=state.shape[0]
        )
        copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
            state, warm_read, warm_write
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
                block_map, lengths, 1024, state_pool_size=state.shape[0]
            )
            copy_aiter_flydsl_gdn_decode_state_at_block_boundary(
                state, read_indices, write_indices
            )

        # Capture length is not a boundary. Replay the same graph with a
        # boundary length; the device-side indices and copy must update.
        state[2].fill_(-7)
        lengths.fill_(1025)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(state[2], state[1], rtol=0, atol=0)

    def test_graph_replay_updates_real_request_invalid_row_flags(self):
        block_map = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1002], device="cuda", dtype=torch.int32)
        metadata = _make_state_metadata(
            block_map,
            lengths,
            1024,
            state_pool_size=3,
        )

        # Warm Triton compilation before capture.
        _prepare_decode_state_indices(metadata)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, _, invalid_row_flags = _prepare_decode_state_indices(metadata)
        torch.cuda.synchronize()
        self.assertEqual(invalid_row_flags.cpu().tolist(), [0])

        lengths.fill_(4097)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(invalid_row_flags.cpu().tolist(), [1])


@unittest.skipUnless(
    torch.version.hip is not None and torch.cuda.is_available(),
    "AITER FlyDSL numerical tests require ROCm",
)
class AiterFlydslGdnDecodeRocmTest(unittest.TestCase):
    def setUp(self):
        _reset_adapter_process_state(self)

    def test_required_aiter_symbol_is_available(self):
        adapter._get_aiter_flydsl_gdn_decode.cache_clear()
        flydsl_decode = adapter._get_aiter_flydsl_gdn_decode()
        self.assertTrue(callable(flydsl_decode))
        self.assertIsNone(
            adapter._callable_signature_incompatibility(
                inspect.signature(adapter._flydsl_gdr_decode_contract),
                inspect.signature(flydsl_decode),
            )
        )

    def test_full_decode_entry_graph_replay_matches_eager_at_boundary(self):
        torch.manual_seed(29)
        q, k, v, a, b, state = _make_decode_inputs(batch=1)
        value_heads = v.shape[2]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        block_map = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1002], device="cuda", dtype=torch.int32)

        # Finish AITER lazy import/JIT before entering the capture window.
        warm_state = state.clone()
        warm_read, warm_write = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024, state_pool_size=warm_state.shape[0]
        )
        aiter_flydsl_gdn_decode(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            state=warm_state,
            read_indices=warm_read,
            write_indices=warm_write,
        )
        torch.cuda.synchronize()

        graph_state = state.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_read, graph_write = prepare_aiter_flydsl_gdn_decode_state_indices(
                block_map, lengths, 1024, state_pool_size=graph_state.shape[0]
            )
            graph_output = aiter_flydsl_gdn_decode(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                state=graph_state,
                read_indices=graph_read,
                write_indices=graph_write,
            )
        torch.cuda.synchronize()

        # Capture records but does not execute the non-boundary launch.
        # Clone after capture so eager and replay each execute the same
        # cross-boundary step from identical state.
        eager_state = graph_state.clone()
        lengths.fill_(1025)
        eager_read, eager_write = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024, state_pool_size=eager_state.shape[0]
        )
        eager_output = aiter_flydsl_gdn_decode(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            state=eager_state,
            read_indices=eager_read,
            write_indices=eager_write,
        )

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        torch.testing.assert_close(graph_state, eager_state, rtol=0, atol=0)

    def test_mixed_block_boundary_batch_matches_triton_and_preserves_pool(self):
        torch.manual_seed(17)
        batch, dim = 4, 128
        sequence_lengths = [1001, 1024, 1500, 2048]
        for key_heads, value_heads in ((2, 8), (16, 32)):
            for state_dtype, A_log_dtype in itertools.product(
                (torch.float32, torch.bfloat16),
                (torch.float32, torch.bfloat16),
            ):
                with self.subTest(
                    key_heads=key_heads,
                    value_heads=value_heads,
                    state_dtype=state_dtype,
                    A_log_dtype=A_log_dtype,
                ):
                    q, k, v, a, b, _ = _make_decode_inputs(
                        batch=batch,
                        key_heads=key_heads,
                        value_heads=value_heads,
                        dim=dim,
                        state_dtype=state_dtype,
                    )
                    self.assertFalse(q.is_contiguous())
                    self.assertFalse(k.is_contiguous())
                    self.assertFalse(v.is_contiguous())
                    A_log = torch.randn(value_heads, device="cuda", dtype=A_log_dtype)
                    dt_bias = torch.randn(
                        value_heads, device="cuda", dtype=torch.bfloat16
                    )
                    block_map = torch.arange(
                        1, batch * 3 + 1, device="cuda", dtype=torch.int32
                    ).reshape(batch, 3)
                    lengths_plus_1 = torch.tensor(
                        [length + 1 for length in sequence_lengths],
                        device="cuda",
                        dtype=torch.int32,
                    )

                    state_elements = value_heads * dim * dim
                    packed_initial = (
                        torch.randn(
                            batch * 3 + 1,
                            state_elements + 12288,
                            device="cuda",
                            dtype=state_dtype,
                        )
                        * 0.01
                    )
                    packed_reference = packed_initial.clone()
                    packed_flydsl = packed_initial.clone()
                    state_reference = packed_reference[:, :state_elements].view(
                        batch * 3 + 1, value_heads, dim, dim
                    )
                    state_flydsl = packed_flydsl[:, :state_elements].view(
                        batch * 3 + 1, value_heads, dim, dim
                    )

                    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
                    output_reference, _ = fused_recurrent_gated_delta_rule(
                        q=q,
                        k=k,
                        v=v,
                        g=g.view(batch, 1, value_heads),
                        beta=beta.view(batch, 1, value_heads),
                        initial_state=state_reference,
                        block_map=block_map,
                        sequence_lengths=lengths_plus_1,
                        seq_size_per_block=1024,
                        use_qk_l2norm_in_kernel=True,
                    )
                    read_indices, write_indices = (
                        prepare_aiter_flydsl_gdn_decode_state_indices(
                            block_map,
                            lengths_plus_1,
                            1024,
                            state_pool_size=state_flydsl.shape[0],
                        )
                    )
                    output_flydsl = aiter_flydsl_gdn_decode(
                        A_log=A_log,
                        a=a,
                        dt_bias=dt_bias,
                        q=q,
                        k=k,
                        v=v,
                        b=b,
                        state=state_flydsl,
                        read_indices=read_indices,
                        write_indices=write_indices,
                    )
                    torch.cuda.synchronize()

                    torch.testing.assert_close(
                        output_flydsl,
                        output_reference,
                        rtol=1e-2,
                        atol=1e-2,
                    )
                    write_ids = write_indices.to(torch.int64)
                    state_rtol = 5e-3 if state_dtype == torch.float32 else 1e-2
                    state_atol = 1e-4 if state_dtype == torch.float32 else 1e-3
                    torch.testing.assert_close(
                        state_flydsl[write_ids],
                        state_reference[write_ids],
                        rtol=state_rtol,
                        atol=state_atol,
                    )

                    # The state copy/update must not touch the packed conv tail.
                    self.assertTrue(
                        torch.equal(
                            packed_flydsl[:, state_elements:],
                            packed_initial[:, state_elements:],
                        )
                    )
                    # Only write blocks may change. This includes an explicit
                    # exact check that reserved dummy block 0 stays unchanged.
                    untouched = torch.ones(
                        packed_initial.shape[0],
                        device="cuda",
                        dtype=torch.bool,
                    )
                    untouched[write_ids] = False
                    self.assertTrue(
                        torch.equal(
                            packed_flydsl[untouched, :state_elements],
                            packed_initial[untouched, :state_elements],
                        )
                    )

    def test_padding_row_uses_dummy_block_without_affecting_real_request(self):
        torch.manual_seed(23)
        q, k, v, a, b, _ = _make_decode_inputs(batch=2)
        value_heads, dim = v.shape[2], v.shape[3]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        block_map = torch.tensor([[1, 2], [0, 0]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1025, 0], device="cuda", dtype=torch.int32)
        state_initial = torch.randn(
            3, value_heads, dim, dim, device="cuda", dtype=torch.float32
        )
        state_reference = state_initial.clone()
        state_flydsl = state_initial.clone()

        g, beta = fused_gdn_gating(A_log, a[:1], b[:1], dt_bias)
        output_reference, _ = fused_recurrent_gated_delta_rule(
            q=q[:1],
            k=k[:1],
            v=v[:1],
            g=g.view(1, 1, value_heads),
            beta=beta.view(1, 1, value_heads),
            initial_state=state_reference,
            block_map=block_map[:1],
            sequence_lengths=lengths[:1],
            seq_size_per_block=1024,
            use_qk_l2norm_in_kernel=True,
        )
        read_indices, write_indices = prepare_aiter_flydsl_gdn_decode_state_indices(
            block_map, lengths, 1024, state_pool_size=state_flydsl.shape[0]
        )
        output_flydsl = aiter_flydsl_gdn_decode(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            state=state_flydsl,
            read_indices=read_indices,
            write_indices=write_indices,
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, 0])
        self.assertEqual(write_indices.cpu().tolist(), [2, 0])
        torch.testing.assert_close(
            output_flydsl[:1], output_reference, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            state_flydsl[2], state_reference[2], rtol=5e-3, atol=5e-4
        )
        # Padding may update reserved dummy block 0, but never a real block.
        torch.testing.assert_close(state_flydsl[1], state_initial[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
