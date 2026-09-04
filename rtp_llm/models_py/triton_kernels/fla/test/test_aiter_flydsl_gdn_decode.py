import inspect
import itertools
import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.fla import aiter_flydsl_gdn_decode as adapter
from rtp_llm.models_py.triton_kernels.fla import utils as fla_utils
from rtp_llm.models_py.triton_kernels.fla.aiter_flydsl_gdn_decode import (
    AiterFlydslGdnDecodeStateMetadata,
    aiter_flydsl_gdn_decode,
    is_aiter_flydsl_gdn_decode_supported,
    prepare_aiter_flydsl_gdn_decode_state_indices,
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    cal_block_idx,
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating


def _make_state_metadata(block_map, lengths, block_size, **kwargs):
    host_sequence_lengths = kwargs.pop("host_sequence_lengths", None)
    if host_sequence_lengths is None:
        host_sequence_lengths = (lengths.cpu() - 1).clamp_min(0)
    default_width = block_map.shape[1] if block_map.ndim == 2 else 1
    return AiterFlydslGdnDecodeStateMetadata(
        block_map=block_map,
        block_map_width=kwargs.pop("block_map_width", default_width),
        sequence_lengths_plus_1=lengths,
        seq_size_per_block=block_size,
        host_sequence_lengths=host_sequence_lengths,
        **kwargs,
    )


def _prepare_indices_for_test(
    block_map,
    lengths,
    block_size,
    host_sequence_lengths=None,
    *,
    state_pool_size,
    block_map_width=None,
):
    metadata_kwargs = {"state_pool_size": state_pool_size}
    if block_map_width is not None:
        metadata_kwargs["block_map_width"] = block_map_width
    return prepare_aiter_flydsl_gdn_decode_state_indices(
        _make_state_metadata(
            block_map,
            lengths,
            block_size,
            host_sequence_lengths=host_sequence_lengths,
            **metadata_kwargs,
        )
    )


def _mock_flydsl_decode():
    return mock.create_autospec(adapter._flydsl_gdr_decode_contract)


def _reset_adapter_process_state(test_case: unittest.TestCase) -> None:
    cached_functions = (
        adapter._is_aiter_flydsl_gdn_decode_disabled,
        adapter._get_aiter_flydsl_gdn_decode,
    )
    mutable_sets = (
        adapter._LOGGED_BACKEND_DECISIONS,
        adapter._WARMED_DECODE_SIGNATURES,
    )
    for cached_function in cached_functions:
        cached_function.cache_clear()
        test_case.addCleanup(cached_function.cache_clear)
    for mutable_set in mutable_sets:
        mutable_set.clear()
        test_case.addCleanup(mutable_set.clear)


def _make_decode_inputs(
    *,
    batch: int = 2,
    query_length: int = 1,
    key_heads: int = 2,
    value_heads: int = 8,
    key_dim: int = 128,
    value_dim: int = 128,
    state_dtype: torch.dtype = torch.float32,
):
    if key_dim == value_dim:
        mixed_qkv = torch.randn(
            batch,
            query_length,
            key_heads * 2 + value_heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        q, k, v = torch.split(mixed_qkv, [key_heads, key_heads, value_heads], dim=2)
    else:
        q_storage = torch.randn(
            batch,
            query_length,
            key_heads + 1,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_storage = torch.randn_like(q_storage)
        v_storage = torch.randn(
            batch,
            query_length,
            value_heads + 1,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        q = q_storage[:, :, :key_heads]
        k = k_storage[:, :, :key_heads]
        v = v_storage[:, :, :value_heads]
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
        value_dim,
        key_dim,
        device="cuda",
        dtype=state_dtype,
    )
    return q, k, v, a, b, state


def _make_decode_kwargs(batch=2, **overrides):
    input_keys = {
        "query_length",
        "key_heads",
        "value_heads",
        "key_dim",
        "value_dim",
        "state_dtype",
    }
    input_overrides = {
        key: overrides.pop(key) for key in tuple(overrides) if key in input_keys
    }
    q, k, v, a, b, state = _make_decode_inputs(batch=batch, **input_overrides)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32) * 3 + 1
    kwargs = {
        "A_log": torch.randn(v.shape[2], device="cuda", dtype=torch.float32),
        "a": a,
        "dt_bias": torch.randn(v.shape[2], device="cuda", dtype=torch.bfloat16),
        "q": q,
        "k": k,
        "v": v,
        "b": b,
        "state": state,
        "read_indices": indices,
        "write_indices": indices,
    }
    kwargs.update(overrides)
    return kwargs


def _make_valid_state_metadata(inputs):
    q, _, _, _, _, state = inputs
    batch = q.shape[0]
    return _make_state_metadata(
        torch.ones((batch, 1), device=q.device, dtype=torch.int32),
        torch.full((batch,), 2, device=q.device, dtype=torch.int32),
        1024,
        host_sequence_lengths=torch.ones(batch, dtype=torch.int32),
        state_pool_size=state.shape[0],
    )


def _call_mock_decode(kwargs, flydsl_decode=None, *, capturing=False, arch=True):
    flydsl_decode = flydsl_decode or _mock_flydsl_decode()
    with (
        mock.patch.object(adapter, "is_amd_cdna3", arch),
        mock.patch.object(adapter, "is_amd_cdna4", False),
        mock.patch.object(
            adapter, "_get_aiter_flydsl_gdn_decode", return_value=flydsl_decode
        ),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
    ):
        return aiter_flydsl_gdn_decode(**kwargs)


def _decode_from_block_map(kwargs, block_map, lengths, host_sequence_lengths=None):
    read_indices, write_indices, _ = _prepare_indices_for_test(
        block_map,
        lengths,
        1024,
        host_sequence_lengths=host_sequence_lengths,
        state_pool_size=kwargs["state"].shape[0],
    )
    output = aiter_flydsl_gdn_decode(
        **(kwargs | {"read_indices": read_indices, "write_indices": write_indices})
    )
    return output, read_indices, write_indices


def _triton_decode_reference(q, k, v, a, b, A_log, dt_bias, state, block_map, lengths):
    batch, value_heads = q.shape[0], v.shape[2]
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    output, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g.view(batch, 1, value_heads),
        beta=beta.view(batch, 1, value_heads),
        initial_state=state,
        block_map=block_map,
        sequence_lengths=lengths,
        seq_size_per_block=1024,
        use_qk_l2norm_in_kernel=True,
    )
    return output


def _flydsl_decode(q, k, v, a, b, A_log, dt_bias, state, block_map, lengths):
    read_indices, write_indices, _ = _prepare_indices_for_test(
        block_map, lengths, 1024, state_pool_size=state.shape[0]
    )
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
    )
    return output, read_indices, write_indices


class FlaEnvironmentFlagTest(unittest.TestCase):
    def test_env_flag_parses_true_false_and_default_values(self):
        for value in ("1", "true", "TRUE", "on", "yes"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"TEST_FLA_FLAG": value}),
            ):
                self.assertTrue(fla_utils.env_flag("TEST_FLA_FLAG"))

        for value in ("0", "off", "false", "", "2"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"TEST_FLA_FLAG": value}),
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

        read_indices, write_indices, _ = _prepare_indices_for_test(
            block_map, sequence_lengths_plus_1, 1024, state_pool_size=100
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, 4])
        self.assertEqual(write_indices.cpu().tolist(), [1, 5])

    def test_prepare_decode_indices_uses_explicit_width_for_narrow_graph_view(self):
        block_map_storage = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], device="cuda", dtype=torch.int32
        )
        block_map = block_map_storage[:, :1]
        lengths = torch.tensor([1025, 2049], device="cuda", dtype=torch.int32)
        host_lengths = torch.tensor([1024, 2048], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "exceeds block-map width"):
            _prepare_indices_for_test(
                block_map,
                lengths,
                1024,
                host_sequence_lengths=host_lengths,
                state_pool_size=7,
            )

        read_indices, write_indices, _ = _prepare_indices_for_test(
            block_map,
            lengths,
            1024,
            host_sequence_lengths=host_lengths,
            state_pool_size=7,
            block_map_width=3,
        )
        torch.cuda.synchronize()
        self.assertEqual(read_indices.cpu().tolist(), [1, 5])
        self.assertEqual(write_indices.cpu().tolist(), [2, 6])

    def test_prepare_decode_indices_maps_invalid_rows_to_skip_sentinel(self):
        cases = (
            (
                [[1, 2], [0, 0], [3, 4]],
                [1025, 0, 4097],
                100,
                ([1, -1, -1], [2, -1, -1], [0, 0, 1]),
            ),
            ([[0, 2]], [1025], 3, ([-1], [-1], [1])),
            ([[1, 7]], [1025], 3, ([-1], [-1], [1])),
        )
        for block_map, lengths, pool_size, expected in cases:
            with self.subTest(block_map=block_map):
                metadata = _make_state_metadata(
                    torch.tensor(block_map, device="cuda", dtype=torch.int32),
                    torch.tensor(lengths, device="cuda", dtype=torch.int32),
                    1024,
                    host_sequence_lengths=torch.tensor(
                        [1024, 0, 0][: len(lengths)], dtype=torch.int32
                    ),
                    state_pool_size=pool_size,
                )
                actual = prepare_aiter_flydsl_gdn_decode_state_indices(metadata)
                torch.cuda.synchronize()
                self.assertEqual(tuple(t.cpu().tolist() for t in actual), expected)

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

        read_indices, write_indices, _ = _prepare_indices_for_test(
            block_map, lengths, block_size, state_pool_size=100
        )
        torch.cuda.synchronize()

        expected_read = []
        expected_write = []
        for row, length in enumerate(lengths.cpu().tolist()):
            if length < 2:
                expected_read.append(-1)
                expected_write.append(-1)
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

        read_indices, write_indices, invalid_flags = _prepare_indices_for_test(
            block_map, lengths, 1024, state_pool_size=1
        )

        self.assertEqual(read_indices.numel(), 0)
        self.assertEqual(write_indices.numel(), 0)
        self.assertEqual(invalid_flags.numel(), 0)

    def test_prepare_decode_indices_rejects_real_out_of_range_request(self):
        block_map = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([4097], device="cuda", dtype=torch.int32)
        host_lengths = torch.tensor([4096], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "exceeds block-map width"):
            prepare_aiter_flydsl_gdn_decode_state_indices(
                _make_state_metadata(
                    block_map,
                    lengths,
                    1024,
                    host_sequence_lengths=host_lengths,
                    state_pool_size=3,
                )
            )

    def test_prepare_decode_indices_requires_cpu_host_lengths(self):
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([2], device="cuda", dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "must be a CPU tensor"):
            prepare_aiter_flydsl_gdn_decode_state_indices(
                _make_state_metadata(
                    block_map,
                    lengths,
                    1024,
                    host_sequence_lengths=lengths,
                    state_pool_size=2,
                )
            )

    def test_prepare_decode_indices_validates_arguments(self):
        valid_map = torch.ones((2, 2), device="cuda", dtype=torch.int32)
        valid_lengths = torch.ones((2,), device="cuda", dtype=torch.int32)
        strided_lengths = torch.ones((4,), device="cuda", dtype=torch.int32)[::2]
        cases = (
            (valid_map.reshape(-1), valid_lengths, 1024, "block_map must be 2D"),
            (
                torch.empty((2, 0), device="cuda", dtype=torch.int32),
                valid_lengths,
                1024,
                "at least one block column",
            ),
            (valid_map.to(torch.int64), valid_lengths, 1024, "block_map must be int32"),
            (
                valid_map,
                valid_lengths.reshape(1, 2),
                1024,
                "sequence_lengths_plus_1 must be 1D",
            ),
            (valid_map, valid_lengths.to(torch.int64), 1024, "must be int32"),
            (
                valid_map,
                strided_lengths,
                1024,
                "sequence_lengths_plus_1 must be contiguous",
            ),
            (valid_map.cpu(), valid_lengths, 1024, "must be on the same device"),
            (valid_map, valid_lengths[:1], 1024, "count must equal"),
            (valid_map, valid_lengths, 0, "seq_size_per_block must be positive"),
            (valid_map[:, ::2], valid_lengths, 1024, "columns must be contiguous"),
        )
        for block_map, lengths, block_size, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                prepare_aiter_flydsl_gdn_decode_state_indices(
                    _make_state_metadata(
                        block_map, lengths, block_size, state_pool_size=100
                    )
                )
        with self.assertRaisesRegex(ValueError, "state_pool_size must be positive"):
            prepare_aiter_flydsl_gdn_decode_state_indices(
                _make_state_metadata(valid_map, valid_lengths, 1024, state_pool_size=0)
            )
        with self.assertRaisesRegex(TypeError, "state_pool_size"):
            AiterFlydslGdnDecodeStateMetadata(
                block_map=valid_map,
                block_map_width=valid_map.shape[1],
                sequence_lengths_plus_1=valid_lengths,
                seq_size_per_block=1024,
                host_sequence_lengths=torch.zeros(2, dtype=torch.int32),
            )

    def test_aiter_symbol_resolver_falls_back_on_runtime_import_failure(self):
        original_import = __import__

        def import_with_target_failure(name, *args, **kwargs):
            if name == "aiter.ops.flydsl.linear_attention_kernels":
                raise RuntimeError("extension load failure")
            return original_import(name, *args, **kwargs)

        with (
            self.assertLogs(adapter._LOGGER.name, level="WARNING") as logs,
            mock.patch("builtins.__import__", side_effect=import_with_target_failure),
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

            def supported(inputs=valid, **overrides):
                return is_aiter_flydsl_gdn_decode_supported(
                    *inputs,
                    A_log=overrides.pop("A_log", A_log),
                    dt_bias=overrides.pop("dt_bias", dt_bias),
                    state_metadata=overrides.pop(
                        "state_metadata", _make_valid_state_metadata(inputs)
                    ),
                    **overrides,
                )

            self.assertTrue(supported())
            self.assertTrue(supported(_make_decode_inputs(key_heads=4)))
            self.assertTrue(supported(_make_decode_inputs(key_dim=64, value_dim=96)))
            self.assertFalse(supported(_make_decode_inputs(query_length=2)))
            reason = adapter._aiter_flydsl_gdn_decode_unsupported_reason(
                *_make_decode_inputs(query_length=2),
                A_log=A_log,
                dt_bias=dt_bias,
                scale=None,
            )
            self.assertIn("decode requires T==1", reason)

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
                ("empty-token", _make_decode_inputs(query_length=0)),
                ("invalid-head-ratio", _make_decode_inputs(key_heads=3)),
                ("q-fp16", (q.to(torch.float16), k, v, a, b, state)),
                ("k-shape", (q, k[:, :, :1], v, a, b, state)),
                ("v-shape", (q, k, v[:, :0], a, b, state)),
                ("state-fp16", (q, k, v, a, b, state.to(torch.float16))),
                ("state-unaligned", (q, k, v, a, b, unaligned_state)),
                ("state-row-stride", (q, k, v, a, b, row_strided_state)),
                ("state-inner-stride", (q, k, v, a, b, state.transpose(-1, -2))),
                ("a-shape", (q, k, v, a.reshape(1, -1), b, state)),
            )
            for name, inputs in cases:
                with self.subTest(name=name):
                    self.assertFalse(supported(inputs))

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
                        supported(
                            A_log=case_A_log,
                            dt_bias=case_dt_bias,
                            scale=scale,
                        )
                    )

            with mock.patch.object(
                adapter, "_get_aiter_flydsl_gdn_decode", return_value=None
            ):
                self.assertFalse(supported())

    def test_shape_gate_rejects_unsupported_arch_empty_batch_and_kill_switch(self):
        valid = _make_decode_inputs()
        v = valid[2]
        A_log = torch.randn(v.shape[2], device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v.shape[2], device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(
            adapter, "_get_aiter_flydsl_gdn_decode", return_value=_mock_flydsl_decode()
        ):
            with (
                mock.patch.object(adapter, "is_amd_cdna3", False),
                mock.patch.object(adapter, "is_amd_cdna4", False),
            ):
                self.assertFalse(
                    is_aiter_flydsl_gdn_decode_supported(
                        *valid,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        state_metadata=_make_valid_state_metadata(valid),
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
                self.assertIn("B>=1", reason)

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
            is_aiter_flydsl_gdn_decode_supported(
                *valid,
                A_log=A_log,
                dt_bias=dt_bias,
                state_metadata=_make_valid_state_metadata(valid),
            )
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
        lengths = torch.ones(2, device="cuda", dtype=torch.int32)
        with (
            mock.patch.object(adapter, "is_amd_cdna3", True),
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
        ):
            with self.assertRaisesRegex(TypeError, "state_metadata"):
                is_aiter_flydsl_gdn_decode_supported(
                    *valid,
                    A_log=A_log,
                    dt_bias=dt_bias,
                )
            self.assertFalse(
                is_aiter_flydsl_gdn_decode_supported(
                    *valid,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    state_metadata=_make_state_metadata(
                        torch.ones((2, 2), device="cuda", dtype=torch.int64),
                        lengths,
                        1024,
                        state_pool_size=valid[5].shape[0],
                    ),
                )
            )
            with self.assertLogs(adapter._LOGGER.name, level="INFO") as logs:
                self.assertFalse(
                    is_aiter_flydsl_gdn_decode_supported(
                        *valid,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        state_metadata=_make_state_metadata(
                            torch.ones((3, 2), device="cuda", dtype=torch.int32),
                            torch.ones(3, device="cuda", dtype=torch.int32),
                            1024,
                            state_pool_size=valid[5].shape[0],
                        ),
                    )
                )
            self.assertIn(
                "block-map batch differs from decode input batch",
                "\n".join(logs.output),
            )

    def test_backend_logging_preserves_distinct_fallback_categories(self):
        valid = _make_decode_inputs()
        value_heads = valid[2].shape[2]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        metadata = _make_valid_state_metadata(valid)
        with (
            mock.patch.object(adapter, "is_amd_cdna3", True),
            mock.patch.object(
                adapter,
                "_get_aiter_flydsl_gdn_decode",
                return_value=_mock_flydsl_decode(),
            ),
            self.assertLogs(adapter._LOGGER.name, level="INFO") as logs,
        ):
            self.assertFalse(
                is_aiter_flydsl_gdn_decode_supported(
                    *valid,
                    A_log=A_log.to(torch.float16),
                    dt_bias=dt_bias,
                    state_metadata=metadata,
                )
            )
            multi_token = _make_decode_inputs(query_length=2)
            self.assertFalse(
                is_aiter_flydsl_gdn_decode_supported(
                    *multi_token,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    state_metadata=_make_valid_state_metadata(multi_token),
                )
            )

        messages = "\n".join(logs.output)
        self.assertIn("A_log layout or dtype is unsupported", messages)
        self.assertIn("decode requires T==1", messages)

    def test_decode_entry_validates_auxiliary_arguments(self):
        valid_kwargs = _make_decode_kwargs()
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
        with mock.patch.object(adapter, "is_amd_cdna3", True):
            for changes, message in cases:
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    _call_mock_decode(valid_kwargs | changes)

    def test_decode_entry_forwards_exact_aiter_kwargs(self):
        call_kwargs = _make_decode_kwargs()
        q, v = call_kwargs["q"], call_kwargs["v"]
        a, b = call_kwargs["a"], call_kwargs["b"]
        write_indices = torch.tensor([2, 5], device="cuda", dtype=torch.int32)
        call_kwargs["write_indices"] = write_indices
        flydsl_decode = _mock_flydsl_decode()
        self.assertFalse(a.is_contiguous())
        self.assertFalse(b.is_contiguous())
        self.assertEqual(a.stride(), (v.shape[2] * 2, 1))
        self.assertEqual(b.stride(), (v.shape[2] * 2, 1))

        output = _call_mock_decode(call_kwargs, flydsl_decode)

        kwargs = flydsl_decode.call_args.kwargs
        self.assertIs(kwargs["query"], call_kwargs["q"])
        self.assertIs(kwargs["key"], call_kwargs["k"])
        self.assertIs(kwargs["value"], call_kwargs["v"])
        self.assertEqual(kwargs["a"].shape, (q.shape[0], 1, v.shape[2]))
        self.assertEqual(kwargs["b"].shape, (q.shape[0], 1, v.shape[2]))
        self.assertEqual(kwargs["a"].stride(), (v.shape[2] * 2, v.shape[2], 1))
        self.assertEqual(kwargs["b"].stride(), (v.shape[2] * 2, v.shape[2], 1))
        self.assertIs(kwargs["dt_bias"], call_kwargs["dt_bias"])
        self.assertIs(kwargs["A_log"], call_kwargs["A_log"])
        self.assertIs(kwargs["indices"], write_indices)
        self.assertIs(kwargs["read_indices"], call_kwargs["read_indices"])
        self.assertIs(kwargs["write_indices"], write_indices)
        self.assertIs(kwargs["state"], call_kwargs["state"])
        self.assertIs(kwargs["out"], output)
        self.assertEqual(output.shape, v.shape)
        self.assertFalse(kwargs["need_shuffle_state"])

    def test_decode_entry_fails_fast_if_capture_was_not_warmed(self):
        kwargs = _make_decode_kwargs()
        with self.assertRaisesRegex(RuntimeError, "must run once eagerly"):
            _call_mock_decode(kwargs, capturing=True)

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
        prepare_aiter_flydsl_gdn_decode_state_indices(metadata)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, _, invalid_row_flags = prepare_aiter_flydsl_gdn_decode_state_indices(
                metadata
            )
        graph.replay()
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
        kwargs = _make_decode_kwargs(batch=1)
        state = kwargs["state"]
        block_map = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1002], device="cuda", dtype=torch.int32)

        warm_state = state.clone()
        host_lengths = torch.tensor([1001], dtype=torch.int32)
        _decode_from_block_map(
            kwargs | {"state": warm_state},
            block_map,
            lengths,
            host_lengths,
        )
        torch.cuda.synchronize()

        graph_state = state.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, _, _ = _decode_from_block_map(
                kwargs | {"state": graph_state},
                block_map,
                lengths,
                host_lengths,
            )
        torch.cuda.synchronize()

        eager_state = graph_state.clone()
        lengths.fill_(1025)
        eager_output, _, _ = _decode_from_block_map(
            kwargs | {"state": eager_state},
            block_map,
            lengths,
            host_lengths,
        )

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        torch.testing.assert_close(graph_state, eager_state, rtol=0, atol=0)

    def test_mixed_block_boundary_batch_matches_triton_and_preserves_pool(self):
        torch.manual_seed(17)
        batch, key_dim, value_dim = 4, 128, 128
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
                        key_dim=key_dim,
                        value_dim=value_dim,
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

                    state_elements = value_heads * value_dim * key_dim
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
                        batch * 3 + 1, value_heads, value_dim, key_dim
                    )
                    state_flydsl = packed_flydsl[:, :state_elements].view(
                        batch * 3 + 1, value_heads, value_dim, key_dim
                    )

                    output_reference = _triton_decode_reference(
                        q,
                        k,
                        v,
                        a,
                        b,
                        A_log,
                        dt_bias,
                        state_reference,
                        block_map,
                        lengths_plus_1,
                    )
                    output_flydsl, _, write_indices = _flydsl_decode(
                        q,
                        k,
                        v,
                        a,
                        b,
                        A_log,
                        dt_bias,
                        state_flydsl,
                        block_map,
                        lengths_plus_1,
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

                    # The fused state read/update must not touch the packed conv tail.
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

    def test_unequal_key_value_dims_matches_triton(self):
        """Cover the state tensor's distinct (V, K) dimensions and strides."""
        torch.manual_seed(31)
        batch, key_heads, value_heads = 2, 2, 8
        key_dim, value_dim = 64, 96
        q, k, v, a, b, _ = _make_decode_inputs(
            batch=batch,
            key_heads=key_heads,
            value_heads=value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        block_map = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([17, 1025], device="cuda", dtype=torch.int32)
        state_initial = torch.randn(
            5,
            value_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
        )
        state_reference = state_initial.clone()
        state_flydsl = state_initial.clone()

        output_reference = _triton_decode_reference(
            q, k, v, a, b, A_log, dt_bias, state_reference, block_map, lengths
        )
        output_flydsl, _, write_indices = _flydsl_decode(
            q, k, v, a, b, A_log, dt_bias, state_flydsl, block_map, lengths
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            output_flydsl, output_reference, rtol=1e-2, atol=1e-2
        )
        write_ids = write_indices.to(torch.int64)
        torch.testing.assert_close(
            state_flydsl[write_ids],
            state_reference[write_ids],
            rtol=5e-3,
            atol=5e-4,
        )

    def test_padding_row_skips_state_pool_without_affecting_real_request(self):
        torch.manual_seed(23)
        q, k, v, a, b, _ = _make_decode_inputs(batch=2)
        value_heads, value_dim = v.shape[2], v.shape[3]
        key_dim = q.shape[3]
        A_log = torch.randn(value_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.bfloat16)
        block_map = torch.tensor([[1, 2], [0, 0]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1025, 0], device="cuda", dtype=torch.int32)
        state_initial = torch.randn(
            3,
            value_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
        )
        state_reference = state_initial.clone()
        state_flydsl = state_initial.clone()

        output_reference = _triton_decode_reference(
            q[:1],
            k[:1],
            v[:1],
            a[:1],
            b[:1],
            A_log,
            dt_bias,
            state_reference,
            block_map[:1],
            lengths[:1],
        )
        output_flydsl, read_indices, write_indices = _flydsl_decode(
            q, k, v, a, b, A_log, dt_bias, state_flydsl, block_map, lengths
        )
        torch.cuda.synchronize()

        self.assertEqual(read_indices.cpu().tolist(), [1, -1])
        self.assertEqual(write_indices.cpu().tolist(), [2, -1])
        torch.testing.assert_close(
            output_flydsl[:1], output_reference, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            output_flydsl[1], torch.zeros_like(output_flydsl[1]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            state_flydsl[2], state_reference[2], rtol=5e-3, atol=5e-4
        )
        # AITER's negative sentinel skips the padding row entirely.
        torch.testing.assert_close(state_flydsl[0], state_initial[0], rtol=0, atol=0)
        torch.testing.assert_close(state_flydsl[1], state_initial[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
