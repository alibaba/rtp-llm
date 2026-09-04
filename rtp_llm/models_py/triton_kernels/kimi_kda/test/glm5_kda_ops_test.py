import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from rtp_llm.models_py.triton_kernels.fla import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
    get_kda_chunk_size,
    prepare_kda_recurrent_checkpoint_metadata,
    store_kda_recurrent_checkpoints,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter


class Glm5KdaOpsTest(unittest.TestCase):
    HEADS = 4
    HEAD_DIM = 128

    def setUp(self):
        torch.manual_seed(7)
        shape = (1, 8, self.HEADS, self.HEAD_DIM)
        self.q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self.beta = torch.randn(
            shape[:-1], device="cuda", dtype=torch.bfloat16
        ).sigmoid()
        self.a_log = torch.randn(self.HEADS, device="cuda", dtype=torch.float32).mul_(
            0.1
        )
        self.dt_bias = torch.randn(
            self.HEADS, self.HEAD_DIM, device="cuda", dtype=torch.float32
        ).mul_(0.01)

    @staticmethod
    def _cosine(actual, expected):
        return F.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        ).item()

    def test_kda_chunk_size_env(self):
        with mock.patch.dict(os.environ, {"KDA_CHUNK_SIZE": "256"}):
            self.assertEqual(get_kda_chunk_size(), 256)
        with mock.patch.dict(os.environ, {"KDA_CHUNK_SIZE": "63"}):
            with self.assertRaisesRegex(ValueError, "64, 128, 256"):
                get_kda_chunk_size()

    @staticmethod
    def _cula_chunk_kda():
        for root in tuple(sys.path):
            cutlass_packages = Path(root) / "nvidia_cutlass_dsl" / "python_packages"
            if (cutlass_packages / "cutlass" / "__init__.py").is_file():
                sys.path.insert(0, str(cutlass_packages))
                break
        from cula.kda import chunk_kda

        return chunk_kda

    @torch.inference_mode()
    def test_cula_checkpoint_api_is_stable_without_final_state(self):
        cula_chunk_kda = self._cula_chunk_kda()
        lengths = [70, 130]
        interval = 64
        token_count = sum(lengths)
        heads = 2
        state_dim = self.HEAD_DIM
        shape = (1, token_count, heads, state_dim)
        q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        gate = torch.randn_like(q)
        beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
        initial = torch.randn(
            len(lengths),
            heads,
            state_dim,
            state_dim,
            dtype=torch.float32,
            device="cuda",
        )
        cu_host = torch.tensor([0, lengths[0], token_count], dtype=torch.int32)
        cu_device = cu_host.cuda()
        checkpoint_count = sum(
            (length + interval - 1) // interval for length in lengths
        )
        alog = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(heads * state_dim, dtype=torch.float32, device="cuda")

        def run(output_final_state):
            checkpoints = torch.empty(
                1,
                checkpoint_count,
                heads,
                state_dim,
                state_dim,
                dtype=torch.float32,
                device="cuda",
            )
            output, final_state, published = cula_chunk_kda(
                q,
                k,
                v,
                gate,
                beta,
                scale=state_dim**-0.5,
                initial_state=initial,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                cu_seqlens=cu_device,
                cu_seqlens_cpu=cu_host,
                safe_gate=True,
                lower_bound=-5.0,
                disable_recompute=False,
                use_intracard_cp=False,
                A_log=alog,
                dt_bias=dt_bias,
                checkpoint_interval=interval,
                checkpoint_states=checkpoints,
            )
            self.assertEqual(published.data_ptr(), checkpoints.data_ptr())
            return output, final_state, checkpoints

        output, no_final_state, checkpoints = run(False)
        expected_output, final_state, expected_checkpoints = run(True)
        self.assertIsNone(no_final_state)
        self.assertIsNotNone(final_state)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(checkpoints, expected_checkpoints, rtol=0, atol=0)
        torch.testing.assert_close(final_state[0], checkpoints[0, 1], rtol=0, atol=0)
        torch.testing.assert_close(final_state[1], checkpoints[0, 4], rtol=0, atol=0)

    def test_cula_checkpoint_store_skips_compact_null_pages(self):
        metadata = prepare_kda_recurrent_checkpoint_metadata(
            torch.tensor([256], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            128,
            torch.device("cuda"),
        )
        checkpoints = torch.stack(
            [
                torch.full((1, 4, 4), 11.0, dtype=torch.float32, device="cuda"),
                torch.full((1, 4, 4), 22.0, dtype=torch.float32, device="cuda"),
            ]
        )
        cache = torch.zeros(5, 1, 4, 4, dtype=torch.float32, device="cuda")
        store_kda_recurrent_checkpoints(
            checkpoints,
            metadata,
            torch.tensor([[0, 4]], dtype=torch.int32, device="cuda"),
            cache,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(cache[4], checkpoints[1], rtol=0, atol=0)
        torch.testing.assert_close(
            cache[:4], torch.zeros_like(cache[:4]), rtol=0, atol=0
        )

    def test_chunk256_matches_chunk64_for_packed_varlen(self):
        lengths = [73, 227]
        token_count = sum(lengths)
        shape = (1, token_count, self.HEADS, self.HEAD_DIM)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16).sigmoid()
        cu_seqlens = torch.tensor(
            [0, lengths[0], token_count], device="cuda", dtype=torch.int32
        )
        initial_state = torch.zeros(
            len(lengths),
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        common = {
            "initial_state": initial_state,
            "output_final_state": True,
            "cu_seqlens": cu_seqlens,
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "safe_gate": True,
            "lower_bound": -5.0,
        }

        output64, state64 = chunk_kda(q, k, v, g, beta, chunk_size=64, **common)
        output256, state256 = chunk_kda(q, k, v, g, beta, chunk_size=256, **common)

        self.assertGreater(self._cosine(output256, output64), 0.999)
        self.assertGreater(self._cosine(state256, state64), 0.999)

    def test_chunk256_store_writes_first_completed_chunk(self):
        chunk_states = torch.stack(
            [
                torch.full(
                    (self.HEADS, self.HEAD_DIM, self.HEAD_DIM),
                    value,
                    device="cuda",
                    dtype=torch.float32,
                )
                for value in (10.0, 20.0, 30.0)
            ]
        ).unsqueeze(0)
        final_state = torch.full(
            (1, self.HEADS, self.HEAD_DIM, self.HEAD_DIM),
            99.0,
            device="cuda",
            dtype=torch.float32,
        )
        cache = torch.zeros(
            6,
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        store_ssm_state_to_block_map(
            chunk_states,
            final_state,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([0, 600], device="cuda", dtype=torch.int32),
            torch.tensor([[1, 2, 3, 4, 5]], device="cuda", dtype=torch.int32),
            cache,
            seq_size_per_block=128,
            chunk_size=256,
        )

        torch.testing.assert_close(cache[2], chunk_states[0, 1])
        torch.testing.assert_close(cache[4], chunk_states[0, 2])
        torch.testing.assert_close(cache[5], final_state[0])

    def _reference(self, lower_bound=None):
        q = F.normalize(self.q.float(), dim=-1).mul_(self.HEAD_DIM**-0.5)
        k = F.normalize(self.k.float(), dim=-1)
        v = self.v.float()
        gate_input = self.g.float() + self.dt_bias.view(1, 1, self.HEADS, self.HEAD_DIM)
        a = torch.exp(self.a_log).view(1, 1, self.HEADS, 1)
        gate = (
            lower_bound * torch.sigmoid(a * gate_input)
            if lower_bound is not None
            else -a * F.softplus(gate_input)
        )
        state = torch.zeros(
            1,
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        outputs = []
        for token in range(self.q.shape[1]):
            state.mul_(torch.exp(gate[:, token]).unsqueeze(-1))
            key = k[:, token].unsqueeze(-1)
            value = v[:, token].unsqueeze(-2)
            correction = (state * key).sum(dim=-2, keepdim=True)
            value = (value - correction) * self.beta[:, token, :, None, None]
            state.add_(key * value)
            outputs.append(torch.matmul(q[:, token].unsqueeze(-2), state).squeeze(-2))
        return torch.stack(outputs, dim=1), state

    def test_gate_matches_torch(self):
        actual = fused_kda_gate(
            self.g.flatten(0, 1),
            self.a_log,
            dt_bias=self.dt_bias,
        )
        expected = -torch.exp(self.a_log).view(1, self.HEADS, 1) * F.softplus(
            self.g.flatten(0, 1).float() + self.dt_bias
        )
        self.assertGreater(self._cosine(actual, expected), 0.999)
        self.assertTrue((actual <= 0).all())

    def test_packed_varlen_conv_matches_serial_sequences(self):
        lengths = [5, 9, 3]
        cu_seqlens = torch.tensor([0, 5, 14, 17], device="cuda", dtype=torch.int32)
        channels = 64
        inputs = torch.randn(
            sum(lengths), channels, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        prefix_lengths = torch.zeros(len(lengths), device="cuda", dtype=torch.int32)

        packed = causal_conv1d_fn(
            inputs.transpose(0, 1),
            weight,
            None,
            None,
            cu_seqlens,
            None,
            prefix_lengths,
            64,
        ).transpose(0, 1)
        serial = []
        offset = 0
        for length in lengths:
            serial.append(
                causal_conv1d_fn(
                    inputs[offset : offset + length].transpose(0, 1),
                    weight,
                    None,
                    None,
                    torch.tensor([0, length], device="cuda", dtype=torch.int32),
                    None,
                    torch.zeros(1, device="cuda", dtype=torch.int32),
                    64,
                ).transpose(0, 1)
            )
            offset += length
        torch.testing.assert_close(packed, torch.cat(serial), rtol=0, atol=0)

    def test_mixed_prefix_shared_block_conv_matches_serial_sequences(self):
        lengths = [5, 7, 7]
        cu_seqlens = torch.tensor([0, 5, 12, 19], device="cuda", dtype=torch.int32)
        prefix_lengths = torch.tensor([0, 64, 64], device="cuda", dtype=torch.int32)
        block_map = torch.tensor(
            [[1, -1], [2, 3], [2, 4]], device="cuda", dtype=torch.int32
        )
        channels = 64
        inputs = torch.randn(
            sum(lengths), channels, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        # Production stores [block, state_len, channel] and passes a transposed
        # view so the feature dimension is contiguous for the Triton kernel.
        original_cache = torch.randn(
            5, 3, channels, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        packed_cache = original_cache.clone()

        packed = causal_conv1d_fn(
            inputs.transpose(0, 1),
            weight,
            None,
            packed_cache,
            cu_seqlens,
            block_map,
            prefix_lengths,
            64,
        ).transpose(0, 1)
        serial_outputs = []
        serial_caches = []
        offset = 0
        for batch_idx, length in enumerate(lengths):
            cache = original_cache.clone()
            serial_outputs.append(
                causal_conv1d_fn(
                    inputs[offset : offset + length].transpose(0, 1),
                    weight,
                    None,
                    cache,
                    torch.tensor([0, length], device="cuda", dtype=torch.int32),
                    block_map[batch_idx : batch_idx + 1],
                    prefix_lengths[batch_idx : batch_idx + 1],
                    64,
                ).transpose(0, 1)
            )
            serial_caches.append(cache)
            offset += length

        torch.testing.assert_close(packed, torch.cat(serial_outputs), rtol=0, atol=0)
        torch.testing.assert_close(packed_cache[1], serial_caches[0][1])
        torch.testing.assert_close(packed_cache[3], serial_caches[1][3])
        torch.testing.assert_close(packed_cache[4], serial_caches[2][4])
        torch.testing.assert_close(packed_cache[2], original_cache[2])

    def test_packed_varlen_kda_matches_serial_sequences(self):
        lengths = [5, 9, 3]
        token_count = sum(lengths)
        shape = (1, token_count, self.HEADS, self.HEAD_DIM)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16).sigmoid()
        cu_seqlens = torch.tensor([0, 5, 14, 17], device="cuda", dtype=torch.int32)
        initial_state = torch.zeros(
            len(lengths),
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        common = {
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "safe_gate": True,
            "lower_bound": -5.0,
        }

        packed_output, packed_state = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state.clone(),
            cu_seqlens=cu_seqlens,
            **common,
        )
        serial_outputs = []
        serial_states = []
        offset = 0
        for batch_idx, length in enumerate(lengths):
            output, state = chunk_kda(
                q[:, offset : offset + length],
                k[:, offset : offset + length],
                v[:, offset : offset + length],
                g[:, offset : offset + length],
                beta[:, offset : offset + length],
                initial_state=initial_state[batch_idx : batch_idx + 1].clone(),
                **common,
            )
            serial_outputs.append(output)
            serial_states.append(state)
            offset += length

        torch.testing.assert_close(
            packed_output, torch.cat(serial_outputs, dim=1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed_state, torch.cat(serial_states, dim=0), rtol=0, atol=0
        )

    def test_mixed_prefix_shared_block_kda_cache_matches_serial_sequences(self):
        lengths = [5, 7, 7]
        token_count = sum(lengths)
        cu_seqlens = torch.tensor([0, 5, 12, 19], device="cuda", dtype=torch.int32)
        prefix_lengths = torch.tensor([0, 64, 64], device="cuda", dtype=torch.int32)
        block_map = torch.tensor(
            [[1, -1], [2, 3], [2, 4]], device="cuda", dtype=torch.int32
        )
        shape = (1, token_count, self.HEADS, self.HEAD_DIM)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16).sigmoid()
        original_cache = torch.randn(
            5,
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        common = {
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "safe_gate": True,
            "lower_bound": -5.0,
            "return_intermediate_states": True,
        }

        packed_initial = torch.empty(
            len(lengths),
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        load_initial_state_from_block_map(
            prefix_lengths,
            block_map,
            original_cache,
            packed_initial,
            64,
        )
        packed_output, packed_final, packed_h = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=packed_initial,
            cu_seqlens=cu_seqlens,
            **common,
        )
        packed_cache = original_cache.clone()
        store_ssm_state_to_block_map(
            packed_h.float(),
            packed_final.float(),
            prefix_lengths,
            cu_seqlens,
            block_map,
            packed_cache,
            64,
            chunk_size=64,
        )

        serial_outputs = []
        serial_caches = []
        offset = 0
        for batch_idx, length in enumerate(lengths):
            initial = torch.empty(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            )
            load_initial_state_from_block_map(
                prefix_lengths[batch_idx : batch_idx + 1],
                block_map[batch_idx : batch_idx + 1],
                original_cache,
                initial,
                64,
            )
            output, final, intermediate = chunk_kda(
                q[:, offset : offset + length],
                k[:, offset : offset + length],
                v[:, offset : offset + length],
                g[:, offset : offset + length],
                beta[:, offset : offset + length],
                initial_state=initial,
                **common,
            )
            cache = original_cache.clone()
            local_cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            store_ssm_state_to_block_map(
                intermediate.float(),
                final.float(),
                prefix_lengths[batch_idx : batch_idx + 1],
                local_cu,
                block_map[batch_idx : batch_idx + 1],
                cache,
                64,
                chunk_size=64,
            )
            serial_outputs.append(output)
            serial_caches.append(cache)
            offset += length

        torch.testing.assert_close(
            packed_output, torch.cat(serial_outputs, dim=1), rtol=0, atol=0
        )
        torch.testing.assert_close(packed_cache[1], serial_caches[0][1])
        torch.testing.assert_close(packed_cache[3], serial_caches[1][3])
        torch.testing.assert_close(packed_cache[4], serial_caches[2][4])
        torch.testing.assert_close(packed_cache[2], original_cache[2])

    def test_prefill_and_decode_match_torch(self):
        expected, expected_state = self._reference()
        initial_state = torch.zeros_like(expected_state)
        actual, actual_state = chunk_kda(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.a_log,
            dt_bias=self.dt_bias.flatten(),
        )
        self.assertGreater(self._cosine(actual, expected), 0.98)
        self.assertGreater(self._cosine(actual_state, expected_state), 0.98)

        decode, _ = fused_recurrent_kda(
            self.q[:, -1:],
            self.k[:, -1:],
            self.v[:, -1:],
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=self._reference_prefix_state(),
            A_log=self.a_log,
            dt_bias=self.dt_bias.flatten(),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
        )
        self.assertGreater(self._cosine(decode, expected[:, -1:]), 0.98)

    def test_bounded_prefill_state_matches_decode(self):
        lower_bound = -5.0
        common = {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "lower_bound": lower_bound,
        }
        full_output, _ = chunk_kda(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            safe_gate=True,
            **common,
        )
        _, prefix_state = chunk_kda(
            self.q[:, :-1],
            self.k[:, :-1],
            self.v[:, :-1],
            self.g[:, :-1],
            self.beta[:, :-1],
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            safe_gate=True,
            **common,
        )
        decode_output, _ = fused_recurrent_kda(
            self.q[:, -1:],
            self.k[:, -1:],
            self.v[:, -1:],
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=prefix_state,
            **common,
        )

        self.assertGreater(self._cosine(decode_output, full_output[:, -1:]), 0.999)

    def test_bounded_paged_prefill_state_matches_decode(self):
        lower_bound = -5.0
        common = {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "lower_bound": lower_bound,
        }
        prefix_length = self.q.shape[1] - 1
        _, prefix_state, intermediate_states = chunk_kda(
            self.q[:, :prefix_length],
            self.k[:, :prefix_length],
            self.v[:, :prefix_length],
            self.g[:, :prefix_length],
            self.beta[:, :prefix_length],
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            cu_seqlens=torch.tensor(
                [0, prefix_length], device="cuda", dtype=torch.int32
            ),
            safe_gate=True,
            return_intermediate_states=True,
            **common,
        )
        page_size = 64
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        paged_state = torch.zeros(
            2,
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        store_ssm_state_to_block_map(
            intermediate_states.float(),
            prefix_state,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([0, prefix_length], device="cuda", dtype=torch.int32),
            block_map,
            paged_state,
            page_size,
            chunk_size=64,
        )
        torch.testing.assert_close(paged_state[1], prefix_state[0])

        decode_args = (
            self.q[:, -1:],
            self.k[:, -1:],
            self.v[:, -1:],
            self.g[:, -1:],
            self.beta[:, -1:],
        )
        expected, _ = fused_recurrent_kda(
            *decode_args,
            initial_state=prefix_state.clone(),
            **common,
        )
        actual, _ = fused_recurrent_kda(
            *decode_args,
            initial_state=paged_state,
            block_map=block_map,
            seq_size_per_block=page_size,
            sequence_lengths=torch.tensor(
                [prefix_length + 1], device="cuda", dtype=torch.int32
            ),
            **common,
        )

        self.assertGreater(self._cosine(actual, expected), 0.999)

    def test_paged_prefill_conv_state_matches_decode(self):
        token_count = 9
        prefix_length = token_count - 2
        channels = self.HEAD_DIM
        inputs = torch.randn(token_count, channels, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        conv_cache = torch.zeros(3, 3, channels, device="cuda", dtype=torch.bfloat16)
        prefix_cu_seqlens = torch.tensor(
            [0, prefix_length], device="cuda", dtype=torch.int32
        )
        causal_conv1d_fn(
            inputs[:prefix_length].transpose(0, 1),
            weight,
            None,
            conv_cache.transpose(1, 2),
            prefix_cu_seqlens,
            block_map,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        )
        torch.testing.assert_close(
            conv_cache[1], inputs[prefix_length - 3 : prefix_length]
        )

        first_actual = causal_conv1d_update(
            inputs[prefix_length : prefix_length + 1]
            .reshape(1, 1, channels)
            .transpose(1, 2),
            conv_cache.transpose(1, 2),
            weight,
            activation="silu",
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [prefix_length + 1], device="cuda", dtype=torch.int32
            ),
        )
        torch.testing.assert_close(
            conv_cache[1], inputs[prefix_length - 2 : prefix_length + 1]
        )
        second_actual = causal_conv1d_update(
            inputs[-1:].reshape(1, 1, channels).transpose(1, 2),
            conv_cache.transpose(1, 2),
            weight,
            activation="silu",
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [token_count], device="cuda", dtype=torch.int32
            ),
        )
        expected = causal_conv1d_fn(
            inputs.transpose(0, 1),
            weight,
            None,
            None,
            torch.tensor([0, token_count], device="cuda", dtype=torch.int32),
            None,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        )

        torch.testing.assert_close(first_actual[0, :, 0], expected[:, -2])
        torch.testing.assert_close(second_actual[0, :, 0], expected[:, -1])

    def test_combined_paged_cache_prefill_then_decode(self):
        token_count = 8
        prefix_length = token_count - 1
        channels = 3 * self.HEADS * self.HEAD_DIM
        inputs = torch.randn(token_count, channels, device="cuda", dtype=torch.bfloat16)
        conv_weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        converter = LinearCacheConverter(
            local_num_v_heads=self.HEADS,
            head_v_dim=self.HEAD_DIM,
            head_k_dim=self.HEAD_DIM,
            ssm_state_dtype=torch.float32,
            linear_conv_kernel_dim=4,
            qkv_size=channels,
            conv_state_dtype=torch.bfloat16,
        )
        cache = torch.zeros(
            2,
            converter.block_size_bytes // 2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ssm_cache = converter.get_ssm_state_tensor(cache)
        conv_cache = converter.get_conv_state_tensor(cache)
        block_map = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        prefix_cu = torch.tensor([0, prefix_length], device="cuda", dtype=torch.int32)

        prefix_qkv = causal_conv1d_fn(
            inputs[:prefix_length].transpose(0, 1),
            conv_weight,
            None,
            conv_cache.transpose(1, 2),
            prefix_cu,
            block_map,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        ).transpose(0, 1)
        prefix_q, prefix_k, prefix_v = (
            part.view(1, prefix_length, self.HEADS, self.HEAD_DIM).contiguous()
            for part in prefix_qkv.split(channels // 3, dim=-1)
        )
        common = {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "lower_bound": -5.0,
        }
        _, prefix_state, intermediate_states = chunk_kda(
            prefix_q,
            prefix_k,
            prefix_v,
            self.g[:, :prefix_length],
            self.beta[:, :prefix_length],
            initial_state=torch.zeros(
                1,
                self.HEADS,
                self.HEAD_DIM,
                self.HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            ),
            output_final_state=True,
            cu_seqlens=prefix_cu,
            safe_gate=True,
            return_intermediate_states=True,
            **common,
        )
        store_ssm_state_to_block_map(
            intermediate_states.float(),
            prefix_state.float(),
            torch.zeros(1, device="cuda", dtype=torch.int32),
            prefix_cu,
            block_map,
            ssm_cache,
            64,
            chunk_size=64,
        )
        torch.testing.assert_close(ssm_cache[1], prefix_state[0])

        decode_qkv = causal_conv1d_update(
            inputs[-1:].reshape(1, 1, channels).transpose(1, 2),
            conv_cache.transpose(1, 2),
            conv_weight,
            activation="silu",
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [token_count], device="cuda", dtype=torch.int32
            ),
        ).transpose(1, 2)
        decode_q, decode_k, decode_v = (
            part.view(1, 1, self.HEADS, self.HEAD_DIM).contiguous()
            for part in decode_qkv.split(channels // 3, dim=-1)
        )
        reference_decode, _ = fused_recurrent_kda(
            decode_q,
            decode_k,
            decode_v,
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=prefix_state.clone(),
            **common,
        )
        actual, _ = fused_recurrent_kda(
            decode_q,
            decode_k,
            decode_v,
            self.g[:, -1:],
            self.beta[:, -1:],
            initial_state=ssm_cache,
            block_map=block_map,
            seq_size_per_block=64,
            sequence_lengths=torch.tensor(
                [token_count], device="cuda", dtype=torch.int32
            ),
            **common,
        )

        full_qkv = causal_conv1d_fn(
            inputs.transpose(0, 1),
            conv_weight,
            None,
            None,
            torch.tensor([0, token_count], device="cuda", dtype=torch.int32),
            None,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            64,
        ).transpose(0, 1)
        full_q, full_k, full_v = (
            part.view(1, token_count, self.HEADS, self.HEAD_DIM).contiguous()
            for part in full_qkv.split(channels // 3, dim=-1)
        )
        torch.testing.assert_close(decode_qkv[0, 0], full_qkv[-1])
        expected, _ = chunk_kda(
            full_q,
            full_k,
            full_v,
            self.g,
            self.beta,
            output_final_state=False,
            safe_gate=True,
            **common,
        )
        self.assertGreater(self._cosine(reference_decode, expected[:, -1:]), 0.999)
        self.assertGreater(self._cosine(actual, expected[:, -1:]), 0.999)

    def test_bounded_prefill_matches_torch(self):
        lower_bound = -5.0
        expected, expected_state = self._reference(lower_bound)
        actual, actual_state = chunk_kda(
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            initial_state=torch.zeros_like(expected_state),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.a_log,
            dt_bias=self.dt_bias.flatten(),
            safe_gate=True,
            lower_bound=lower_bound,
        )

        self.assertGreater(self._cosine(actual, expected), 0.999)
        self.assertGreater(self._cosine(actual_state, expected_state), 0.999)

    def test_reuse_tail_fused_state_matches_cublas(self):
        """The one-launch chunk-state path must preserve CUBLAS numerics."""
        initial_state = torch.randn(
            1,
            self.HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        ).mul_(0.01)
        common = {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "A_log": self.a_log,
            "dt_bias": self.dt_bias.flatten(),
            "lower_bound": -5.0,
        }

        for token_count in (1, 6, 73, 128, 585):
            with self.subTest(token_count=token_count):
                shape = (1, token_count, self.HEADS, self.HEAD_DIM)
                q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                beta = (
                    torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
                    .float()
                    .sigmoid()
                )
                cu_seqlens = torch.tensor(
                    [0, token_count], device="cuda", dtype=torch.int32
                )

                chunk_output, chunk_state, chunk_h = chunk_kda(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=initial_state.clone(),
                    output_final_state=True,
                    return_intermediate_states=True,
                    cu_seqlens=cu_seqlens,
                    safe_gate=True,
                    **common,
                )
                fused_output, fused_state, fused_h = chunk_kda(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=initial_state.clone(),
                    output_final_state=True,
                    return_intermediate_states=True,
                    cu_seqlens=cu_seqlens,
                    safe_gate=True,
                    fuse_state_recurrence=True,
                    **common,
                )

                output_cosine = self._cosine(fused_output, chunk_output)
                state_cosine = self._cosine(fused_state, chunk_state)
                h_cosine = self._cosine(fused_h, chunk_h)
                print(
                    "reuse-tail fused-state/cublas",
                    token_count,
                    "output_cosine=",
                    output_cosine,
                    "state_cosine=",
                    state_cosine,
                    "h_cosine=",
                    h_cosine,
                    "output_max_abs=",
                    (fused_output.float() - chunk_output.float()).abs().max().item(),
                    "state_max_abs=",
                    (fused_state - chunk_state).abs().max().item(),
                )
                self.assertGreater(output_cosine, 0.999)
                self.assertGreater(state_cosine, 0.999)
                self.assertGreater(h_cosine, 0.999)

    def _reference_prefix_state(self):
        q, k, v, g, beta = self.q, self.k, self.v, self.g, self.beta
        self.q, self.k, self.v, self.g, self.beta = (
            q[:, :-1],
            k[:, :-1],
            v[:, :-1],
            g[:, :-1],
            beta[:, :-1],
        )
        try:
            return self._reference()[1]
        finally:
            self.q, self.k, self.v, self.g, self.beta = q, k, v, g, beta


if __name__ == "__main__":
    unittest.main()
