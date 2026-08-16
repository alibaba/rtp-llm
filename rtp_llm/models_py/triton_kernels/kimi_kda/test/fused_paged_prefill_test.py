from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda import (
    kimi_kda_load_recurrent_state,
    kimi_kda_short_conv_paged_prefill,
    kimi_kda_store_recurrent_checkpoints,
    prepare_kimi_kda_recurrent_checkpoint_metadata,
    prepare_kimi_kda_short_conv_metadata,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KimiKDAFusedPagedPrefillTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260816)

    @staticmethod
    def _linear_block_map(batch: int, pages: int) -> torch.Tensor:
        # Deliberately sparse and non-monotonic within each row. The physical
        # IDs are not derived from an MLA/kernel block table.
        ids = torch.arange(1, batch * pages + 1, dtype=torch.int32)
        ids = ids.reshape(batch, pages)
        ids = torch.flip(ids, dims=(1,)) * 2 + 1
        return ids.cuda()

    @staticmethod
    def _chunk_kda():
        # The CUTLASS DSL Bazel wheels keep the public `cutlass` package in a
        # nested python_packages directory.  Add that runfiles directory just
        # as the production launcher does before importing cuLA.
        for root in tuple(sys.path):
            cutlass_packages = Path(root) / "nvidia_cutlass_dsl" / "python_packages"
            if (cutlass_packages / "cutlass" / "__init__.py").is_file():
                sys.path.insert(0, str(cutlass_packages))
                break
        from cula.kda import chunk_kda

        return chunk_kda

    def _conv_case(
        self,
        lengths: list[int],
        prefixes: list[int],
        page_size: int,
    ) -> None:
        from fla.modules.conv import ShortConvolution

        width = 4
        history_size = width - 1
        projection_size = 32
        channels = 3 * projection_size
        token_count = sum(lengths)
        max_absolute_length = max(
            prefix + length for prefix, length in zip(prefixes, lengths)
        )
        pages = (max_absolute_length + page_size - 1) // page_size + 1
        block_map = self._linear_block_map(len(lengths), pages)
        block_count = int(block_map.max().item()) + 3

        mixed_qkv = torch.randn(
            token_count, channels, dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.randn(
            channels, width, dtype=torch.bfloat16, device="cuda"
        )
        fla_convs = []
        for projection in range(3):
            layer = ShortConvolution(
                projection_size,
                width,
                bias=False,
                activation="silu",
                backend="triton",
                device=mixed_qkv.device,
                dtype=weight.dtype,
            ).eval()
            begin = projection * projection_size
            end = begin + projection_size
            with torch.no_grad():
                layer.weight.copy_(weight[begin:end].unsqueeze(1))
            fla_convs.append(layer)
        initial_cache = torch.randn(
            block_count,
            history_size,
            channels,
            dtype=torch.bfloat16,
            device="cuda",
        )
        actual_cache = initial_cache.clone()
        expected_cache = initial_cache.clone()
        cu_seqlens_host = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        cu_seqlens = cu_seqlens_host.cuda()
        prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device="cuda")

        expected_outputs: list[torch.Tensor] = []
        expected_final_states: list[torch.Tensor] = []
        start = 0
        host_map = block_map.cpu().tolist()
        for sequence, (length, prefix) in enumerate(zip(lengths, prefixes)):
            sequence_input = mixed_qkv[start : start + length]
            if prefix:
                initial_page = (prefix - 1) // page_size
                initial_block = host_map[sequence][initial_page]
                history = (
                    initial_cache[initial_block].transpose(0, 1).contiguous()
                )
            else:
                history = torch.zeros(
                    channels,
                    history_size,
                    dtype=mixed_qkv.dtype,
                    device=mixed_qkv.device,
                )
            projection_outputs: list[torch.Tensor] = []
            for projection in range(3):
                begin = projection * projection_size
                end = begin + projection_size
                fla_cache = None
                if prefix:
                    fla_cache = torch.zeros(
                        1,
                        projection_size,
                        width,
                        dtype=sequence_input.dtype,
                        device=sequence_input.device,
                    )
                    fla_cache[:, :, 1:].copy_(history[begin:end].unsqueeze(0))
                fla_output, _ = fla_convs[projection](
                    sequence_input[:, begin:end].unsqueeze(0),
                    cache=fla_cache,
                    output_final_state=True,
                )
                projection_outputs.append(fla_output.squeeze(0))
            expected_outputs.append(torch.cat(projection_outputs, dim=1))

            local_ends = list(range(page_size, length + 1, page_size))
            if not local_ends or local_ends[-1] != length:
                local_ends.append(length)
            combined = torch.cat((history.transpose(0, 1), sequence_input), dim=0)
            expected_final_states.append(combined[-history_size:])
            for local_end in local_ends:
                absolute_end = prefix + local_end
                page = (absolute_end - 1) // page_size
                block = host_map[sequence][page]
                combined_end = history_size + local_end
                expected_cache[block].copy_(
                    combined[combined_end - history_size : combined_end]
                )
            start += length

        metadata = prepare_kimi_kda_short_conv_metadata(
            cu_seqlens_host, mixed_qkv.device
        )
        *actual_planes, final_state = kimi_kda_short_conv_paged_prefill(
            mixed_qkv,
            weight,
            actual_cache,
            block_map,
            prefix_lengths,
            cu_seqlens,
            page_size,
            metadata,
            return_final_state=True,
        )
        self.assertIsNotNone(final_state)
        self.assertTrue(all(plane.is_contiguous() for plane in actual_planes))
        actual_output = torch.stack(actual_planes)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual_output,
            torch.stack(torch.cat(expected_outputs).split(projection_size, dim=1)),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            final_state,
            torch.stack(expected_final_states),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual_cache[0], initial_cache[0], rtol=0, atol=0)

    def test_fused_conv_matches_three_independent_semantics_mixed_batch(self) -> None:
        self._conv_case(
            lengths=[130, 77, 2],
            prefixes=[0, 128, 64],
            page_size=64,
        )

    def test_fused_conv_handles_short_tails_and_large_pages(self) -> None:
        self._conv_case(
            lengths=[1, 2, 3, 513],
            prefixes=[0, 512, 1024, 0],
            page_size=512,
        )

    def test_fused_conv_continues_from_temporary_state(self) -> None:
        page_size = 64
        projection_size = 32
        channels = 3 * projection_size
        mixed_qkv = torch.randn(
            130, channels, dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.randn(channels, 4, dtype=torch.bfloat16, device="cuda")
        block_map = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda")

        def run(
            x: torch.Tensor,
            prefix: int,
            cache: torch.Tensor,
            current: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cu_host = torch.tensor([0, x.shape[0]], dtype=torch.int32)
            q, k, v, final = kimi_kda_short_conv_paged_prefill(
                x,
                weight,
                cache,
                block_map,
                torch.tensor([prefix], dtype=torch.int32, device="cuda"),
                cu_host.cuda(),
                page_size,
                prepare_kimi_kda_short_conv_metadata(cu_host, x.device),
                current_conv_state=current,
                continuation_mask=(
                    torch.tensor([True], dtype=torch.bool, device="cuda")
                    if current is not None
                    else None
                ),
                return_final_state=True,
            )
            assert final is not None
            return torch.stack((q, k, v)), final

        full_output, full_final = run(
            mixed_qkv,
            0,
            torch.zeros(
                4, 3, channels, dtype=torch.bfloat16, device="cuda"
            ),
        )
        split_cache = torch.zeros(
            4, 3, channels, dtype=torch.bfloat16, device="cuda"
        )
        first_output, first_final = run(mixed_qkv[:64], 0, split_cache)
        split_cache[1].fill_(123)
        second_output, second_final = run(
            mixed_qkv[64:], 64, split_cache, first_final
        )

        torch.testing.assert_close(
            torch.cat((first_output, second_output), dim=1),
            full_output,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(second_final, full_final, rtol=0, atol=0)

    def test_recurrent_gather_and_store_use_physical_blocks(self) -> None:
        lengths = [130, 77, 2]
        prefixes = [0, 128, 64]
        page_size = 64
        batch = len(lengths)
        pages = 6
        heads = 2
        key_dim = 8
        value_dim = 12
        block_map = self._linear_block_map(batch, pages)
        block_count = int(block_map.max().item()) + 3
        initial_cache = torch.randn(
            block_count,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device="cuda",
        )
        actual_cache = initial_cache.clone()
        prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device="cuda")
        input_lengths_host = torch.tensor(lengths, dtype=torch.int32)
        input_lengths = input_lengths_host.cuda()
        host_map = block_map.cpu().tolist()

        actual_initial = kimi_kda_load_recurrent_state(
            prefix_lengths, block_map, actual_cache, page_size
        )
        expected_initial = torch.zeros_like(actual_initial)
        for sequence, prefix in enumerate(prefixes):
            if prefix:
                page = (prefix - 1) // page_size
                expected_initial[sequence].copy_(
                    initial_cache[host_map[sequence][page]]
                )
        torch.testing.assert_close(actual_initial, expected_initial, rtol=0, atol=0)

        metadata = prepare_kimi_kda_recurrent_checkpoint_metadata(
            input_lengths_host,
            torch.tensor(prefixes, dtype=torch.int32),
            page_size,
            actual_cache.device,
        )
        checkpoints = torch.randn(
            metadata.total_checkpoints,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device="cuda",
        )
        expected_cache = initial_cache.clone()
        checkpoint = 0
        for sequence, (prefix, length) in enumerate(zip(prefixes, lengths)):
            count = (length + page_size - 1) // page_size
            for local_checkpoint in range(count):
                local_end = min((local_checkpoint + 1) * page_size, length)
                page = (prefix + local_end - 1) // page_size
                expected_cache[host_map[sequence][page]].copy_(
                    checkpoints[checkpoint]
                )
                checkpoint += 1

        kimi_kda_store_recurrent_checkpoints(
            checkpoints,
            metadata,
            block_map,
            actual_cache,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(actual_cache[0], initial_cache[0], rtol=0, atol=0)

    def test_out_of_range_physical_id_cannot_access_cache(self) -> None:
        page_size = 64
        cache = torch.randn(
            4, 1, 4, 4, dtype=torch.float32, device="cuda"
        )
        initial_cache = cache.clone()
        prefix_lengths = torch.tensor([64], dtype=torch.int32, device="cuda")
        block_map = torch.tensor([[99]], dtype=torch.int32, device="cuda")

        gathered = kimi_kda_load_recurrent_state(
            prefix_lengths, block_map, cache, page_size
        )
        torch.testing.assert_close(gathered, torch.zeros_like(gathered), rtol=0, atol=0)

        lengths_host = torch.tensor([1], dtype=torch.int32)
        metadata = prepare_kimi_kda_recurrent_checkpoint_metadata(
            lengths_host,
            torch.tensor([64], dtype=torch.int32),
            page_size,
            cache.device,
        )
        kimi_kda_store_recurrent_checkpoints(
            torch.randn(
                1, 1, 4, 4, dtype=torch.float32, device="cuda"
            ),
            metadata,
            block_map,
            cache,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(cache, initial_cache, rtol=0, atol=0)

    @torch.inference_mode()
    def test_cula_checkpoints_include_each_terminal_without_final_state(self) -> None:
        chunk_kda = self._chunk_kda()

        lengths = [70, 130]
        interval = 64
        token_count = sum(lengths)
        heads = 2
        state_dim = 128
        q = torch.randn(
            1,
            token_count,
            heads,
            state_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        gate = torch.randn_like(q)
        beta = torch.randn(
            1, token_count, heads, dtype=torch.bfloat16, device="cuda"
        )
        initial = torch.randn(
            len(lengths),
            heads,
            state_dim,
            state_dim,
            dtype=torch.float32,
            device="cuda",
        )
        cu_host = torch.tensor([0, 70, 200], dtype=torch.int32)
        cu_device = cu_host.cuda()
        checkpoint_count = sum(
            (length + interval - 1) // interval for length in lengths
        )
        alog = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(
            heads * state_dim, dtype=torch.float32, device="cuda"
        )

        def run(output_final_state: bool):
            checkpoint_buffer = torch.empty(
                1,
                checkpoint_count,
                heads,
                state_dim,
                state_dim,
                dtype=torch.float32,
                device="cuda",
            )
            output, final_state, checkpoints = chunk_kda(
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
                checkpoint_states=checkpoint_buffer,
            )
            self.assertEqual(checkpoints.data_ptr(), checkpoint_buffer.data_ptr())
            return output, final_state, checkpoints

        output_without_final, no_final, checkpoints_without_final = run(False)
        output_with_final, final, checkpoints_with_final = run(True)
        self.assertIsNone(no_final)
        self.assertIsNotNone(final)
        torch.testing.assert_close(
            output_without_final, output_with_final, rtol=0, atol=0
        )
        torch.testing.assert_close(
            checkpoints_without_final, checkpoints_with_final, rtol=0, atol=0
        )
        assert final is not None
        torch.testing.assert_close(
            final[0], checkpoints_without_final[0, 1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            final[1], checkpoints_without_final[0, 4], rtol=0, atol=0
        )

    @torch.inference_mode()
    def test_two_round_physical_cache_reload_matches_unsplit_kda(self) -> None:
        chunk_kda = self._chunk_kda()
        page_size = 64
        heads = 1
        state_dim = 128
        channels = 3 * state_dim
        lengths = [130, 77]
        token_count = sum(lengths)
        block_map = self._linear_block_map(len(lengths), 4)
        block_count = int(block_map.max().item()) + 3
        mixed_qkv = torch.randn(
            token_count, channels, dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.randn(channels, 4, dtype=torch.bfloat16, device="cuda")
        raw_gate = torch.randn(
            token_count, heads, state_dim, dtype=torch.bfloat16, device="cuda"
        )
        raw_beta = torch.randn(
            token_count, heads, dtype=torch.bfloat16, device="cuda"
        )
        a_log = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(
            heads * state_dim, dtype=torch.float32, device="cuda"
        )
        initial_conv = torch.randn(
            block_count, 3, channels, dtype=torch.bfloat16, device="cuda"
        )
        initial_ssm = torch.randn(
            block_count,
            heads,
            state_dim,
            state_dim,
            dtype=torch.float32,
            device="cuda",
        )

        def run_batch(
            batch_qkv: torch.Tensor,
            batch_gate: torch.Tensor,
            batch_beta: torch.Tensor,
            batch_lengths: list[int],
            prefixes: list[int],
            linear_block_map: torch.Tensor,
            conv_cache: torch.Tensor,
            ssm_cache: torch.Tensor,
        ) -> torch.Tensor:
            cu_host = torch.tensor(
                [0, *torch.tensor(batch_lengths).cumsum(0).tolist()],
                dtype=torch.int32,
            )
            cu_device = cu_host.cuda()
            prefix_device = torch.tensor(
                prefixes, dtype=torch.int32, device="cuda"
            )
            lengths_device = torch.tensor(
                batch_lengths, dtype=torch.int32, device="cuda"
            )
            q, k, v, final_conv = kimi_kda_short_conv_paged_prefill(
                batch_qkv,
                weight,
                conv_cache,
                linear_block_map,
                prefix_device,
                cu_device,
                page_size,
                prepare_kimi_kda_short_conv_metadata(cu_host, batch_qkv.device),
            )
            self.assertIsNone(final_conv)
            checkpoint_metadata = prepare_kimi_kda_recurrent_checkpoint_metadata(
                torch.tensor(batch_lengths, dtype=torch.int32),
                torch.tensor(prefixes, dtype=torch.int32),
                page_size,
                batch_qkv.device,
            )
            checkpoints = torch.empty(
                1,
                checkpoint_metadata.total_checkpoints,
                heads,
                state_dim,
                state_dim,
                dtype=torch.float32,
                device="cuda",
            )
            output, final_state, published = chunk_kda(
                q.reshape(1, -1, heads, state_dim),
                k.reshape(1, -1, heads, state_dim),
                v.reshape(1, -1, heads, state_dim),
                batch_gate.reshape(1, -1, heads, state_dim),
                batch_beta.reshape(1, -1, heads),
                scale=state_dim**-0.5,
                initial_state=kimi_kda_load_recurrent_state(
                    prefix_device, linear_block_map, ssm_cache, page_size
                ),
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                cu_seqlens=cu_device,
                cu_seqlens_cpu=cu_host,
                safe_gate=True,
                lower_bound=-5.0,
                disable_recompute=False,
                use_intracard_cp=False,
                A_log=a_log,
                dt_bias=dt_bias,
                checkpoint_interval=page_size,
                checkpoint_states=checkpoints,
            )
            self.assertIsNone(final_state)
            self.assertEqual(published.data_ptr(), checkpoints.data_ptr())
            kimi_kda_store_recurrent_checkpoints(
                checkpoints,
                checkpoint_metadata,
                linear_block_map,
                ssm_cache,
            )
            return output.reshape(-1, heads, state_dim)

        unsplit_conv = initial_conv.clone()
        unsplit_ssm = initial_ssm.clone()
        unsplit_output = run_batch(
            mixed_qkv,
            raw_gate,
            raw_beta,
            lengths,
            [0, 0],
            block_map,
            unsplit_conv,
            unsplit_ssm,
        )

        split_conv = initial_conv.clone()
        split_ssm = initial_ssm.clone()
        split_output = torch.empty_like(unsplit_output)
        source_starts = [0, lengths[0]]
        processed = [0, 0]
        for round_lengths in ([64, 64], [66, 13]):
            pieces = []
            gate_pieces = []
            beta_pieces = []
            for sequence, round_length in enumerate(round_lengths):
                source = source_starts[sequence] + processed[sequence]
                pieces.append(mixed_qkv[source : source + round_length])
                gate_pieces.append(raw_gate[source : source + round_length])
                beta_pieces.append(raw_beta[source : source + round_length])
            round_output = run_batch(
                torch.cat(pieces),
                torch.cat(gate_pieces),
                torch.cat(beta_pieces),
                list(round_lengths),
                list(processed),
                block_map,
                split_conv,
                split_ssm,
            )
            packed_start = 0
            for sequence, round_length in enumerate(round_lengths):
                source = source_starts[sequence] + processed[sequence]
                split_output[source : source + round_length].copy_(
                    round_output[packed_start : packed_start + round_length]
                )
                processed[sequence] += round_length
                packed_start += round_length

        torch.cuda.synchronize()
        torch.testing.assert_close(split_output, unsplit_output, rtol=0, atol=0)
        torch.testing.assert_close(split_conv, unsplit_conv, rtol=0, atol=0)
        torch.testing.assert_close(split_ssm, unsplit_ssm, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
