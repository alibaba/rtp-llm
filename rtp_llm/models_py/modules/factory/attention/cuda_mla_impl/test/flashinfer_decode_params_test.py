from types import SimpleNamespace
from unittest import TestCase, main, mock, skipUnless

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    MlaFlashInferDecodeOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferDecodeImpl,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.model_weight import W


CUDA_AVAILABLE = torch.cuda.is_available()


def _reference(
    sequence_lengths_plus_1: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
):
    safe_page_size = max(page_size, 1)
    sequence_lengths = sequence_lengths_plus_1.clamp_min(1).to(torch.int32)
    max_blocks = block_table.shape[1]
    pages_per_batch = torch.div(
        sequence_lengths + safe_page_size - 1,
        safe_page_size,
        rounding_mode="floor",
    ).clamp_max(max_blocks)
    decode_page_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            pages_per_batch.cumsum(0).to(torch.int32),
        )
    )
    page_indices = torch.cat(
        [
            block_table[batch, : int(page_count)]
            for batch, page_count in enumerate(pages_per_batch)
        ]
    )
    batch_size = sequence_lengths.numel()
    return {
        "batch_indice_d": torch.arange(batch_size, dtype=torch.int32),
        "page_indice_d": page_indices.to(torch.int32),
        "decode_page_indptr_d": decode_page_indptr,
        "paged_kv_last_page_len_d": ((sequence_lengths - 1) % safe_page_size + 1).to(
            torch.int32
        ),
        "qo_indptr_d": torch.arange(batch_size + 1, dtype=torch.int32),
        "kvlen_d": sequence_lengths,
        "positions_d": sequence_lengths - 1,
    }


@skipUnless(CUDA_AVAILABLE, "requires CUDA")
class FlashInferDecodeParamsTest(TestCase):
    @staticmethod
    def _reserve_decode_buffers(
        params, batch_size: int, max_blocks: int, page_size: int
    ) -> None:
        # The replay-only API deliberately forbids allocations. Mirror the
        # production capture path by reserving its persistent buffers first.
        params.fill_params(
            torch.empty(0, dtype=torch.int32),
            torch.zeros(batch_size, dtype=torch.int32),
            torch.ones(batch_size, dtype=torch.int32),
            torch.zeros((batch_size, max_blocks), dtype=torch.int32),
            max(page_size, 1),
            False,
        )

    def _run_case(
        self,
        batch_size: int,
        page_size: int,
        max_blocks: int,
    ) -> None:
        boundary_lengths = torch.tensor(
            [
                0,
                1,
                max(page_size - 1, 0),
                max(page_size, 0),
                max(page_size + 1, 1),
                max(2 * page_size - 1, 1),
                max(2 * page_size, 1),
                max(2 * page_size + 1, 1),
                65535,
                65536,
                65537,
            ],
            dtype=torch.int32,
        )
        repeats = (
            batch_size + boundary_lengths.numel() - 1
        ) // boundary_lengths.numel()
        sequence_lengths = boundary_lengths.repeat(repeats)[:batch_size]
        block_table = (
            torch.arange(batch_size * max_blocks, dtype=torch.int32).reshape(
                batch_size, max_blocks
            )
            * 17
            + torch.arange(batch_size, dtype=torch.int32).unsqueeze(1) * 13
            + 7
        )
        expected = _reference(sequence_lengths, block_table, page_size)

        params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._reserve_decode_buffers(params, batch_size, max_blocks, page_size)
        params.fill_decode_cuda_graph_params(
            sequence_lengths.cuda(), block_table.cuda(), page_size
        )
        host_plan_fits = bool(
            (
                torch.div(
                    sequence_lengths.clamp_min(1) + max(page_size, 1) - 1,
                    max(page_size, 1),
                    rounding_mode="floor",
                )
                <= max_blocks
            )
            .all()
            .item()
        )
        if host_plan_fits:
            params.fill_decode_cuda_graph_plan_host_params(
                sequence_lengths - 1,
                block_table.cuda(),
                page_size,
            )
        torch.cuda.synchronize()

        valid_pages = int(expected["decode_page_indptr_d"][-1])
        actual = {
            "batch_indice_d": params.batch_indice_d.cpu(),
            "page_indice_d": params.page_indice_d[:valid_pages].cpu(),
            "decode_page_indptr_d": params.decode_page_indptr_d.cpu(),
            "paged_kv_last_page_len_d": params.paged_kv_last_page_len_d.cpu(),
            "qo_indptr_d": params.qo_indptr_d.cpu(),
            "kvlen_d": params.kvlen_d.cpu(),
            "positions_d": params.positions_d.cpu(),
        }
        for name, expected_tensor in expected.items():
            torch.testing.assert_close(actual[name], expected_tensor, rtol=0, atol=0)
        if host_plan_fits:
            torch.testing.assert_close(
                params.qo_indptr_h,
                expected["qo_indptr_d"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                params.decode_page_indptr_h,
                expected["decode_page_indptr_d"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                params.kvlen_h,
                expected["kvlen_d"],
                rtol=0,
                atol=0,
            )
            self.assertEqual(params.page_indice_d.numel(), valid_pages)

    def test_parallel_and_serial_fallback_shapes(self) -> None:
        cases = [
            (1, 128, 513),
            (8, 1, 7),
            (31, 64, 8),
            (32, 128, 8),
            (33, 128, 8),
            (128, 128, 513),
            (1024, 4096, 17),
            (1025, 128, 4),
        ]
        for batch_size, page_size, max_blocks in cases:
            with self.subTest(
                batch_size=batch_size,
                page_size=page_size,
                max_blocks=max_blocks,
            ):
                self._run_case(batch_size, page_size, max_blocks)

    def test_non_positive_page_size_preserves_existing_clamp(self) -> None:
        self._run_case(batch_size=8, page_size=0, max_blocks=7)

    def test_host_plan_rejects_insufficient_block_table_capacity(self) -> None:
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._reserve_decode_buffers(params, batch_size=1, max_blocks=2, page_size=64)
        block_table = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
        params.fill_decode_cuda_graph_params(
            torch.tensor([129], dtype=torch.int32, device="cuda"),
            block_table,
            64,
        )

        with self.assertRaisesRegex(RuntimeError, "needs 3 cache pages"):
            params.fill_decode_cuda_graph_plan_host_params(
                torch.tensor([128], dtype=torch.int32),
                block_table,
                64,
            )

    def test_cuda_graph_replay_reads_live_inputs_without_reallocation(self) -> None:
        batch_size = 33
        page_size = 128
        max_blocks = 8
        sequence_lengths = torch.tensor(
            [1 + (batch * 29) % 700 for batch in range(batch_size)],
            dtype=torch.int32,
            device="cuda",
        )
        block_table = torch.arange(
            batch_size * max_blocks, dtype=torch.int32, device="cuda"
        ).reshape(batch_size, max_blocks)
        params = rtp_llm_ops.FlashInferMlaAttnParams()

        # Allocate all persistent buffers before capture.
        self._reserve_decode_buffers(params, batch_size, max_blocks, page_size)
        params.fill_decode_cuda_graph_params(sequence_lengths, block_table, page_size)
        torch.cuda.synchronize()
        output_names = (
            "batch_indice_d",
            "page_indice_d",
            "decode_page_indptr_d",
            "paged_kv_last_page_len_d",
            "qo_indptr_d",
            "kvlen_d",
            "positions_d",
        )
        pointers_before = {
            name: getattr(params, name).data_ptr() for name in output_names
        }

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            params.fill_decode_cuda_graph_params(
                sequence_lengths, block_table, page_size
            )

        for replay in range(10):
            updated_lengths = torch.tensor(
                [1 + (replay * 71 + batch * 43) % 1024 for batch in range(batch_size)],
                dtype=torch.int32,
            )
            updated_blocks = (
                torch.arange(batch_size * max_blocks, dtype=torch.int32).reshape(
                    batch_size, max_blocks
                )
                * (replay + 3)
                + 101
            )
            sequence_lengths.copy_(updated_lengths)
            block_table.copy_(updated_blocks)
            graph.replay()
            torch.cuda.synchronize()

            expected = _reference(updated_lengths, updated_blocks, page_size)
            valid_pages = int(expected["decode_page_indptr_d"][-1])
            actual = {
                "batch_indice_d": params.batch_indice_d.cpu(),
                "page_indice_d": params.page_indice_d[:valid_pages].cpu(),
                "decode_page_indptr_d": params.decode_page_indptr_d.cpu(),
                "paged_kv_last_page_len_d": params.paged_kv_last_page_len_d.cpu(),
                "qo_indptr_d": params.qo_indptr_d.cpu(),
                "kvlen_d": params.kvlen_d.cpu(),
                "positions_d": params.positions_d.cpu(),
            }
            for name, expected_tensor in expected.items():
                torch.testing.assert_close(
                    actual[name], expected_tensor, rtol=0, atol=0
                )

        self.assertEqual(
            pointers_before,
            {name: getattr(params, name).data_ptr() for name in output_names},
        )

    def test_generic_plan_snapshots_survive_back_to_back_host_updates(self) -> None:
        batch_size = 2
        page_size = 64
        max_blocks = 3
        empty_prefix = torch.empty(0, dtype=torch.int32)
        input_lengths = torch.ones(batch_size, dtype=torch.int32)
        first_lengths = torch.tensor([63, 127], dtype=torch.int32)
        second_lengths = torch.tensor([0, 64], dtype=torch.int32)
        first_blocks = torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int32
        )
        second_blocks = torch.tensor(
            [[30, 31, 32], [40, 41, 42]], dtype=torch.int32
        )

        params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._reserve_decode_buffers(params, batch_size, max_blocks, page_size)
        torch.cuda.synchronize()

        delayed_stream = torch.cuda.Stream()
        with torch.cuda.stream(delayed_stream):
            # Keep the first H2D queued while the CPU immediately prepares the
            # second replay. Without per-call snapshots, the first transfer
            # observes the second replay's in-place host metadata.
            torch.cuda._sleep(100_000_000)
            params.fill_params(
                empty_prefix,
                first_lengths,
                input_lengths,
                first_blocks,
                page_size,
                True,
            )
            first_pages = params.page_indice_d.clone()
            first_indptr = params.decode_page_indptr_d.clone()
            first_kvlen = params.kvlen_d.clone()
            first_positions = params.positions_d.clone()
            first_slots = params.slot_mapping.clone()

            params.fill_params(
                empty_prefix,
                second_lengths,
                input_lengths,
                second_blocks,
                page_size,
                True,
            )
            completed = delayed_stream.record_event()

        completed.synchronize()
        torch.testing.assert_close(
            first_pages.cpu(), torch.tensor([10, 20, 21], dtype=torch.int32)
        )
        torch.testing.assert_close(
            first_indptr.cpu(), torch.tensor([0, 1, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            first_kvlen.cpu(), torch.tensor([64, 128], dtype=torch.int32)
        )
        torch.testing.assert_close(
            first_positions.cpu(), first_lengths, rtol=0, atol=0
        )
        torch.testing.assert_close(
            first_slots.cpu(), torch.tensor([703, 1407], dtype=torch.int64)
        )

    def test_tokenspeed_compact_metadata_fusion_uses_current_stream(self) -> None:
        batch_size = 3
        page_size = 4
        max_blocks = 5
        sequence_lengths = torch.tensor([1, 7, 13], dtype=torch.int32)
        block_table = torch.tensor(
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
            ],
            dtype=torch.int32,
        )
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._reserve_decode_buffers(params, batch_size, max_blocks, page_size)
        sequence_lengths_d = sequence_lengths.cuda()
        block_table_d = block_table.cuda()
        dense_tables = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device="cuda"
        )
        dense_lengths = torch.full((batch_size,), -1, dtype=torch.int32, device="cuda")

        current_stream = torch.cuda.current_stream()
        metadata_stream = torch.cuda.Stream()
        metadata_stream.wait_stream(current_stream)
        with torch.cuda.stream(metadata_stream):
            params.fill_decode_cuda_graph_params(
                sequence_lengths_d, block_table_d, page_size
            )
            params.fill_tokenspeed_metadata(
                dense_tables, dense_lengths, batch_size, max_blocks
            )
            completed = metadata_stream.record_event()
        current_stream.wait_event(completed)

        torch.testing.assert_close(
            dense_tables.cpu(),
            torch.tensor(
                [
                    [11, 0, 0, 0, 0],
                    [21, 22, 0, 0, 0],
                    [31, 32, 33, 34, 0],
                ],
                dtype=torch.int32,
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dense_lengths.cpu(), sequence_lengths, rtol=0, atol=0
        )

    def test_tokenspeed_compact_metadata_fusion_is_graph_replay_safe(self) -> None:
        batch_size = 4
        page_size = 64
        max_blocks = 6
        sequence_lengths = torch.ones(batch_size, dtype=torch.int32, device="cuda")
        block_table = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device="cuda"
        )
        dense_tables = torch.empty_like(block_table)
        dense_lengths = torch.empty_like(sequence_lengths)
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._reserve_decode_buffers(params, batch_size, max_blocks, page_size)
        params.fill_decode_cuda_graph_params(sequence_lengths, block_table, page_size)
        params.fill_tokenspeed_metadata(
            dense_tables, dense_lengths, batch_size, max_blocks
        )
        torch.cuda.synchronize()
        output_pointers = (dense_tables.data_ptr(), dense_lengths.data_ptr())

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            params.fill_decode_cuda_graph_params(
                sequence_lengths, block_table, page_size
            )
            params.fill_tokenspeed_metadata(
                dense_tables, dense_lengths, batch_size, max_blocks
            )

        for replay in range(4):
            host_lengths = torch.tensor(
                [1, 64 + replay, 128 + replay * 64, 383 - replay * 32],
                dtype=torch.int32,
            )
            host_tables = (
                torch.arange(batch_size * max_blocks, dtype=torch.int32).reshape(
                    batch_size, max_blocks
                )
                + replay * 100
                + 1
            )
            sequence_lengths.copy_(host_lengths)
            block_table.copy_(host_tables)
            graph.replay()
            torch.cuda.synchronize()

            expected_tables = torch.zeros_like(host_tables)
            for batch, seq_len in enumerate(host_lengths.tolist()):
                live_blocks = min((seq_len + page_size - 1) // page_size, max_blocks)
                expected_tables[batch, :live_blocks] = host_tables[batch, :live_blocks]
            torch.testing.assert_close(
                dense_tables.cpu(), expected_tables, rtol=0, atol=0
            )
            torch.testing.assert_close(
                dense_lengths.cpu(), host_lengths, rtol=0, atol=0
            )
            self.assertEqual(
                output_pointers, (dense_tables.data_ptr(), dense_lengths.data_ptr())
            )

    def test_tokenspeed_compact_metadata_rejects_invalid_inputs(self) -> None:
        batch_size = 2
        max_blocks = 4
        block_tables = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device="cuda"
        )
        sequence_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        uninitialized = rtp_llm_ops.FlashInferMlaAttnParams()
        with self.assertRaisesRegex(RuntimeError, "page_indice_d must be defined"):
            uninitialized.fill_tokenspeed_metadata(
                block_tables, sequence_lengths, batch_size, max_blocks
            )

        params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._reserve_decode_buffers(params, batch_size, max_blocks, 64)
        invalid_calls = [
            (block_tables.cpu(), sequence_lengths, batch_size, max_blocks),
            (
                block_tables.to(torch.int64),
                sequence_lengths,
                batch_size,
                max_blocks,
            ),
            (block_tables.reshape(-1), sequence_lengths, batch_size, max_blocks),
            (block_tables[:, ::2], sequence_lengths, batch_size, 2),
            (block_tables, sequence_lengths.cpu(), batch_size, max_blocks),
            (
                block_tables,
                sequence_lengths.to(torch.int64),
                batch_size,
                max_blocks,
            ),
            (
                block_tables,
                sequence_lengths.reshape(batch_size, 1),
                batch_size,
                max_blocks,
            ),
            (block_tables, sequence_lengths, -1, max_blocks),
            (block_tables, sequence_lengths, batch_size, 0),
            (block_tables, sequence_lengths, batch_size, 2**31 - 1),
            (block_tables, sequence_lengths, batch_size + 1, max_blocks),
            (block_tables, sequence_lengths, batch_size, max_blocks + 1),
        ]
        for args in invalid_calls:
            with self.subTest(
                block_device=args[0].device,
                block_dtype=args[0].dtype,
                block_shape=tuple(args[0].shape),
                sequence_device=args[1].device,
                sequence_dtype=args[1].dtype,
                sequence_shape=tuple(args[1].shape),
                batch_size=args[2],
                padded_blocks=args[3],
            ):
                with self.assertRaises(RuntimeError):
                    params.fill_tokenspeed_metadata(*args)

        if torch.cuda.device_count() >= 2:
            other_device_tables = block_tables.to("cuda:1")
            with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
                params.fill_tokenspeed_metadata(
                    other_device_tables,
                    sequence_lengths,
                    batch_size,
                    max_blocks,
                )

    def test_rejects_invalid_inputs(self) -> None:
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        sequence_lengths = torch.ones(2, dtype=torch.int32, device="cuda")
        block_table = torch.ones((2, 4), dtype=torch.int32, device="cuda")

        invalid_calls = [
            (sequence_lengths.cpu(), block_table, 128),
            (sequence_lengths.to(torch.int64), block_table, 128),
            (sequence_lengths, block_table.cpu(), 128),
            (sequence_lengths, block_table.to(torch.int64), 128),
            (sequence_lengths, block_table.reshape(-1), 128),
            (sequence_lengths, block_table[:1], 128),
        ]
        for sequence_arg, block_arg, page_size in invalid_calls:
            with self.subTest(
                sequence_device=sequence_arg.device,
                sequence_dtype=sequence_arg.dtype,
                block_device=block_arg.device,
                block_dtype=block_arg.dtype,
                block_shape=tuple(block_arg.shape),
            ):
                with self.assertRaises(RuntimeError):
                    params.fill_decode_cuda_graph_params(
                        sequence_arg, block_arg, page_size
                    )

    def test_mla_replay_uses_device_bulk_metadata_for_q1(self) -> None:
        impl = object.__new__(MlaFlashInferDecodeImpl)
        impl.fmha_params = mock.Mock()
        impl.fmha_impl = mock.Mock()
        impl.seq_size_per_block = 64
        impl.prepare = mock.Mock()
        sequence_lengths_d = torch.tensor([18, 130], dtype=torch.int32, device="cuda")
        sequence_lengths_host = torch.tensor([17, 129], dtype=torch.int32)
        block_table_d = torch.arange(8, dtype=torch.int32, device="cuda").reshape(
            2, 4
        )
        inputs = SimpleNamespace(
            is_target_verify=False,
            is_prefill=False,
            sequence_lengths_plus_1_d=sequence_lengths_d,
            sequence_lengths_host=sequence_lengths_host,
            kv_cache_kernel_block_id_device=block_table_d,
        )

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl."
            "flashinfer_mla_wrapper.check_attention_inputs"
        ) as check_inputs:
            impl.prepare_cuda_graph(inputs)

        check_inputs.assert_called_once_with(inputs)
        impl.fmha_params.fill_decode_cuda_graph_params.assert_called_once_with(
            sequence_lengths_d, block_table_d, 64
        )
        impl.fmha_params.fill_decode_cuda_graph_plan_host_params.assert_called_once_with(
            sequence_lengths_host, block_table_d, 64
        )
        impl.fmha_impl.plan.assert_called_once_with(impl.fmha_params)
        impl.prepare.assert_not_called()
        self.assertIs(impl.attn_inputs, inputs)

    def test_mla_target_verify_replay_keeps_generic_planner(self) -> None:
        impl = object.__new__(MlaFlashInferDecodeImpl)
        impl.fmha_params = mock.Mock()
        impl.fmha_impl = mock.Mock()
        impl.seq_size_per_block = 64
        impl.prepare = mock.Mock()
        inputs = SimpleNamespace(
            is_target_verify=True,
            sequence_lengths_plus_1_d=torch.ones(
                1, dtype=torch.int32, device="cuda"
            ),
            sequence_lengths_host=torch.zeros(1, dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.zeros(
                (1, 4), dtype=torch.int32, device="cuda"
            ),
        )

        impl.prepare_cuda_graph(inputs)

        impl.prepare.assert_called_once_with(inputs, forbid_realloc=True)
        impl.fmha_params.fill_decode_cuda_graph_params.assert_not_called()
        impl.fmha_impl.plan.assert_not_called()

    def test_mla_q1_cuda_graph_attention_matches_eager_across_live_metadata(
        self,
    ) -> None:
        torch.manual_seed(37)
        batch_size = 2
        num_heads = 12
        kv_lora_rank = 512
        rope_dim = 64
        nope_dim = 128
        value_dim = 128
        page_size = 64
        max_blocks = 4
        max_context_len = page_size * max_blocks

        kc_weight = (
            torch.randn(
                num_heads,
                nope_dim,
                kv_lora_rank,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.02
        ).to(torch.bfloat16)
        vc_weight = (
            torch.randn(
                num_heads,
                kv_lora_rank,
                value_dim,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.02
        ).to(torch.bfloat16)
        q_nope = torch.randn(
            batch_size,
            num_heads,
            nope_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_pe = torch.randn(
            batch_size,
            num_heads,
            rope_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        cache = torch.randn(
            batch_size * max_blocks,
            page_size,
            kv_lora_rank + rope_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        layer_cache = SimpleNamespace(kv_cache_base=cache)
        block_table_d = torch.arange(
            batch_size * max_blocks, dtype=torch.int32, device="cuda"
        ).reshape(batch_size, max_blocks)
        block_table_h = block_table_d.cpu()

        op = MlaFlashInferDecodeOp(
            num_heads,
            kv_lora_rank,
            rope_dim,
            nope_dim,
            page_size,
            1.0,
            True,
            False,
            [{W.mla_kc: kc_weight, W.mla_vc: vc_weight}],
            max_bs=batch_size,
            max_context_len=max_context_len,
            num_tokens=batch_size,
            is_cuda_graph=True,
        )
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        capture_sequence_host = torch.tensor([127, 128], dtype=torch.int32)
        params.fill_params(
            torch.empty(0, dtype=torch.int32),
            capture_sequence_host,
            torch.ones(batch_size, dtype=torch.int32),
            block_table_h,
            page_size,
            False,
        )
        op.plan(params)
        op.forward(q_nope, q_pe, layer_cache, 0)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = op.forward(q_nope, q_pe, layer_cache, 0)

        def eager_reference(kv_lengths, physical_blocks):
            q_ckv = torch.einsum(
                "bhn,hnk->bhk", q_nope.float(), kc_weight.float()
            )
            outputs = []
            scale = (nope_dim + rope_dim) ** -0.5
            for batch, kv_length in enumerate(kv_lengths):
                page_count = (kv_length + page_size - 1) // page_size
                page_ids = physical_blocks[batch, :page_count].tolist()
                request_cache = cache[page_ids].reshape(
                    -1, kv_lora_rank + rope_dim
                )[:kv_length]
                ckv = request_cache[:, :kv_lora_rank].float()
                kpe = request_cache[:, kv_lora_rank:].float()
                logits = (
                    torch.einsum("hk,tk->ht", q_ckv[batch], ckv)
                    + torch.einsum("hr,tr->ht", q_pe[batch].float(), kpe)
                ) * scale
                attended = torch.softmax(logits, dim=-1) @ ckv
                outputs.append(
                    torch.einsum("hk,hkv->hv", attended, vc_weight.float())
                )
            return torch.stack(outputs)

        cases = [
            ([1, 2], [[3, 2, 1, 0], [4, 5, 6, 7]]),
            ([63, 64], [[0, 1, 2, 3], [7, 6, 5, 4]]),
            ([65, 127], [[2, 1, 0, 3], [4, 6, 5, 7]]),
            ([128, 129], [[1, 3, 0, 2], [5, 7, 4, 6]]),
        ]
        captured_signature = op._cuda_graph_plan_signature
        for kv_lengths, physical_blocks in cases:
            live_blocks_h = torch.tensor(physical_blocks, dtype=torch.int32)
            live_lengths_h = torch.tensor(kv_lengths, dtype=torch.int32)
            block_table_d.copy_(live_blocks_h)
            params.fill_decode_cuda_graph_params(
                live_lengths_h.cuda(), block_table_d, page_size
            )
            params.fill_decode_cuda_graph_plan_host_params(
                live_lengths_h - 1, block_table_d, page_size
            )
            op.plan(params)
            graph.replay()
            torch.cuda.synchronize()

            expected = eager_reference(kv_lengths, live_blocks_h)
            torch.testing.assert_close(
                graph_output.float(), expected, rtol=2e-2, atol=2e-2
            )
            self.assertEqual(op._cuda_graph_plan_signature, captured_signature)


if __name__ == "__main__":
    main()
