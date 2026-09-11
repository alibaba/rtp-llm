import unittest
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    mla_page_rr_cache as mla_page_rr_cache_module,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    build_flashmla_device_params,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardRoute,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import (
    MlaPageRRCacheAdapter,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_forward_test_utils import (
    PAGE_SIZE,
    CaseInputs,
    DeterministicPackedProjection,
    make_case_inputs,
    make_op,
    output_and_lse,
)
from rtp_llm.ops.compute_ops import LayerKVCache


class FlashMLADensePrefillPackedKVTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("K3 FlashMLA runtime test requires SM103a")

    @staticmethod
    def _make_inputs(
        q_lens: Sequence[int],
        prefix_lens: Sequence[int],
        *,
        strided_k_pe: bool = False,
    ) -> CaseInputs:
        inputs = make_case_inputs(q_lens, prefix_lens)
        q_lens = tuple(q_lens)
        prefix_lens = tuple(prefix_lens)
        max_blocks = max(
            (q_len + prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
            for q_len, prefix_len in zip(q_lens, prefix_lens, strict=True)
        )
        block_table = torch.zeros(
            (len(q_lens), max_blocks), dtype=torch.int32, device="cuda"
        )
        page_cursor = 0
        for owner, prefix_len in enumerate(prefix_lens):
            page_count = (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
            block_table[owner, :page_count] = torch.arange(
                page_cursor,
                page_cursor + page_count,
                dtype=torch.int32,
                device="cuda",
            )
            page_cursor += page_count

        max_q_len = max(q_lens)
        padding_offset = [
            owner * max_q_len - sum(q_lens[:owner])
            for owner, q_len in enumerate(q_lens)
            for _ in range(q_len)
        ]
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            total_tokens=sum(q_lens),
            input_lengths_host=torch.tensor(q_lens, dtype=torch.int32),
            prefix_lengths_host=torch.tensor(prefix_lens, dtype=torch.int32),
            input_lengths=torch.tensor(q_lens, dtype=torch.int32, device="cuda"),
            prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
            cu_seqlens=inputs.params.qo_indptr_d,
            cu_kv_seqlens=inputs.params.prefill_ragged_kv_len_indptr_d,
            padding_offset=torch.tensor(
                padding_offset, dtype=torch.int32, device="cuda"
            ),
            kv_cache_kernel_block_id_device_by_group=[block_table],
            kv_cache_kernel_block_id_device=block_table,
        )
        params = build_flashmla_device_params(attn_inputs, PAGE_SIZE)
        k_pe = inputs.k_pe
        if strided_k_pe:
            k_pe_storage = torch.empty(
                (sum(q_lens), k_pe.shape[1] + 17),
                dtype=k_pe.dtype,
                device=k_pe.device,
            )
            strided_k_pe_view = k_pe_storage.narrow(1, 11, k_pe.shape[1])
            strided_k_pe_view.copy_(k_pe)
            k_pe = strided_k_pe_view
        return CaseInputs(
            params,
            inputs.q,
            inputs.compressed_kv,
            k_pe,
            inputs.kv_cache,
        )

    @staticmethod
    def _reference(inputs: CaseInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent paged-cache gather and bottom-right causal attention."""
        outputs, lses = [], []
        q_start = 0
        params = inputs.params
        for owner, (q_len, prefix_len) in enumerate(
            zip(params.q_lens_host, params.prefix_lens_host, strict=True)
        ):
            current = inputs.compressed_kv[q_start : q_start + q_len]
            current_kpe = inputs.k_pe[q_start : q_start + q_len]
            if prefix_len:
                pages = params.attn_inputs.kv_cache_kernel_block_id_device[
                    owner, : (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
                ].long()
                prefix = inputs.kv_cache.kv_cache_base[pages].flatten(0, 1)[:prefix_len]
                latent = torch.cat((prefix[:, :512], current))
                kpe = torch.cat((prefix[:, 512:576], current_kpe))
            else:
                latent, kpe = current, current_kpe
            # DeterministicPackedProjection repeats the same K/V across heads.
            key = torch.cat((latent[:, :128], kpe), dim=-1).float()
            value = latent[:, 128:256].float()
            query = inputs.q[q_start : q_start + q_len].float()
            scores = torch.einsum("qhd,kd->hqk", query, key) * (192**-0.5)
            query_positions = prefix_len + torch.arange(q_len, device=query.device)
            key_positions = torch.arange(key.shape[0], device=query.device)
            scores.masked_fill_(
                key_positions[None, :] > query_positions[:, None], -torch.inf
            )
            outputs.append(torch.einsum("hqk,kd->qhd", scores.softmax(-1), value))
            lses.append(scores.logsumexp(-1).T)
            q_start += q_len
        return torch.cat(outputs), torch.cat(lses)

    def test_full_and_hybrid_match_causal_reference(self) -> None:
        cases = (
            ((2, 3), (0, 0), 256, False),
            ((2, 3), (128, 128), 384, False),
            ((2, 3), (128, 128), 256, False),
            ((2, 3, 1), (128, 0, 128), 256, False),
            ((1, 1, 1), (384, 0, 384), 256, False),
            ((2, 3), (0, 300), 128, True),
            ((1, 1), (127, 640), 256, False),
        )
        for q_lens, prefix_lens, capacity, strided in cases:
            with self.subTest(q=q_lens, prefix=prefix_lens, capacity=capacity):
                inputs = self._make_inputs(q_lens, prefix_lens, strided_k_pe=strided)
                expected, expected_lse = self._reference(inputs)
                for budget in (0, capacity):
                    op = make_op(expanded_kv_capacity_tokens=budget)
                    op.plan(inputs.params)
                    with mock.patch.object(
                        op,
                        "_create_kv_b_proj",
                        return_value=DeterministicPackedProjection(),
                    ):
                        output, lse = output_and_lse(op, inputs)
                    torch.testing.assert_close(
                        output.float(), expected, rtol=2e-2, atol=1e-3
                    )
                    torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)
                    if op._forward_plan.route is FlashMLAForwardRoute.HYBRID:
                        workspace = op._forward_workspace
                        self.assertLessEqual(workspace.packed_kv.shape[0], capacity)

    def test_workspace_reuse_and_replan_preserve_results(self) -> None:
        inputs = self._make_inputs((2, 3, 1), (384, 0, 384))
        op = make_op(expanded_kv_capacity_tokens=256)
        op.plan(inputs.params)
        with mock.patch.object(
            op, "_create_kv_b_proj", return_value=DeterministicPackedProjection()
        ):
            first, _ = output_and_lse(op, inputs)
            workspace = op._forward_workspace
            # Change the next layer's values so stale outputs cannot pass.
            inputs.compressed_kv.add_(0.25)
            second, lse = output_and_lse(op, inputs)
            self.assertIs(op._forward_workspace, workspace)
            self.assertFalse(torch.equal(first, second))
            expected, expected_lse = self._reference(inputs)
            torch.testing.assert_close(second.float(), expected, rtol=2e-2, atol=1e-3)
            torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)

            next_inputs = self._make_inputs((4,), (0,))
            op.plan(next_inputs.params)
            self.assertIsNone(op._forward_workspace)
            self.assertEqual(op._prefix_runtime_launches, ())
            output, lse = output_and_lse(op, next_inputs)
            expected, expected_lse = self._reference(next_inputs)
            torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=1e-3)
            torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)

    def _run_page_rr_reference_case(
        self,
        *,
        shard_size: int,
        q_lens: Sequence[int],
        prefix_lens: Sequence[int],
        expanded_kv_capacity_tokens: int,
        expected_prefix_launches: int | None,
    ) -> None:
        inputs = self._make_inputs(q_lens, prefix_lens)
        # Use unit-scale inputs for the random projection comparison.
        for tensor in (
            inputs.q,
            inputs.compressed_kv,
            inputs.k_pe,
            inputs.kv_cache.kv_cache_base,
        ):
            tensor.mul_(8)
        source = inputs.kv_cache.kv_cache_base
        canonical_table = inputs.params.attn_inputs.kv_cache_kernel_block_id_device
        width = max(
            (length + PAGE_SIZE * shard_size - 1) // (PAGE_SIZE * shard_size)
            for length in prefix_lens
        )
        batches, caches, tables = [], [], []
        for rank in range(shard_size):
            cache = source.new_full((len(prefix_lens) * width + 1, PAGE_SIZE, 576), -91)
            table = canonical_table.new_full((len(prefix_lens), width), -1)
            for request, length in enumerate(prefix_lens):
                for local_page, global_page in enumerate(
                    range(rank, (length + PAGE_SIZE - 1) // PAGE_SIZE, shard_size)
                ):
                    block = cache.shape[0] - 1 - request * width - local_page
                    table[request, local_page] = block
                    cache[block].copy_(source[canonical_table[request, global_page]])
            adapter = MlaPageRRCacheAdapter(PAGE_SIZE, shard_size, rank)
            batches.append(adapter._pack_prefix(cache, table, prefix_lens))
            caches.append(cache)
            tables.append(table)

        local_attn_inputs = SimpleNamespace(**vars(inputs.params.attn_inputs))
        local_attn_inputs.kv_cache_kernel_block_id_device = tables[0]
        local_attn_inputs.kv_cache_kernel_block_id_device_by_group = [tables[0]]
        local_cache = LayerKVCache()
        local_cache.kv_cache_base = caches[0]
        local_inputs = CaseInputs(
            build_flashmla_device_params(local_attn_inputs, PAGE_SIZE),
            inputs.q,
            inputs.compressed_kv,
            inputs.k_pe,
            local_cache,
        )
        op = make_op(
            expanded_kv_capacity_tokens=expanded_kv_capacity_tokens,
            external_prefix_cache=True,
        )
        op.plan(local_inputs.params)
        if expected_prefix_launches is not None:
            self.assertEqual(len(op._prefix_runtime_launches), expected_prefix_launches)
        reference = make_op(expanded_kv_capacity_tokens=0)
        reference.plan(inputs.params)
        projection = DeterministicPackedProjection()
        with mock.patch.object(reference, "_create_kv_b_proj", return_value=projection):
            expected_out, expected_lse = output_and_lse(reference, inputs)

        gathered = torch.stack(batches)

        def gather(local_payload, output, group):
            torch.testing.assert_close(local_payload, batches[0], rtol=0, atol=0)
            output.copy_(gathered.view_as(output))

        with mock.patch.object(
            mla_page_rr_cache_module.collective_torch,
            "all_gather_into",
            side_effect=gather,
        ) as collective, mock.patch.object(
            op, "_create_kv_b_proj", return_value=projection
        ), mock.patch.object(
            op,
            "_live_reuse_cache_page_indices",
            side_effect=AssertionError("canonical prefix must bypass rank-local pages"),
        ):
            canonical = MlaPageRRCacheAdapter(PAGE_SIZE, shard_size, 0).read_prefix(
                caches[0], tables[0], prefix_lens
            )
            actual_out, actual_lse = output_and_lse(op, local_inputs, canonical)
        self.assertEqual(collective.call_count, 1)
        torch.testing.assert_close(actual_out, expected_out, rtol=2e-2, atol=0.03125)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)

    def test_page_rr_tp2_tp8_and_tp16_match_replicated_canonical_attention(
        self,
    ) -> None:
        cases = (
            (2, [3, 5], [257, 129], 0, 0),
            (2, [3, 5], [257, 129], 128, None),
            (8, [1], [65536], 16384, 4),
            (8, [1, 1, 1], [1023, 1024, 1025], 0, 0),
            (16, [1, 1, 1], [2047, 2048, 2049], 0, 0),
            (2, [2, 3, 1], [129, 0, 127], 0, 0),
            (2, [1], [128], 128, 1),
            (2, [1], [512], 128, 4),
            (2, [2, 3, 1], [128, 0, 128], 256, 1),
        )
        for shard_size, q_lens, prefix_lens, capacity_tokens, launch_count in cases:
            with self.subTest(
                shard_size=shard_size,
                prefix_lens=prefix_lens,
                capacity_tokens=capacity_tokens,
            ):
                self._run_page_rr_reference_case(
                    shard_size=shard_size,
                    q_lens=q_lens,
                    prefix_lens=prefix_lens,
                    expanded_kv_capacity_tokens=capacity_tokens,
                    expected_prefix_launches=launch_count,
                )


if __name__ == "__main__":
    unittest.main()
