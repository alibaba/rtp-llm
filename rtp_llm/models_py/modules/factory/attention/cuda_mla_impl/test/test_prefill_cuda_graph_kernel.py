"""
Unit test: verify FlashInferMlaAttnParams.fill_prefill_cuda_graph_params (the
device-only fast path used by MTP draft prefill) produces outputs identical to
the CPU fillParams baseline.

The fast path replaces:
    self.fmha_params.fill_params(
        prefix_lengths_d, sequence_lengths_d, input_lengths_d,
        kv_cache_block_id_host, seq_size_per_block, forbid_realloc)

with a device-only kernel + cudaMemcpyAsync to populate host indptr mirrors,
costing one cudaStreamSynchronize instead of the 3+ syncs in the slow path.
"""

from unittest import TestCase, main, skipIf

import libth_transformer_config  # noqa: F401
import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def _has_cuda():
    return torch.cuda.is_available()


@skipIf(not _has_cuda(), "requires CUDA")
class TestPrefillCudaGraphKernel(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0")
        self.batch_size = 4
        # MTP draft prefill: ntpb = gen_num_per_cycle + 1
        self.num_tokens_per_bs = 4
        self.total_tokens = self.batch_size * self.num_tokens_per_bs
        self.seq_size_per_block = 64
        self.max_blocks = 16

        self.input_lengths = torch.full(
            (self.batch_size,),
            self.num_tokens_per_bs,
            dtype=torch.int32,
            device=self.device,
        )
        # Different prefix lengths per batch
        self.prefix_lengths = torch.tensor(
            [100, 250, 500, 800], dtype=torch.int32, device=self.device
        )
        self.block_table = torch.randint(
            0,
            1000,
            (self.batch_size, self.max_blocks),
            dtype=torch.int32,
            device=self.device,
        )

    def _build_reference(self):
        """Reference: CPU fillParams with prefill-shaped inputs."""
        ref_params = rtp_llm_ops.FlashInferMlaAttnParams()
        empty_seq = torch.empty(0, dtype=torch.int32, device=self.device)
        # fillParams takes host block table; pass cpu mirror.
        block_table_h = self.block_table.cpu().contiguous()
        ref_params.fill_params(
            self.prefix_lengths,
            empty_seq,
            self.input_lengths,
            block_table_h,
            self.seq_size_per_block,
            False,
        )
        torch.cuda.synchronize()
        return {
            "batch_indice": ref_params.batch_indice_d.clone(),
            "positions": ref_params.positions_d.clone(),
            "kvlen": ref_params.kvlen_d.clone(),
            "qo_indptr": ref_params.qo_indptr_d.clone(),
            "decode_page_indptr": ref_params.decode_page_indptr_d.clone(),
            "paged_kv_last_page_len": ref_params.paged_kv_last_page_len_d.clone(),
            "prefill_ragged_kv_len_indptr": ref_params.prefill_ragged_kv_len_indptr_d.clone(),
            "page_indice": ref_params.page_indice_d.clone(),
            "slot_mapping": ref_params.slot_mapping.clone(),
            "qo_indptr_h": ref_params.qo_indptr_h.clone(),
            "kvlen_h": ref_params.kvlen_h.clone(),
            "prefill_ragged_kv_len_indptr_h": ref_params.prefill_ragged_kv_len_indptr_h.clone(),
            "page_count": int(ref_params.decode_page_indptr_d[self.batch_size].item()),
        }

    def _setup_test_params(self):
        """Initialize test_params via fillParams (capture-time setup)."""
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        empty_seq = torch.empty(0, dtype=torch.int32, device=self.device)
        params.fill_params(
            self.prefix_lengths,
            empty_seq,
            self.input_lengths,
            self.block_table.cpu().contiguous(),
            self.seq_size_per_block,
            False,
        )
        torch.cuda.synchronize()
        return params

    def test_kernel_matches_cpu_fillparams(self):
        ref = self._build_reference()

        # Capture: fillParams sets shapes and host indptr.
        test_params = self._setup_test_params()

        # Replay: fast path device-only fill.
        test_params.fill_prefill_cuda_graph_params(
            self.input_lengths,
            self.prefix_lengths,
            self.block_table,
            self.seq_size_per_block,
            self.total_tokens,
        )
        torch.cuda.synchronize()

        # Shape contract.
        self.assertEqual(test_params.batch_indice_d.shape[0], self.total_tokens)
        self.assertEqual(test_params.positions_d.shape[0], self.total_tokens)
        self.assertEqual(test_params.kvlen_d.shape[0], self.batch_size)
        self.assertEqual(test_params.qo_indptr_d.shape[0], self.batch_size + 1)

        # Device value match.
        torch.testing.assert_close(
            test_params.batch_indice_d[: self.total_tokens],
            ref["batch_indice"][: self.total_tokens],
            msg="batch_indice",
        )
        torch.testing.assert_close(
            test_params.positions_d[: self.total_tokens],
            ref["positions"][: self.total_tokens],
            msg="positions",
        )
        torch.testing.assert_close(
            test_params.kvlen_d[: self.batch_size],
            ref["kvlen"][: self.batch_size],
            msg="kvlen",
        )
        torch.testing.assert_close(
            test_params.qo_indptr_d[: self.batch_size + 1],
            ref["qo_indptr"][: self.batch_size + 1],
            msg="qo_indptr",
        )
        torch.testing.assert_close(
            test_params.decode_page_indptr_d[: self.batch_size + 1],
            ref["decode_page_indptr"][: self.batch_size + 1],
            msg="decode_page_indptr",
        )
        torch.testing.assert_close(
            test_params.paged_kv_last_page_len_d[: self.batch_size],
            ref["paged_kv_last_page_len"][: self.batch_size],
            msg="paged_kv_last_page_len",
        )
        torch.testing.assert_close(
            test_params.prefill_ragged_kv_len_indptr_d[: self.batch_size + 1],
            ref["prefill_ragged_kv_len_indptr"][: self.batch_size + 1],
            msg="prefill_ragged_kv_len_indptr",
        )
        torch.testing.assert_close(
            test_params.page_indice_d[: ref["page_count"]],
            ref["page_indice"][: ref["page_count"]],
            msg="page_indice",
        )
        torch.testing.assert_close(
            test_params.slot_mapping[: self.total_tokens],
            ref["slot_mapping"][: self.total_tokens],
            msg="slot_mapping",
        )

        # Host indptr mirrors must match — FlashInfer plan() reads them.
        torch.testing.assert_close(
            test_params.qo_indptr_h[: self.batch_size + 1],
            ref["qo_indptr_h"][: self.batch_size + 1],
            msg="qo_indptr_h",
        )
        torch.testing.assert_close(
            test_params.kvlen_h[: self.batch_size],
            ref["kvlen_h"][: self.batch_size],
            msg="kvlen_h",
        )
        torch.testing.assert_close(
            test_params.prefill_ragged_kv_len_indptr_h[: self.batch_size + 1],
            ref["prefill_ragged_kv_len_indptr_h"][: self.batch_size + 1],
            msg="prefill_ragged_kv_len_indptr_h",
        )

    def test_kernel_clears_padding_slots_for_smaller_replay(self):
        """Replay may use a graph captured for more tokens than are live."""
        test_params = self._setup_test_params()

        # First write full-capacity metadata so the later smaller replay has
        # stale slot_mapping values to clear.
        test_params.fill_prefill_cuda_graph_params(
            self.input_lengths,
            self.prefix_lengths,
            self.block_table,
            self.seq_size_per_block,
            self.total_tokens,
        )
        torch.cuda.synchronize()

        live_tokens = self.total_tokens // 2
        replay_input_lengths = torch.tensor(
            [self.num_tokens_per_bs, self.num_tokens_per_bs, 0, 0],
            dtype=torch.int32,
            device=self.device,
        )
        replay_prefix_lengths = torch.tensor(
            [100, 250, 0, 0],
            dtype=torch.int32,
            device=self.device,
        )

        ref_params = rtp_llm_ops.FlashInferMlaAttnParams()
        ref_params.fill_params(
            replay_prefix_lengths,
            torch.empty(0, dtype=torch.int32, device=self.device),
            replay_input_lengths,
            self.block_table.cpu().contiguous(),
            self.seq_size_per_block,
            False,
        )

        test_params.fill_prefill_cuda_graph_params(
            replay_input_lengths,
            replay_prefix_lengths,
            self.block_table,
            self.seq_size_per_block,
            self.total_tokens,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            test_params.slot_mapping[:live_tokens],
            ref_params.slot_mapping[:live_tokens],
            msg="live slot_mapping",
        )
        self.assertTrue(
            torch.all(test_params.slot_mapping[live_tokens : self.total_tokens] == -1)
            .cpu()
            .item()
        )
        self.assertTrue(
            torch.all(test_params.batch_indice_d[live_tokens : self.total_tokens] == 0)
            .cpu()
            .item()
        )
        self.assertTrue(
            torch.all(test_params.positions_d[live_tokens : self.total_tokens] == 0)
            .cpu()
            .item()
        )
        torch.testing.assert_close(
            test_params.qo_indptr_d[: self.batch_size + 1],
            ref_params.qo_indptr_d[: self.batch_size + 1],
            msg="qo_indptr smaller replay",
        )


if __name__ == "__main__":
    main()
