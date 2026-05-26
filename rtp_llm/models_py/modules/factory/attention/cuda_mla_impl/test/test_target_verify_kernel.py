"""
Unit test: verify fillTargetVerifyCudaGraphParams (GPU kernel, replay path)
produces identical outputs to fillParams (CPU path, capture path).

The fast path replaces:
    self.fmha_params.fill_params(attn_inputs, seq_size_per_block, forbid_realloc)
when attn_inputs.is_target_verify=True. The reference output is exactly the
state of SparseMlaParams after fill_params with target-verify inputs.

Two scenarios are tested:
1. Fresh state: kernel called right after fill_params (capture).
2. After decode mutation: a decode-shaped fill_params runs in between, mutating
   tensor shapes; the kernel must restore them. SparseMlaImpl handles BOTH
   decode and target_verify for sparse MLA (MlaFlashInferDecodeImpl rejects
   when attn_configs.is_sparse=True), so this scenario IS the e2e flow.
"""

from unittest import TestCase, main, skipIf

# libth_transformer_config must be loaded first to register types referenced
# by librtp_compute_ops's pybind default args.
import libth_transformer_config  # noqa: F401
import torch
from librtp_compute_ops import PyAttentionInputs

from rtp_llm.ops.compute_ops import rtp_llm_ops


def _has_cuda():
    return torch.cuda.is_available()


@skipIf(not _has_cuda(), "requires CUDA")
class TestSparseMlaTargetVerifyParams(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0")
        self.batch_size = 4
        self.num_tokens_per_bs = 4  # MTP: gen_num_per_cycle (3) + 1
        self.total_tokens = self.batch_size * self.num_tokens_per_bs
        self.seq_size_per_block = 64
        self.max_blocks = 16

        self.input_lengths = torch.full(
            (self.batch_size,),
            self.num_tokens_per_bs,
            dtype=torch.int32,
            device=self.device,
        )
        # Different prefix lengths per batch to exercise position/page math
        self.prefix_lengths = torch.tensor(
            [100, 250, 500, 800], dtype=torch.int32, device=self.device
        )
        # Block table — 2D [batch, max_blocks]
        self.block_table = torch.randint(
            0,
            1000,
            (self.batch_size, self.max_blocks),
            dtype=torch.int32,
            device=self.device,
        )

    def _build_target_verify_inputs(self):
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.is_target_verify = True
        attn_inputs.input_lengths = self.input_lengths
        attn_inputs.prefix_lengths = self.prefix_lengths
        attn_inputs.kv_cache_block_id_host = self.block_table.cpu().contiguous()
        attn_inputs.kv_cache_block_id_device = self.block_table
        attn_inputs.kv_cache_kernel_block_id_device = self.block_table
        attn_inputs.kv_cache_kernel_block_id_host = self.block_table.cpu().contiguous()
        return attn_inputs

    def _build_decode_inputs(self):
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = False
        attn_inputs.is_target_verify = False
        attn_inputs.sequence_lengths = torch.tensor(
            [400, 500, 600, 700], dtype=torch.int32
        )
        attn_inputs.input_lengths = torch.ones(self.batch_size, dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_block_id_host = self.block_table.cpu().contiguous()
        attn_inputs.kv_cache_block_id_device = self.block_table
        attn_inputs.kv_cache_kernel_block_id_device = self.block_table
        attn_inputs.kv_cache_kernel_block_id_host = self.block_table.cpu().contiguous()
        return attn_inputs

    def _build_reference(self):
        """Compute reference outputs using CPU fillParams with target-verify inputs.

        This is exactly the call site the GPU kernel replaces:
            self.fmha_params.fill_params(attn_inputs, seq_size_per_block, forbid_realloc)
        with attn_inputs.is_target_verify=True.
        """
        ref_params = rtp_llm_ops.SparseMlaParams()
        ref_params.fill_params(
            self._build_target_verify_inputs(), self.seq_size_per_block, False
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
            "expanded_seq_lens": ref_params.expanded_seq_lens.clone(),
            "topk_indices_offset": ref_params.topk_indices_offset.clone(),
            "ks": ref_params.ks.clone(),
            "ke": ref_params.ke.clone(),
            "slot_mapping": ref_params.slot_mapping.clone(),
            "page_count": int(ref_params.decode_page_indptr_d[self.batch_size].item()),
        }

    def _assert_match(self, test_params, ref):
        """Compare test_params outputs against reference (target-verify CPU)."""
        self.assertEqual(test_params.batch_indice_d.shape[0], self.total_tokens)
        self.assertEqual(
            test_params.expanded_seq_lens.shape[0],
            self.total_tokens,
            "expanded_seq_lens shape — _refresh_paged_mqa_schedule_metadata "
            "skips when numel==0",
        )
        self.assertEqual(
            test_params.batch_indice_h.shape[0],
            self.total_tokens,
            "batch_indice_h shape — plan() reads .shape[0] for sched_meta count",
        )

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
            test_params.expanded_seq_lens[: self.total_tokens],
            ref["expanded_seq_lens"][: self.total_tokens],
            msg="expanded_seq_lens",
        )
        torch.testing.assert_close(
            test_params.topk_indices_offset[: self.total_tokens],
            ref["topk_indices_offset"][: self.total_tokens],
            msg="topk_indices_offset",
        )
        torch.testing.assert_close(
            test_params.ks[: self.total_tokens],
            ref["ks"][: self.total_tokens],
            msg="ks",
        )
        torch.testing.assert_close(
            test_params.ke[: self.total_tokens],
            ref["ke"][: self.total_tokens],
            msg="ke",
        )
        torch.testing.assert_close(
            test_params.slot_mapping[: self.total_tokens],
            ref["slot_mapping"][: self.total_tokens],
            msg="slot_mapping",
        )

    def test_kernel_matches_cpu_fresh_state(self):
        """After fill_params (target_verify capture), call the GPU kernel
        immediately. Output should still match the CPU reference."""
        ref = self._build_reference()
        test_params = rtp_llm_ops.SparseMlaParams()
        test_params.fill_params(
            self._build_target_verify_inputs(), self.seq_size_per_block, False
        )
        torch.cuda.synchronize()
        self.assertEqual(test_params.target_verify_total_tokens, self.total_tokens)
        test_params.fill_target_verify_cuda_graph_params(
            self.input_lengths,
            self.prefix_lengths,
            self.block_table,
            self.seq_size_per_block,
        )
        torch.cuda.synchronize()
        self._assert_match(test_params, ref)

    def _build_decode_reference(self):
        """Reference: CPU fillParams with decode-shaped inputs."""
        ref_params = rtp_llm_ops.SparseMlaParams()
        ref_params.fill_params(
            self._build_decode_inputs(), self.seq_size_per_block, False
        )
        torch.cuda.synchronize()
        return {
            "batch_indice": ref_params.batch_indice_d.clone(),
            "positions": ref_params.positions_d.clone(),
            "kvlen": ref_params.kvlen_d.clone(),
            "qo_indptr": ref_params.qo_indptr_d.clone(),
            "decode_page_indptr": ref_params.decode_page_indptr_d.clone(),
            "paged_kv_last_page_len": ref_params.paged_kv_last_page_len_d.clone(),
            "page_indice": ref_params.page_indice_d.clone(),
            "expanded_seq_lens": ref_params.expanded_seq_lens.clone(),
            "page_count": int(ref_params.decode_page_indptr_d[self.batch_size].item()),
        }

    def test_decode_fast_path_matches_cpu(self):
        """SparseMla decode fast path (used by MTP draft) outputs match
        CPU fillParams baseline."""
        ref = self._build_decode_reference()
        test_params = rtp_llm_ops.SparseMlaParams()
        test_params.fill_params(
            self._build_decode_inputs(), self.seq_size_per_block, False
        )
        torch.cuda.synchronize()

        seqlen_plus_1 = (
            self._build_decode_inputs().sequence_lengths.to(self.device) + 1
        ).to(torch.int32)
        test_params.fill_sparse_mla_decode_cuda_graph_params(
            seqlen_plus_1,
            self.block_table,
            self.seq_size_per_block,
        )
        torch.cuda.synchronize()

        # Shape contract
        self.assertEqual(test_params.batch_indice_d.shape[0], self.batch_size)
        self.assertEqual(test_params.kvlen_d.shape[0], self.batch_size)
        self.assertEqual(test_params.ks.numel(), 0)
        self.assertEqual(test_params.ke.numel(), 0)
        self.assertEqual(test_params.topk_indices_offset.numel(), 0)
        self.assertEqual(test_params.expanded_seq_lens.shape[0], self.batch_size)

        # Value match against CPU baseline
        torch.testing.assert_close(
            test_params.batch_indice_d[: self.batch_size],
            ref["batch_indice"][: self.batch_size],
            msg="batch_indice",
        )
        torch.testing.assert_close(
            test_params.positions_d[: self.batch_size],
            ref["positions"][: self.batch_size],
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
            test_params.page_indice_d[: ref["page_count"]],
            ref["page_indice"][: ref["page_count"]],
            msg="page_indice",
        )

    def test_kernel_matches_cpu_after_decode_mutation(self):
        """Realistic e2e: SparseMlaImpl handles both decode and target_verify
        (sparse MLA selects SparseMlaImpl for both). Decode mutates shapes
        between target_verify replays; the kernel must restore them.

        Output should still match the CPU target_verify reference.
        """
        ref = self._build_reference()
        test_params = rtp_llm_ops.SparseMlaParams()
        # 1) Capture: fill_params with target verify
        test_params.fill_params(
            self._build_target_verify_inputs(), self.seq_size_per_block, False
        )
        torch.cuda.synchronize()
        # 2) Decode step in between mutates shapes
        test_params.fill_params(
            self._build_decode_inputs(), self.seq_size_per_block, False
        )
        torch.cuda.synchronize()
        self.assertEqual(test_params.batch_indice_d.shape[0], self.batch_size)
        # 3) Replay: GPU kernel must restore target_verify state
        test_params.fill_target_verify_cuda_graph_params(
            self.input_lengths,
            self.prefix_lengths,
            self.block_table,
            self.seq_size_per_block,
        )
        torch.cuda.synchronize()
        self._assert_match(test_params, ref)


if __name__ == "__main__":
    main()
