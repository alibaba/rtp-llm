"""CPU contracts for V4.1 compression, query scaling, and shared selections."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    AttentionV41FP8,
    compress_pairs,
    mask_candidate_logits,
    rms_norm,
    rope_only,
    select_candidate_blocks,
)


class AttentionV41Test(unittest.TestCase):
    def test_cp_gather_updates_input_with_symm_memory_wrapper(self):
        from rtp_llm.models_py.distributed import collective_torch as collectives

        value = torch.tensor([[1.0, 2.0]])
        source = SimpleNamespace(
            _cp_ctx=SimpleNamespace(cp_size=4, kv_cache_sharded=True)
        )

        class SymmetricCommunicator:
            def should_torch_symm_mem_allreduce(self, tensor):
                return True

            def all_reduce(self, tensor, out=None):
                reduced = tensor * 4
                if out is not None:
                    out.copy_(reduced)
                    return out
                return reduced

        symm = SimpleNamespace(
            get_symm_mem_communicator=lambda: SymmetricCommunicator()
        )
        with patch.object(
            collectives, "_get_rocm_rccl", return_value=None
        ), patch.object(collectives, "_get_symm_mem", return_value=symm):
            actual = AttentionV41FP8._gather_shards(source, value)
            projected = torch.tensor([[3.0, 5.0]])
            AttentionV41FP8._prefill_output_all_reduce(
                SimpleNamespace(tp_size=4), projected
            )
        self.assertIs(actual, value)
        torch.testing.assert_close(value, torch.tensor([[4.0, 8.0]]))
        torch.testing.assert_close(projected, torch.tensor([[12.0, 20.0]]))

    def test_v41_metadata_allocates_swa_and_keeps_owner_block_tables(self):
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv4.decode.forward import (
            decode_metadata_compress_ratios,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
        )

        args = SimpleNamespace(
            v41_config={"kv_source_layer_ids": [2, 8, 14, 20]},
            compress_ratios=[0, 0, 2, 1],
            n_layers=4,
        )
        specs = {
            SWA_KV: (256, 128, 3),
            CSA_KV: (64, 128, 3),
            HCA_KV: (128, 128, 3),
            INDEXER_KV: (128, 128, 3),
            CSA_STATE: (8, 128, 3),
        }
        meta = allocate_decode_metadata_fp8(
            max_batch_size=2,
            q_len=6,
            window_size=128,
            head_dim=512,
            max_seq_len=256,
            compress_ratios=decode_metadata_compress_ratios(args),
            index_topk=512,
            device=torch.device("cpu"),
            paged_pool_specs=specs,
        )
        self.assertEqual(meta.slot_mapping_compressed, {})
        self.assertEqual(set(meta.pool_block_tables), set(specs))
        torch.testing.assert_close(
            meta.req_id_per_token, torch.tensor([0] * 6 + [1] * 6, dtype=torch.int32)
        )
        args.v41_config = None
        args.compress_ratios = [0, 4, 128, 0]
        self.assertEqual(decode_metadata_compress_ratios(args), [0, 4, 128, 0])

    def test_pair_pooling_is_channelwise_and_does_not_round_before_norm(self):
        torch.manual_seed(71)
        values = torch.randn(7, 2, 512) * 7
        scores = torch.randn_like(values) * 2
        norm = torch.randn(512).bfloat16()
        actual = compress_pairs(values, scores, norm, 1e-6)
        expected = torch.empty(7, 512)
        for group in range(7):
            for channel in range(512):
                p = torch.softmax(scores[group, :, channel], dim=0)
                expected[group, channel] = (values[group, :, channel] * p).sum()
        expected = rms_norm(expected, norm, 1e-6).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        rounded_early = rms_norm(
            (values * scores.softmax(1)).sum(1).bfloat16(), norm, 1e-6
        )
        self.assertTrue((actual != rounded_early).any())

    def test_rope_does_not_apply_post_q_normalization(self):
        q = torch.tensor([[[3.0, 4.0, 5.0, 6.0]]], dtype=torch.bfloat16)
        rotated = rope_only(q.clone(), torch.tensor([[1j]]), 2)
        torch.testing.assert_close(
            rotated, torch.tensor([[[3.0, 4.0, -6.0, 5.0]]], dtype=q.dtype)
        )
        torch.testing.assert_close(
            rotated.float().square().sum(), q.float().square().sum()
        )

    def test_topk_consumers_reuse_source_selection_without_reindexing(self):
        selected = torch.tensor([[2, 0, -1]], dtype=torch.int32)
        consumer = SimpleNamespace(
            is_index_source=False,
            index_source_layer_id=20,
            _shared_attention={"topk": {20: selected}},
        )
        actual = AttentionV41FP8._select_indices(consumer, None, None, None, None)
        self.assertIs(actual, selected)

    def test_short_context_selects_completed_groups_only(self):
        source = SimpleNamespace(
            is_index_source=True,
            index_source_layer_id=2,
            kv_source_layer_id=2,
            layer_id=2,
            compress_ratio=2,
            index_topk=4,
            _shared_attention={"global": {2: [(None, torch.zeros(3, 128))]}},
        )
        positions = torch.tensor([0, 1, 2, 3, 4, 5])
        actual = AttentionV41FP8._select_indices(
            source, torch.zeros(6, 1), None, positions, torch.zeros(6, dtype=torch.long)
        )
        expected = torch.tensor(
            [
                [-1, -1, -1, -1],
                [0, -1, -1, -1],
                [0, -1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, 2, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)

    def test_continuation_pair_uses_previous_fp32_state(self):
        attn = AttentionV41FP8.__new__(AttentionV41FP8)
        torch.nn.Module.__init__(attn)
        attn.layer_id = 2
        attn.kv_source_layer_id = 2
        attn.compress_ratio = 2
        attn.head_dim = 4
        attn.rope_head_dim = 2
        attn.eps = 1e-6
        attn.global_wkv = torch.eye(4, dtype=torch.bfloat16)
        attn.global_wgate = torch.eye(4, dtype=torch.bfloat16) * 0.125
        attn.global_norm = torch.ones(4, dtype=torch.bfloat16)
        attn.index_wk = torch.eye(4, dtype=torch.bfloat16)
        attn.index_k_norm = torch.ones(4, dtype=torch.bfloat16)
        attn.freqs_cis = torch.ones(8, 1, dtype=torch.complex64)
        attn._shared_attention = {"layers": {2: attn}}
        attn._source_pool = lambda region: None
        attn._write_states = lambda *args: None
        first = torch.tensor([[1.25, -2.5, 4.75, 3.0]], dtype=torch.bfloat16)
        second = torch.tensor([[4.0, 0.5, -1.5, 2.0]], dtype=torch.bfloat16)
        previous = torch.cat((first.float(), first.float() * 0.125), -1)
        attn._read_state = lambda *args: previous
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.compressor._linear_bf16_bf16_fp32",
            side_effect=lambda x, w: torch.nn.functional.linear(x.float(), w.float()),
        ):
            attn._produce_global(
                second,
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([1]),
                prefill=True,
            )
        actual, _ = attn._shared_attention["global"][2][0]
        values = torch.stack((first.float(), second.float()), 1)
        expected = compress_pairs(values, values * 0.125, attn.global_norm, attn.eps)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_candidates_use_block_max_and_pin_newest(self):
        logits = torch.tensor([[8.0, 9.0, 3.0, 4.0, 1.0, -torch.inf]])
        candidates = select_candidate_blocks(logits, torch.tensor([5]), 2, 2)
        self.assertEqual(set(candidates[0].tolist()), {0, 2})
        later_logits = torch.tensor([[1.0, 2.0, 100.0, 200.0, 0.0, -torch.inf]])
        mask_candidate_logits(later_logits, candidates, 2)
        self.assertEqual(later_logits.argmax(-1).item(), 1)
        self.assertTrue(torch.isneginf(later_logits[0, 2:4]).all())

    def test_cp4_assigns_each_ratio2_entry_exactly_once(self):
        positions = torch.arange(2048, dtype=torch.int64)
        request = torch.zeros_like(positions)
        # Physical owner blocks span eight kernel pages.
        bt = torch.arange(1, 17).view(1, -1)
        results = [
            cp_kv_slot_mapping(
                positions, bt, request, 128, 64, 2, 4, rank, owner_tokens_per_block=1024
            )
            for rank in range(4)
        ]
        owned = torch.stack([slots >= 0 for slots in results]).sum(0)
        self.assertTrue((owned[positions % 2 == 0] == 0).all())
        self.assertTrue((owned[positions % 2 == 1] == 1).all())
        self.assertTrue((results[0][1024:] == -1).all())
        self.assertTrue((results[1][:1024] == -1).all())
        self.assertTrue((results[2] == -1).all())
        self.assertTrue((results[3] == -1).all())


if __name__ == "__main__":
    unittest.main()
