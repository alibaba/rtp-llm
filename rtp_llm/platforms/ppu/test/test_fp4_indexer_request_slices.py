import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.platforms.ppu.models.dsv4.ppu_fp4_indexer import PpuFP4Indexer


class PpuFP4IndexerRequestSlicesTest(unittest.TestCase):
    def test_only_fp4_prepares_request_slices(self):
        for cls in (IndexerFP8, PpuFP4Indexer):
            with self.subTest(indexer=cls.__name__):
                indexer = cls.__new__(cls)
                torch.nn.Module.__init__(indexer)
                indexer.freqs_cis = torch.zeros((32, 1), dtype=torch.complex64)
                indexer.compress_ratio = 4
                indexer._kv_block_table = None
                indexer._kv_eb = 0

                # FP8 must not copy Q/K bounds to the host for unused slices.
                def cpu(tensor):
                    if cls is IndexerFP8:
                        raise AssertionError("unexpected host copy")
                    return tensor

                with patch.object(torch.Tensor, "cpu", cpu):
                    meta = indexer.prepare(
                        1,
                        5,
                        0,
                        torch.device("cpu"),
                        use_varlen=True,
                        batch_size=2,
                        cu_seqlens=torch.tensor([0, 2, 5]),
                        input_lengths=torch.tensor([2, 3]),
                        prefix_lengths=torch.tensor([14, 17]),
                        position_ids=torch.tensor([14, 15, 17, 18, 19]),
                        req_id_per_token=torch.tensor([0, 0, 1, 1, 1]),
                        has_prefix=True,
                    )
                self.assertEqual(
                    meta.request_score_slices,
                    ((0, 2, 0, 4), (2, 5, 4, 9)) if cls is PpuFP4Indexer else None,
                )

    def test_varlen_scores_each_request_against_local_k_slice(self):
        indexer = PpuFP4Indexer.__new__(PpuFP4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer._cp_ctx = None
        indexer._kv_pool_view = torch.empty((1, 1, 68), dtype=torch.uint8)
        indexer._kv_block_table = torch.ones((2, 1), dtype=torch.int32)
        indexer._kv_eb = 1
        indexer.n_heads = 2
        indexer.index_topk = 2
        indexer.weight_scale = 1.0
        indexer.weights_proj = torch.empty((2, 3))
        indexer.freqs_cis = torch.zeros((32, 1), dtype=torch.complex64)
        indexer._prefill_score_chunk_rows = 0
        indexer._propagate_pool_to_nested = Mock()
        indexer._clear_nested_pool = Mock()
        indexer._compute_indexer_q = Mock(
            return_value=torch.zeros((5, 2, 128), dtype=torch.bfloat16)
        )
        indexer.compressor = Mock()
        indexer.compressor._profile_label = ""

        meta = SimpleNamespace(
            M=5,
            T=9,
            sp_int=0,
            freqs_cis_slice=torch.zeros((5, 1), dtype=torch.complex64),
            compressor_meta=SimpleNamespace(positions=torch.arange(5)),
            block_table_i32=torch.ones((2, 1), dtype=torch.int32),
            cu_kv_seqlens=torch.tensor([0, 4, 9], dtype=torch.int32),
            ks=torch.tensor([0, 0, 4, 4, 4], dtype=torch.int32),
            ke=torch.tensor([1, 4, 5, 7, 9], dtype=torch.int32),
            request_score_slices=((0, 2, 0, 4), (2, 5, 4, 9)),
        )
        x = torch.zeros((5, 3), dtype=torch.bfloat16)
        qr = torch.zeros((5, 3), dtype=torch.bfloat16)
        gathered_k = torch.arange(9 * 64, dtype=torch.int8).reshape(9, 64)
        gathered_scale = torch.arange(9, dtype=torch.int32).reshape(9, 1)
        score_calls = []
        topk_calls = []

        def fake_score(q_pair, k_pair, weights, starts, ends, **kwargs):
            score_calls.append(
                {
                    "rows": q_pair[0].shape[0],
                    "width": k_pair[0].shape[0],
                    "starts": starts.clone(),
                    "ends": ends.clone(),
                }
            )
            return torch.zeros(
                (q_pair[0].shape[0], k_pair[0].shape[0]), dtype=torch.bfloat16
            )

        def fake_topk(logits, starts, ends, out):
            topk_calls.append((logits.shape, starts.clone(), ends.clone()))
            out.fill_(0)
            return out

        indexer._score = fake_score
        with patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_fp4_indexer.F.linear",
            return_value=torch.zeros((5, 2), dtype=torch.bfloat16),
        ), patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_fp4_indexer.quantize_q",
            return_value=(
                torch.zeros((5, 2, 128), dtype=torch.int8),
                torch.ones((5, 2), dtype=torch.int32),
                torch.zeros((5, 2), dtype=torch.bfloat16),
            ),
        ), patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_fp4_indexer.gather_k",
            return_value=(gathered_k, gathered_scale),
        ) as gather, patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_fp4_indexer.topk_bf16",
            side_effect=fake_topk,
        ):
            result = indexer.forward(x, qr, meta, workspace=object())

        gather.assert_called_once()
        self.assertEqual(result.shape, (5, 2))
        self.assertEqual([call["width"] for call in score_calls], [4, 5])
        self.assertEqual([call["rows"] for call in score_calls], [2, 3])
        self.assertTrue(torch.equal(score_calls[0]["starts"], torch.tensor([0, 0])))
        self.assertTrue(torch.equal(score_calls[0]["ends"], torch.tensor([1, 4])))
        self.assertTrue(torch.equal(score_calls[1]["starts"], torch.tensor([0, 0, 0])))
        self.assertTrue(torch.equal(score_calls[1]["ends"], torch.tensor([1, 3, 5])))
        self.assertEqual([shape for shape, _, _ in topk_calls], [(2, 4), (3, 5)])
        indexer._clear_nested_pool.assert_called_once()


if __name__ == "__main__":
    unittest.main()
