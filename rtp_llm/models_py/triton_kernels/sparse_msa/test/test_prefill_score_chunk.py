import os
import sys
import unittest
from unittest import mock

import torch


def _ensure_cutlass_dsl_on_path() -> None:
    try:
        import nvidia_cutlass_dsl
    except Exception:
        return
    package_dir = nvidia_cutlass_dsl.__path__[0]
    python_packages_dir = os.path.join(package_dir, "python_packages")
    if os.path.isdir(python_packages_dir) and python_packages_dir not in sys.path:
        sys.path.insert(0, python_packages_dir)


def _fmha_available() -> bool:
    try:
        _ensure_cutlass_dsl_on_path()
        import fmha_sm100.api  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillScoreChunkTest(unittest.TestCase):
    def setUp(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import score_chunk

        _ensure_cutlass_dsl_on_path()
        score_chunk.M3_PREFILL_WORKSPACE_CACHE.clear()
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("M3_MSA_INDEX_SCORE_CHUNK_ROWS", None)

    def tearDown(self) -> None:
        self.env.stop()

    def _inputs(self):
        torch.manual_seed(20260814)
        device = torch.device("cuda")
        block_size = 128
        query_lens = [130, 97]
        prefix_lens = [33, 205]
        seq_lens = [prefix + query for prefix, query in zip(prefix_lens, query_lens)]
        total_q = sum(query_lens)
        num_heads = 4
        head_dim = 64
        max_slots = 1024

        q = torch.randn(
            total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            max_slots, 1, head_dim, dtype=torch.bfloat16, device=device
        )
        req_to_token = torch.zeros(
            len(query_lens), max(seq_lens), dtype=torch.int32, device=device
        )
        req_to_token[0, : seq_lens[0]] = torch.arange(
            seq_lens[0], dtype=torch.int32, device=device
        )
        second_start = 512
        req_to_token[1, : seq_lens[1]] = torch.arange(
            second_start,
            second_start + seq_lens[1],
            dtype=torch.int32,
            device=device,
        )
        return dict(
            q=q,
            k_cache=k_cache,
            req_to_token=req_to_token,
            slot_ids=torch.arange(len(query_lens), dtype=torch.int64, device=device),
            cu_seqlens=torch.tensor(
                [0, query_lens[0], total_q], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            prefix_lens=torch.tensor(
                prefix_lens, dtype=torch.int32, device=device
            ),
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(seq_lens),
            block_size_k=block_size,
            topk=2,
            init_blocks=0,
            local_blocks=0,
        )

    def test_triton_fused_chunk_matches_full(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import score_chunk
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
            flash_prefill_with_fused_topk_index,
        )

        inputs = self._inputs()
        kwargs = dict(inputs)
        kwargs["idx_q"] = kwargs.pop("q")
        kwargs["idx_k_cache"] = kwargs.pop("k_cache")
        _, full_topk = flash_prefill_with_fused_topk_index(**kwargs)

        os.environ["M3_MSA_INDEX_SCORE_CHUNK_ROWS"] = "64"
        _, chunked_topk = flash_prefill_with_fused_topk_index(**kwargs)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(chunked_topk, full_topk))
        workspace = score_chunk.M3_PREFILL_WORKSPACE_CACHE[inputs["q"].device]
        full_score_bytes = (
            inputs["q"].shape[0]
            * inputs["q"].shape[1]
            * ((inputs["max_seqlen_k"] + inputs["block_size_k"] - 1) // inputs["block_size_k"])
            * 4
        )
        self.assertLess(workspace.numel(), full_score_bytes)

    def test_legacy_chunk_matches_full(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.flash_with_topk_idx import (
            flash_prefill_with_topk_index,
        )

        inputs = self._inputs()
        kwargs = dict(inputs)
        kwargs.update(
            v_cache=None,
            sink=None,
            block_size_q=1,
            disable_index_value=True,
        )
        _, full_topk = flash_prefill_with_topk_index(**kwargs)

        os.environ["M3_MSA_INDEX_SCORE_CHUNK_ROWS"] = "64"
        _, chunked_topk = flash_prefill_with_topk_index(**kwargs)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(chunked_topk, full_topk))

    @unittest.skipUnless(_fmha_available(), "fmha_sm100 required")
    def test_fmha_chunk_matches_full(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
            build_index_score_plan,
            flash_prefill_topk_to_block_tables,
        )

        inputs = self._inputs()
        q = inputs["q"]
        full_plan = build_index_score_plan(
            inputs["cu_seqlens"],
            inputs["seq_lens"],
            inputs["prefix_lens"],
            q.shape[1],
            1,
            inputs["block_size_k"],
        )
        kwargs = dict(
            idx_q=q,
            idx_k_cache=inputs["k_cache"],
            req_to_token=inputs["req_to_token"],
            cu_seqlens=inputs["cu_seqlens"],
            seq_lens=inputs["seq_lens"],
            prefix_lens=inputs["prefix_lens"],
            max_seqlen_q=inputs["max_seqlen_q"],
            max_seqlen_k=inputs["max_seqlen_k"],
            block_size_k=inputs["block_size_k"],
            topk=inputs["topk"],
            num_pages=(inputs["max_seqlen_k"] + inputs["block_size_k"] - 1)
            // inputs["block_size_k"],
            init_blocks=0,
            local_blocks=0,
            index_score_plan=full_plan,
            emit_block_table=False,
        )
        _, _, full_topk = flash_prefill_topk_to_block_tables(**kwargs)
        kwargs["emit_block_table"] = True
        full_bt, full_lens, full_topk_with_bt = flash_prefill_topk_to_block_tables(
            **kwargs
        )
        self.assertTrue(torch.equal(full_topk_with_bt, full_topk))

        os.environ["M3_MSA_INDEX_SCORE_CHUNK_ROWS"] = "64"
        kwargs["index_score_plan"] = {}
        kwargs["emit_block_table"] = False
        _, _, chunked_topk = flash_prefill_topk_to_block_tables(**kwargs)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(chunked_topk, full_topk))

        kwargs["emit_block_table"] = True
        chunked_bt, chunked_lens, chunked_topk_with_bt = (
            flash_prefill_topk_to_block_tables(**kwargs)
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(chunked_topk_with_bt, full_topk))
        self.assertTrue(torch.equal(chunked_bt, full_bt))
        self.assertTrue(torch.equal(chunked_lens, full_lens))


if __name__ == "__main__":
    unittest.main()
