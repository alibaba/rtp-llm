"""CPU regression for paged Indexer's unused cumulative-query allocation."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

SOURCE = Path(__file__).parents[1] / "indexer_op.py"


def paged_method():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IndexerOp"
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_topk_paged"
    )


class IndexerPagedUnusedMetadataTest(unittest.TestCase):
    def test_no_unused_cumulative_query_tensor_or_arange(self):
        method = paged_method()
        self.assertFalse(
            any(
                isinstance(node, ast.Name) and node.id == "cu_seqlens_q"
                for node in ast.walk(method)
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "arange"
                for node in ast.walk(method)
            )
        )

    def test_actual_method_keeps_decode_target_and_draft_contract(self):
        initialized_before = torch.cuda.is_initialized()
        cases = [
            (1, False, False),
            (4, True, False),
            (6, True, False),
            (4, False, True),
            (6, False, True),
        ]
        for queries, target_verify, draft_extend in cases:
            for cached_schedule in (False, True):
                with self.subTest(
                    queries=queries,
                    target_verify=target_verify,
                    draft_extend=draft_extend,
                    cached_schedule=cached_schedule,
                ):
                    rows = 2 * queries
                    table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
                    lengths = torch.arange(rows, dtype=torch.int32) + 100
                    observed = {}

                    def logits(
                        q,
                        cache,
                        weights,
                        context_lens,
                        block_table,
                        metadata,
                        max_seq_len,
                        clean_logits,
                    ):
                        observed.update(
                            q=q,
                            context_lens=context_lens,
                            table=block_table,
                            metadata=metadata,
                            max_seq_len=max_seq_len,
                        )
                        self.assertFalse(clean_logits)
                        return torch.zeros((rows, max_seq_len))

                    def topk(scores, causal_lengths, output, workspace, k, max_seq_len):
                        self.assertIs(causal_lengths, lengths)
                        self.assertEqual((k, max_seq_len), (2048, 1024))
                        output.fill_(17)

                    metadata = torch.ones(1, dtype=torch.int32)

                    def schedule(context_lens, page_size, num_sms):
                        self.assertFalse(cached_schedule)
                        self.assertTrue(torch.equal(context_lens, lengths[:, None]))
                        self.assertEqual((page_size, num_sms), (64, 148))
                        observed["generated_schedule"] = True
                        return metadata

                    namespace = {
                        "torch": torch,
                        "Any": Any,
                        "KVCache": object,
                        "deep_gemm": SimpleNamespace(
                            fp8_paged_mqa_logits=logits,
                            get_num_sms=lambda: 148,
                            get_paged_mqa_logits_metadata=schedule,
                        ),
                        "_physical_block_table": lambda inputs: table,
                        "_get_topk_workspace": lambda device: None,
                    }
                    exec(
                        compile(
                            ast.Module(body=[paged_method()], type_ignores=[]),
                            str(SOURCE),
                            "exec",
                        ),
                        namespace,
                    )
                    owner = SimpleNamespace(
                        index_n_heads=32,
                        index_topk=2048,
                        blocksize=64,
                        _head_dim_with_sf=lambda: 132,
                        _kv_cache_blocks=lambda cache: torch.zeros(
                            (3, 64, 132), dtype=torch.uint8
                        ),
                        _paged_topk_op=topk,
                    )
                    params = SimpleNamespace(
                        expanded_seq_lens=lengths,
                        kvlen_d=lengths,
                        schedule_metadata=metadata if cached_schedule else None,
                    )
                    # The paged API consumes context lengths, not cumulative Q offsets.
                    inputs = SimpleNamespace(
                        is_target_verify=target_verify, is_draft_extend=draft_extend
                    )
                    result = namespace["_get_topk_paged"](
                        owner,
                        torch.zeros((rows, 32, 128)),
                        torch.ones((rows, 32)),
                        None,
                        params,
                        inputs,
                    )
                    self.assertEqual(result.shape, (rows, 2048))
                    self.assertTrue(bool((result == 17).all()))
                    self.assertTrue(
                        torch.equal(
                            observed["table"], table.repeat_interleave(queries, dim=0)
                        )
                    )
                    self.assertTrue(
                        torch.equal(observed["context_lens"], lengths[:, None])
                    )
                    self.assertEqual(observed["q"].shape, (rows, 1, 32, 128))
                    self.assertIs(observed["metadata"], metadata)
                    self.assertEqual(
                        observed.get("generated_schedule", False), not cached_schedule
                    )
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)


if __name__ == "__main__":
    unittest.main()
