"""DSpARK proposal scheduling follows the same backend as sparse decode."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode import (
    decode_attn_metadata,
    fp8_sparse_attn_decode_op,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class _DSparkTestCase(unittest.TestCase):
    def run_attention(self, op, metadata, device, gamma=3):
        path = (
            Path(__file__).resolve().parents[4]
            / "model_desc/deepseek_v4_dspark_model.py"
        )
        node = next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == "_forward_dspark_attention"
        )
        body = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ]
        query = Mock(device=device)
        qkv = types.SimpleNamespace(q=query, kv=object(), freqs_cis=object())
        writer, projector = Mock(), Mock(return_value=object())
        env = {
            "torch": torch,
            "decode_compute_qkv": Mock(return_value=qkv),
            "decode_write_swa_fp8": writer,
            "decode_output_proj": projector,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                str(path),
                "exec",
            ),
            env,
        )
        attention = types.SimpleNamespace(
            head_dim=512,
            n_heads=8,
            attn_sink=object(),
            _get_fp8_decode_op=Mock(return_value=op),
        )
        pool, batch, topk = object(), 2, 128
        lengths = torch.full((batch * gamma,), 5, dtype=torch.int32)
        indices = torch.arange(batch * gamma * topk, dtype=torch.int32).view(
            batch * gamma, topk
        )
        owner = types.SimpleNamespace(
            _swa_cache_bound=Mock(return_value=nullcontext((attention, 64, pool))),
            _global_pool_slots=Mock(return_value=torch.arange(batch * gamma)),
            _build_noncausal_indices=Mock(return_value=(indices, lengths)),
        )
        result = env[node.name](
            owner,
            0,
            torch.zeros(batch, gamma, 4),
            torch.zeros(batch, gamma, dtype=torch.long),
            torch.ones(batch, dtype=torch.int32),
            torch.ones(batch, dtype=torch.bool),
            torch.zeros(batch, 1, dtype=torch.int32),
            64,
            metadata,
        )
        attention._get_fp8_decode_op.assert_called_once_with()
        self.assertIs(result, projector.return_value)
        self.assertEqual(writer.call_args.kwargs["q_len"], gamma)
        return query, pool, lengths, indices


class DSparkSparseBackendTest(_DSparkTestCase):
    def test_default_preserves_per_step_flashmla_metadata_and_query_geometry(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 0.5)
        planner = Mock(side_effect=lambda **kw: (object(), None))
        wheel = types.SimpleNamespace(get_mla_metadata=planner)
        with patch.object(
            fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
        ), patch.object(
            op, "_forward_flash_mla", return_value=object()
        ) as kernel, patch.dict(
            sys.modules, {"rtp_llm.models_py.utils.flash_mla_legacy": wheel}
        ):
            previous = None
            for _ in range(2):
                step = types.SimpleNamespace(sched_meta_cache={})
                query, pool, lengths, indices = self.run_attention(
                    op, step, torch.device("cpu")
                )
                passed = kernel.call_args.kwargs
                self.assertIs(passed["q"], query)
                self.assertIs(passed["kv_cache"], pool)
                self.assertIs(passed["topk_length"], lengths)
                torch.testing.assert_close(passed["topk_idxs"], indices.view(2, 3, 128))
                self.assertIs(passed["sched_meta"], step.sched_meta_cache[(2, None)])
                self.assertIsNot(passed["sched_meta"], previous)
                previous = passed["sched_meta"]
        self.assertEqual(planner.call_count, 2)
        planner.assert_called_with(
            cache_seqlens=None,
            num_q_tokens_per_head_k=2 * 3 * 8,
            topk=128,
            num_heads_q=8,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )

    def test_provider_without_flashmla_scheduler_never_builds_flashmla_metadata(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 0.5)
        kernel = Mock(return_value=object())
        with patch.object(
            op, "_prepare_backend", return_value=(kernel, False)
        ), patch.object(
            decode_attn_metadata,
            "get_or_build_sched_meta",
            side_effect=AssertionError("unexpected FlashMLA planner"),
        ), patch.dict(
            sys.modules, {"rtp_llm.models_py.utils.flash_mla_legacy": None}
        ):
            self.run_attention(op, object(), torch.device("cpu"), gamma=5)
        self.assertIsNone(kernel.call_args.kwargs["sched_meta"])


if __name__ == "__main__":
    unittest.main()
