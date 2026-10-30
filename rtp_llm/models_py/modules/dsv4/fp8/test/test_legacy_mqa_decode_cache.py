"""Execute decode cache packing on CPU against the native block-plane ABI."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]


def _function(path, name):
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace, namespace[name]


class LegacyMqaDecodeCacheTest(unittest.TestCase):
    def _check_cache(self, model, page, scale_fmt):
        base = ROOT / "models_py/modules"
        namespace, forward = _function(
            base / model / "decode/indexer_decode_op.py", "_forward_fast"
        )
        _, unpack = _function(base / "base/cuda/indexer_op.py", "_unpack_ue8m0_scale")
        _, metadata = _function(
            base / "dsv4/fp8/_indexer_score.py", "get_fp8_paged_mqa_logits_metadata"
        )
        batch, queries, heads, dim = 2, 2, 32, 128
        tokens = page + 6
        padded = page * 2
        quantized = []

        def quantize(values, **kwargs):
            if kwargs["scale_ue8m0"]:
                self.assertTrue(kwargs["column_major_scales"])
                self.assertTrue(kwargs["scale_tma_aligned"])
            payload = (
                (torch.arange(values.numel()).reshape(values.shape) % 15 - 7)
                .float()
                .to(torch.float8_e4m3fn)
            )
            exponents = torch.arange(values.shape[0], dtype=torch.int32) % 5 + 125
            scales = (
                exponents.view(-1, 1)
                if kwargs["scale_ue8m0"]
                else torch.arange(values.shape[0], dtype=torch.float32).view(-1, 1)
                + 1.25
            )
            quantized.append((payload, scales))
            return payload, scales

        def paged_logits(q, cache, weights, lengths, table, schedule, width, **kwargs):
            self.assertEqual(schedule, "legacy-schedule")
            self.assertEqual(cache.shape, (batch * 2, page, 1, dim + 4))
            self.assertEqual(lengths.shape, (batch, queries))
            self.assertTrue(
                torch.equal(
                    lengths,
                    torch.tensor(
                        [[tokens, tokens], [tokens - 3, tokens - 3]], dtype=torch.int32
                    ),
                )
            )
            self.assertEqual(table.tolist(), [[0, 1], [2, 3]])
            key, scales = quantized[1]
            scales = unpack(scales) if scale_fmt == "ue8m0" else scales
            raw = cache.view(-1)
            for block in range(batch * 2):
                # Read the byte offsets used by the native fused-cache ABI.
                start = block * page * (dim + 4)
                payload_end = start + page * dim
                scale_end = start + page * (dim + 4)
                self.assertTrue(
                    torch.equal(
                        raw[start:payload_end],
                        key[block * page : (block + 1) * page]
                        .contiguous()
                        .view(torch.uint8)
                        .reshape(-1),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        raw[payload_end:scale_end].view(torch.float32),
                        scales[block * page : (block + 1) * page].reshape(-1),
                    )
                )
            return torch.arange(batch * queries * width, dtype=torch.float32).view(
                batch * queries, width
            )

        backend = SimpleNamespace(
            get_num_sms=lambda: 148,
            get_legacy_paged_mqa_logits_metadata=lambda *args: "legacy-schedule",
            fp8_paged_mqa_logits=paged_logits,
        )
        namespace.update(
            deep_gemm=backend,
            get_fp8_paged_mqa_logits_metadata=metadata,
            select_indexer_topk=lambda scores, k, **kwargs: scores.topk(k).indices,
        )
        modules = {
            "rtp_llm.models_py.kernels.cuda.fp8_kernel": SimpleNamespace(
                sgl_per_token_group_quant_fp8=quantize
            ),
            "rtp_llm.models_py.modules.base.cuda.indexer_op": SimpleNamespace(
                _unpack_ue8m0_scale=unpack
            ),
        }
        op = SimpleNamespace(
            page_block_size=page,
            block_size=128,
            scale_fmt=scale_fmt,
            softmax_scale=0.5,
            index_topk=4,
        )
        out = torch.empty((batch, queries, 4), dtype=torch.int32)
        with patch.dict(sys.modules, modules):
            actual = forward(
                op,
                torch.ones((batch, queries, heads, dim), dtype=torch.bfloat16),
                torch.ones((batch, tokens, dim), dtype=torch.bfloat16),
                torch.ones((batch, queries, heads), dtype=torch.bfloat16),
                torch.tensor([tokens, tokens - 3], dtype=torch.int32),
                out,
            )
        self.assertIs(actual, out)
        self.assertEqual(out[0, 0].tolist(), list(range(padded - 1, padded - 5, -1)))

    def test_both_decoders_pack_native_payload_and_fp32_scale_planes(self):
        for model in ("dsv4", "dsv41"):
            for page in (64, 128):
                for scale_fmt in ("fp32", "ue8m0"):
                    with self.subTest(model=model, page=page, scale_fmt=scale_fmt):
                        self._check_cache(model, page, scale_fmt)


if __name__ == "__main__":
    unittest.main()
