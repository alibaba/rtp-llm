"""Source-owned RoPE helper tests without loading model weights or RTP ops."""

import ast
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

SOURCE = Path(__file__).resolve().parents[1] / "msa_attention.py"


def methods():
    text = SOURCE.read_text()
    cls = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == "MSAAttention"
    )
    return text, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def model(cache, interleave=False):
    text, nodes = methods()
    namespace = {"torch": torch}
    for name in ("_apply_rope", "_apply_rope_contiguous"):
        exec(
            compile(
                "from __future__ import annotations\n"
                + ast.get_source_segment(text, nodes[name]),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
    obj = SimpleNamespace(
        cos_sin_cache=cache, _rope_interleave=interleave, _rope_theta=10000000.0
    )
    for name in ("_apply_rope", "_apply_rope_contiguous"):
        setattr(obj, name, MethodType(namespace[name], obj))
    return obj


class SourceContractTest(unittest.TestCase):
    def test_only_native_target_verify_uses_output_packing(self):
        text, nodes = methods()
        callers = [
            name
            for name, node in nodes.items()
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_apply_rope_contiguous"
                for n in ast.walk(node)
            )
        ]
        self.assertEqual(callers, ["_forward_target_verify"])
        node = nodes[callers[0]]
        guarded = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "self.nvfp4_kv_cache"
            and any(
                isinstance(x, ast.Call)
                and isinstance(x.func, ast.Attribute)
                and x.func.attr == "_apply_rope_contiguous"
                for statement in n.body
                for x in ast.walk(statement)
            )
        ]
        self.assertEqual(len(guarded), 1)
        self.assertIn(
            "self._apply_rope(q, k, positions)", ast.unparse(guarded[0].orelse)
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PackedRopeTest(unittest.TestCase):
    def test_exact_outputs_and_dynamic_graph(self):
        import flashinfer.rope

        torch.manual_seed(20260927)
        for rotary in (64, 128):
            freq = 10000000.0 ** (
                -torch.arange(0, rotary, 2, device="cuda", dtype=torch.float32) / rotary
            )
            angles = (
                torch.arange(81922, device="cuda", dtype=torch.float32)[:, None] * freq
            )
            cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
            for interleave in (False, True):
                instance = model(cache, interleave)
                for rows in (1, 5, 80, 96, 112, 128):
                    with self.subTest(rotary=rotary, interleave=interleave, rows=rows):
                        source = torch.randn(
                            rows, 9856, device="cuda", dtype=torch.bfloat16
                        )
                        carrier = source.clone()
                        q = carrier[:, :8192].reshape(rows, 64, 128)
                        k = carrier[:, 8192:8704].reshape(rows, 4, 128)
                        positions = torch.zeros(rows, device="cuda", dtype=torch.int32)
                        expected_q, expected_k = (
                            q.clone().contiguous(),
                            k.clone().contiguous(),
                        )
                        instance._apply_rope(expected_q, expected_k, positions)
                        actual = instance._apply_rope_contiguous(q, k, positions)
                        for a, b in zip(actual, (expected_q, expected_k)):
                            self.assertTrue(
                                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                            )
                        carrier.copy_(source)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = instance._apply_rope_contiguous(q, k, positions)
                        for scale, pos in (
                            (0, 0),
                            (1, 81920),
                            (1024, 127),
                            (0.001, 128),
                        ):
                            source.normal_().mul_(scale)
                            carrier.copy_(source)
                            positions.fill_(pos)
                            expected_q = q.clone().contiguous()
                            expected_k = k.clone().contiguous()
                            instance._apply_rope(expected_q, expected_k, positions)
                            graph.replay()
                            torch.cuda.synchronize()
                            for a, b in zip(actual, (expected_q, expected_k)):
                                self.assertTrue(a.is_contiguous())
                                self.assertTrue(
                                    torch.equal(
                                        a.view(torch.uint8), b.view(torch.uint8)
                                    )
                                )
                            if rows > 1:
                                self.assertTrue(torch.equal(carrier, source))
                            # Non-Q/K fields of the fused projection must never change.
                            self.assertTrue(
                                torch.equal(carrier[:, 8704:], source[:, 8704:])
                            )

    def test_no_cache_preserves_existing_path(self):
        instance = model(None)
        carrier = torch.randn(5, 9856, device="cuda", dtype=torch.bfloat16)
        q, k = carrier[:, :8192].reshape(5, 64, 128), carrier[:, 8192:8704].reshape(
            5, 4, 128
        )
        positions = torch.arange(5, device="cuda", dtype=torch.int32)
        expected = (q.contiguous(), k.contiguous())
        instance._apply_rope(*expected, positions)
        actual = instance._apply_rope_contiguous(q, k, positions)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    def test_graph_query_cast_and_persistent_cache_bytes(self):
        from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
            cache_layout,
            quantize_main_index_rows,
        )
        from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_query_cast import (
            fused_query_cast,
        )

        rows = 80
        angles = torch.randn(256, 32, device="cuda", dtype=torch.float32)
        instance = model(torch.cat((angles.cos(), angles.sin()), dim=-1))
        carrier = torch.randn(rows, 9856, device="cuda", dtype=torch.bfloat16)
        q = carrier[:, :8192].reshape(rows, 64, 128)
        k = carrier[:, 8192:8704].reshape(rows, 4, 128)
        v = carrier[:, 8704:9216].reshape(rows, 4, 128)
        idx = carrier[:, 9728:9856].reshape(rows, 1, 128).contiguous()
        idx_q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(rows, device="cuda", dtype=torch.int32)
        slots = torch.arange(rows, device="cuda", dtype=torch.int64) + 127
        outputs, graphs = [], []
        for candidate in (False, True):
            base = torch.full((6, 65536), 165, device="cuda", dtype=torch.uint8)
            side = torch.full((6, 17408), 165, device="cuda", dtype=torch.uint8)
            layout = cache_layout(base, side, 4, 128, 128)
            q8 = torch.empty(q.shape, device="cuda", dtype=torch.float8_e4m3fn)
            iq8 = torch.empty_like(idx_q, dtype=torch.float8_e4m3fn)

            def chain():
                if candidate:
                    qo, ko = instance._apply_rope_contiguous(q, k, positions)
                else:
                    qo, ko = q.contiguous(), k.contiguous()
                    instance._apply_rope(qo, ko, positions)
                fused_query_cast(qo, idx_q, q8, iq8)
                quantize_main_index_rows(
                    ko, v, idx, slots, layout, mma_scale_layout=True
                )

            chain()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain()
            graphs.append(graph)
            outputs.append((base, side, q8, iq8))
        for fake in (True, False, True, False):
            carrier.normal_()
            idx.copy_(carrier[:, 9728:9856].reshape_as(idx))
            positions.copy_(
                torch.randint(0, 256, positions.shape, device="cuda", dtype=torch.int32)
            )
            slots.copy_(torch.arange(rows, device="cuda") + 256)
            slots[::3] = -1
            if fake:
                slots.fill_(-1)
            for out in outputs:
                out[0].fill_(165)
                out[1].fill_(165)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            for a, b in zip(*outputs):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
