"""SM90 fused query operands preserve values, planner ownership and replay."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]
COPY_NAME = "rtp_llm.models_py.modules.dsv41._flash_mla_query_copy_triton"
COPY = _load("flash_query_copy_test_kernel", ROOT / "_flash_mla_query_copy_triton.py")
FLASH = _load("flash_query_copy_test_wrapper", ROOT / "flash_mla.py")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FlashMLAQueryCopyCudaTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")
        self.modules = patch.dict(sys.modules, {COPY_NAME: COPY})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def _inputs(self, batch, queries, extra=True):
        q = torch.randn(batch, queries, 8, 32, device="cuda", dtype=torch.bfloat16)
        indices = torch.randint(0, 512, (batch, queries, 37), device="cuda", dtype=torch.int32)
        extra_indices = (
            torch.randint(0, 128, (batch, queries, 13), device="cuda", dtype=torch.int32)
            if extra else None
        )
        return q, indices, extra_indices

    def _run(self, q, indices, extra, owner):
        # Consume every copied operand without retaining aliases to scratch.
        def native(*, q, attn_sink, **kwargs):
            value = q + kwargs["indices"].sum(-1)[..., None, None].to(q.dtype)
            if kwargs.get("extra_indices_in_kvcache") is not None:
                value = value + kwargs["extra_indices_in_kvcache"].sum(-1)[..., None, None].to(q.dtype)
            return value, value[..., 0].transpose(1, 2).float().contiguous()

        return FLASH._single_query_decode(
            native, q, None,
            dict(tile_scheduler_metadata=owner, indices=indices,
                 extra_indices_in_kvcache=extra),
            lambda: (object(), None),
        )

    def test_exact_operands_and_arbitrary_query_counts(self):
        for batch, queries in ((1, 1), (1, 6), (2, 2), (3, 3), (4, 4), (4, 6), (3, 7), (64, 6), (128, 3)):
            for has_extra in (False, True):
                with self.subTest(batch=batch, queries=queries, extra=has_extra):
                    q, indices, extra = self._inputs(batch, queries, has_extra)
                    owner = SimpleNamespace()
                    # Native contiguous slice baseline, without a product env switch.
                    with patch.object(COPY, "can_fuse_query_copy", return_value=False), patch.object(COPY, "join_query_outputs", return_value=None):
                        expected = self._run(q, indices, extra, SimpleNamespace())
                    with patch.object(COPY, "copy_query_into_scratch", wraps=COPY.copy_query_into_scratch) as fused_copy:
                        actual = self._run(q, indices, extra, owner)
                    self.assertEqual(fused_copy.call_count, queries if batch > 1 and queries > 1 else 0)
                    for left, right in zip(expected, actual):
                        self.assertTrue(torch.equal(left, right))
                    if batch > 1 and queries > 1:
                        scratch = COPY.allocate_query_scratch(q, indices, extra)
                        addresses = [v.data_ptr() for v in scratch if v is not None]
                        for query in range(queries):
                            COPY.copy_query_into_scratch(q, indices, extra, query, scratch)
                            for source, copied in zip((q, indices, extra), scratch):
                                if source is not None:
                                    self.assertTrue(torch.equal(source[:, query:query + 1], copied))
                            self.assertEqual(addresses, [v.data_ptr() for v in scratch if v is not None])

    def test_graph_replay_and_planner_reuse(self):
        for queries in (2, 4, 6, 7):
            with self.subTest(queries=queries):
                q, indices, extra = self._inputs(3, queries)
                owner = SimpleNamespace()
                self._run(q, indices, extra, owner)
                plan = owner._dsv41_query_plan
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = self._run(q, indices, extra, owner)
                self.assertIs(owner._dsv41_query_plan, plan)
                for _ in range(2):
                    q.normal_()
                    indices.random_(0, 512)
                    extra.random_(0, 128)
                    with patch.object(COPY, "can_fuse_query_copy", return_value=False), patch.object(COPY, "join_query_outputs", return_value=None):
                        expected = self._run(q, indices, extra, SimpleNamespace())
                    graph.replay()
                    torch.cuda.synchronize()
                    for left, right in zip(expected, actual):
                        self.assertTrue(torch.equal(left, right))

    def test_unsupported_layouts_keep_native_fallback(self):
        q, indices, extra = self._inputs(3, 6)
        self.assertTrue(COPY.can_fuse_query_copy(q, indices, extra))
        self.assertFalse(COPY.can_fuse_query_copy(q.float(), indices, extra))
        self.assertFalse(COPY.can_fuse_query_copy(q, indices.long(), extra))
        self.assertFalse(COPY.can_fuse_query_copy(q[..., ::2], indices, extra))
        self.assertFalse(COPY.can_fuse_query_copy(q, indices[..., ::2], extra))
        self.assertFalse(COPY.can_fuse_query_copy(q, indices, extra[..., :0]))
        with patch.object(COPY, "copy_query_into_scratch", side_effect=AssertionError("unexpected fused copy")):
            actual = self._run(q.float(), indices.long(), extra.long(), SimpleNamespace())
        with patch.object(COPY, "can_fuse_query_copy", return_value=False):
            expected = self._run(q.float(), indices.long(), extra.long(), SimpleNamespace())
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left, right))

    def test_output_join_is_bitwise_and_has_layout_fallback(self):
        for batch, queries in ((1, 6), (3, 2), (4, 7), (64, 3), (128, 6)):
            with self.subTest(batch=batch, queries=queries):
                outputs = [torch.randn(batch, 1, 8, 32, device="cuda", dtype=torch.bfloat16) for _ in range(queries)]
                lses = [torch.randn(batch, 8, 1, device="cuda") for _ in range(queries)]
                expected = torch.cat(outputs, dim=1), torch.cat(lses, dim=2)
                actual = COPY.join_query_outputs(outputs, lses)
                self.assertIsNotNone(actual)
                for left, right in zip(expected, actual):
                    self.assertTrue(torch.equal(left, right))
                    self.assertEqual(left.stride(), right.stride())
                self.assertIsNone(COPY.join_query_outputs([v[..., ::2] for v in outputs], lses))
                self.assertIsNone(COPY.join_query_outputs(outputs, [v.double() for v in lses]))


if __name__ == "__main__":
    unittest.main()
