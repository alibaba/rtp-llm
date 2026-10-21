import unittest

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_head_graph import (
    DraftHeadGraph,
)
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear


class DraftHeadGraphTest(unittest.TestCase):
    def test_dynamic_graph_and_alternating_batch_keys_consume_once(self):
        torch.manual_seed(20261019)
        head = Fp8Linear(
            (torch.randn(4096, 2560, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.tensor([0.019713], device="cuda"),
            role="DECODE",
        )
        cache = DraftHeadGraph(True)
        graphs, inputs = {}, {}
        for batch in (1, 3, 4, 7, 8):
            x = torch.randn(batch, 2560, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                cache.stage(head, x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cache.stage(head, x)
            inputs[batch], graphs[batch] = x, graph
        pointers = {batch: out.data_ptr() for batch, out in cache.outputs.items()}
        for batch in (8, 1, 4, 3, 7, 1, 8):
            x = inputs[batch]
            for scale in (0.0, 0.01, 1.0, 100.0):
                x.normal_().mul_(scale)
                graphs[batch].replay()
                cache.activate(batch)
                actual = cache.take(x)
                expected = head(x)
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                self.assertEqual(actual.data_ptr(), pointers[batch])
                self.assertIsNone(cache.take(x))
        for graph in graphs.values():
            graph.reset()

    def test_disabled_mode_does_not_evaluate_head(self):
        def fail(_):
            raise AssertionError("disabled cache invoked head")

        cache = DraftHeadGraph(False)
        x = torch.zeros(1, 2560)
        cache.stage(fail, x)
        cache.activate(8)
        self.assertIsNone(cache.take(x))
        self.assertEqual(cache.outputs, {})

    def test_stale_activation_shape_and_missing_key_fail_closed(self):
        cache = DraftHeadGraph(True)
        hidden = torch.zeros(4, 2560, device="cuda", dtype=torch.bfloat16)
        cache.stage(lambda x: x, hidden)
        cache.activate(4)
        cache.clear_activation()
        self.assertIsNone(cache.take(hidden))
        cache.activate(4)
        with self.assertRaisesRegex(RuntimeError, "no captured"):
            cache.activate(3)
        self.assertIsNone(cache.take(hidden))
        for wrong in (
            hidden.cpu(),
            hidden.unsqueeze(0),
            hidden[:0],
            hidden.repeat(2, 1),
        ):
            cache.activate(4)
            with self.assertRaisesRegex(RuntimeError, "rows do not cover"):
                cache.take(wrong)
            self.assertIsNone(cache.take(hidden))

    def test_padded_capture_returns_only_live_rows(self):
        cache = DraftHeadGraph(True)
        hidden = torch.arange(4 * 32, device="cuda").reshape(4, 32).to(torch.bfloat16)
        cache.stage(lambda x: x * 2, hidden)
        cache.activate(4)
        actual = cache.take(hidden[:3])
        self.assertTrue(torch.equal(actual, (hidden * 2)[:3]))
        self.assertIsNone(cache.take(hidden))

    def test_invalid_head_output_is_rejected(self):
        hidden = torch.zeros(4, 32, device="cuda", dtype=torch.bfloat16)
        for bad in (hidden.cpu(), hidden.float(), hidden[:1], hidden.unsqueeze(0)):
            cache = DraftHeadGraph(True)
            with self.assertRaisesRegex(RuntimeError, "local CUDA BF16 logits"):
                cache.stage(lambda x: bad, hidden)
            self.assertEqual(cache.outputs, {})


if __name__ == "__main__":
    unittest.main()
