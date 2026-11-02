import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.target_head_graph import (
    TargetHeadGraph,
)
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear


class TargetHeadGraphTest(unittest.TestCase):
    def test_dynamic_exact_target_rows_and_alternating_request_keys(self):
        torch.manual_seed(20261019)
        head = Fp8Linear(
            (torch.randn(4096, 2560, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.tensor([0.019713], device="cuda"),
            role="DECODE",
        )
        cache = TargetHeadGraph(True)
        graphs, inputs = {}, {}
        for batch in (1, 3, 4, 5, 6, 7, 8):
            x = torch.randn(batch * 4, 2560, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                cache.stage(head, x, batch)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cache.stage(head, x, batch)
            inputs[batch], graphs[batch] = x, graph
        pointers = {b: value[0].data_ptr() for b, value in cache.outputs.items()}
        for batch in (8, 1, 4, 3, 7, 5, 6, 1, 8):
            x = inputs[batch]
            for scale in (0.0, 0.01, 1.0, 100.0):
                x.normal_().mul_(scale)
                graphs[batch].replay()
                cache.activate(batch)
                actual = cache.take(x.clone())
                expected = head(x)
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                self.assertEqual(actual.data_ptr(), pointers[batch])
                self.assertIsNone(cache.take(x))
        for graph in graphs.values():
            graph.reset()

    def test_selected_rows_must_requantize_and_consume_activation(self):
        cache = TargetHeadGraph(True)
        hidden = torch.ones(8, 32, device="cuda", dtype=torch.bfloat16)
        hidden[4:] *= 100
        cache.stage(lambda x: x * 2, hidden, 2)
        for changed in (hidden[:4], hidden.float(), hidden.cpu(), hidden[:, :16]):
            cache.activate(2)
            self.assertIsNone(cache.take(changed))
            self.assertIsNone(cache.take(hidden))
        cache.activate(2)
        cache.clear_activation()
        self.assertIsNone(cache.take(hidden))
        cache.activate(7)
        self.assertIsNone(cache.take(hidden))

    def test_model_head_consumes_local_bf16_once(self):
        hidden = torch.ones(4, 32, device="cuda", dtype=torch.bfloat16)
        cache = TargetHeadGraph(True)
        cache.stage(lambda x: x * 2, hidden, 1)
        activated = []
        coordinator = SimpleNamespace(
            activate_speculative_target_graph_replay=activated.append
        )
        model = SimpleNamespace(
            _target_head_graph=cache,
            _fp8_lm_head=lambda x: x * 3,
            _ple_coordinator=lambda: coordinator,
        )
        method = lambda rows: Qwen4ExpModel._compute_local_logits(model, rows)
        Qwen4ExpModel.on_cuda_graph_replay(
            model,
            1,
            SimpleNamespace(
                attention_inputs={"target": SimpleNamespace(is_target_verify=True)}
            ),
        )
        self.assertEqual(activated, [1])
        actual = method(hidden)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, hidden * 2))
        self.assertTrue(torch.equal(method(hidden), hidden * 3))
        cache.activate(1)
        self.assertTrue(torch.equal(method(hidden[:1]), hidden[:1] * 3))

    def test_disabled_and_malformed_inputs_preserve_fallback(self):
        def fail(_):
            raise AssertionError("disabled head evaluated")

        cache = TargetHeadGraph(False)
        hidden = torch.zeros(4, 32, device="cuda", dtype=torch.bfloat16)
        cache.stage(fail, hidden, 1)
        cache.activate(1)
        self.assertIsNone(cache.take(hidden))
        self.assertEqual(cache.outputs, {})
        cache = TargetHeadGraph(True)
        for count in (0, -1, 3, True):
            with self.assertRaisesRegex(RuntimeError, "exact packed"):
                cache.stage(lambda x: x, hidden, count)
        for invalid in (hidden.cpu(), hidden.float(), hidden[:0], hidden.unsqueeze(0)):
            with self.assertRaisesRegex(RuntimeError, "exact packed"):
                cache.stage(lambda x: x, invalid, 1)
        for invalid in (hidden.cpu(), hidden.float(), hidden[:1], hidden.unsqueeze(0)):
            with self.assertRaisesRegex(RuntimeError, "local CUDA BF16"):
                cache.stage(lambda x: invalid, hidden, 1)
        self.assertEqual(cache.outputs, {})


if __name__ == "__main__":
    unittest.main()
