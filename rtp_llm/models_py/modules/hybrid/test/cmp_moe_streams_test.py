"""CMP production dispatch and CUDA Graph fork/join tests (not model precision)."""
import unittest
import os
from unittest.mock import patch
import torch
from rtp_llm.models_py.modules.hybrid.cmp_moe_streams import CmpMoeStreams
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer

class CmpMoeStreamsTest(unittest.TestCase):
    def test_default_and_explicit_switches(self):
        key = "RTP_LLM_GLM5_CMP_MOE_DUAL_STREAM"
        for value, enabled in ((None, True), ("1", True), ("0", False)):
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
                with patch("torch.cuda.Stream", side_effect=AssertionError("eager stream allocation")):
                    pair = CmpMoeStreams.from_env()
                self.assertEqual(pair is not None, enabled)
                if pair is not None:
                    self.assertIsNone(pair._stream)

    def test_disabled_dispatch_is_original_serial_path(self):
        with patch.dict(os.environ, {"RTP_LLM_GLM5_CMP_MOE_DUAL_STREAM": "0"}), torch.no_grad():
            layer, streams = self.make_layer()
            layer._cmp_moe_streams = CmpMoeStreams.from_env()
            x = torch.ones((4, 6144), device="cuda", dtype=torch.bfloat16)
            with patch.object(CmpMoeStreams, "run", side_effect=AssertionError("disabled helper invoked")):
                result = layer.forward_prepacked(x, None, None)
            self.assertEqual(streams["shared"], streams["routed"])
            torch.testing.assert_close(result, torch.full_like(x, 5), rtol=0, atol=0)

    def make_layer(self, enabled=True):
        streams = {}
        class Routed(torch.nn.Module):
            def forward_prepacked(self, x):
                streams["routed"] = torch.cuda.current_stream().cuda_stream
                return x * 2
        class Shared(torch.nn.Module):
            def forward(self, x):
                streams["shared"] = torch.cuda.current_stream().cuda_stream
                return x * 3
        layer = GenericMoeLayer.__new__(GenericMoeLayer)
        torch.nn.Module.__init__(layer)
        layer.fused_moe, layer.shared_expert = Routed(), Shared()
        layer.fake_balance_expert = None
        layer._use_mega_moe_fused_shared = False
        layer._cmp_moe_streams = CmpMoeStreams() if enabled else None
        return layer, streams

    def test_production_cmp_dispatch_and_graph_replay(self):
        for enabled in (False, True):
            for rows in (1, 4, 16, 32, 33):
                with self.subTest(enabled=enabled, rows=rows), torch.no_grad():
                    layer, streams = self.make_layer(enabled)
                    x = torch.ones((rows,6144), device="cuda", dtype=torch.bfloat16)
                    ids = torch.zeros((rows,8), device="cuda", dtype=torch.int32)
                    weights = torch.ones((rows,8), device="cuda") / 8
                    caller = torch.cuda.Stream()
                    caller.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(caller):
                        for _ in range(3):
                            out = layer.forward_prepacked(x,ids,weights)
                        self.assertEqual(streams["shared"] != streams["routed"], enabled and rows <= 32)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=caller):
                            out = layer.forward_prepacked(x,ids,weights)
                        for value in (0, -2, 7):
                            x.fill_(value)
                            g.replay()
                            caller.synchronize()
                            torch.testing.assert_close(out, torch.full_like(x,value*5), rtol=0, atol=0)

    def test_empty_dtype_cpu_and_grad_fallback(self):
        for rows, dtype, device, grad in ((0, torch.bfloat16, "cuda", False),
                                          (4, torch.float32, "cuda", False),
                                          (4, torch.bfloat16, "cpu", False),
                                          (4, torch.bfloat16, "cuda", True)):
            with self.subTest(rows=rows, dtype=dtype, device=device, grad=grad), torch.set_grad_enabled(grad):
                layer, streams = self.make_layer()
                x = torch.ones((rows, 6144), device=device, dtype=dtype)
                result = layer.forward_prepacked(x, None, None)
                self.assertIsNone(layer._cmp_moe_streams._stream)
                self.assertEqual(streams["shared"], streams["routed"])
                torch.testing.assert_close(result, x * 5, rtol=0, atol=0)

    def test_mixed_input_devices_stays_serial(self):
        with torch.no_grad():
            pair = CmpMoeStreams()
            x = torch.ones(8, device="cuda")
            streams = []
            def branch():
                streams.append(torch.cuda.current_stream().cuda_stream)
                return x + 1
            pair.run(branch, branch, (x, torch.ones(8)))
            self.assertIsNone(pair._stream)
            self.assertEqual(streams[0], streams[1])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_warmed_helper_on_another_device_stays_serial(self):
        with torch.no_grad(), torch.cuda.device(0):
            pair = CmpMoeStreams()
            x = torch.ones(8, device="cuda:0")
            pair.run(lambda: x + 1, lambda: x + 2, (x,))
            original_stream = pair._stream
            y = torch.ones(8, device="cuda:1")
            streams = []
            def branch():
                streams.append(torch.cuda.current_stream(1).cuda_stream)
                return y + 1
            main, side = pair.run(branch, branch, (y,))
            self.assertIs(pair._stream, original_stream)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(streams[0], streams[1])
            torch.testing.assert_close(main + side, y * 2 + 2, rtol=0, atol=0)

    def test_main_exception_does_not_execute_shared(self):
        with torch.no_grad():
            pair = CmpMoeStreams()
            x = torch.ones(8, device="cuda")
            pair.run(lambda: x + 1, lambda: x + 2, (x,))
            def fail():
                raise ValueError("main branch failure")
            with self.assertRaisesRegex(ValueError, "main branch failure"):
                pair.run(fail, lambda: self.fail("shared must not run"), (x,))
            main, side = pair.run(lambda: x + 3, lambda: x + 4, (x,))
            torch.testing.assert_close(main + side, x * 2 + 7, rtol=0, atol=0)

    def test_alternating_graphs_share_helper(self):
        with torch.no_grad():
            first, _ = self.make_layer()
            second, _ = self.make_layer()
            second._cmp_moe_streams = first._cmp_moe_streams
            caller = torch.cuda.Stream()
            caller.wait_stream(torch.cuda.current_stream())
            graphs = []
            with torch.cuda.stream(caller):
                for layer, rows in ((first, 1), (second, 32)):
                    x = torch.ones((rows, 6144), device="cuda", dtype=torch.bfloat16)
                    for _ in range(3):
                        layer.forward_prepacked(x, None, None)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=caller):
                        result = layer.forward_prepacked(x, None, None)
                    graphs.append((graph, x, result))
                for value in (0, -3, 7, 1):
                    for graph, x, result in graphs:
                        x.fill_(value)
                        graph.replay()
                        caller.synchronize()
                        torch.testing.assert_close(result, torch.full_like(result, value * 5), rtol=0, atol=0)

    def test_hook_falls_back(self):
        with torch.no_grad():
            layer, streams = self.make_layer()
            layer.shared_expert.register_forward_hook(lambda *args: None)
            x = torch.ones((16,6144), device="cuda", dtype=torch.bfloat16)
            layer.forward_prepacked(x,None,None)
            self.assertEqual(streams["shared"], streams["routed"])

    def test_cold_capture_is_serial(self):
        with torch.no_grad():
            layer, streams = self.make_layer()
            x = torch.ones((16,6144), device="cuda", dtype=torch.bfloat16)
            caller = torch.cuda.Stream()
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=caller):
                layer.forward_prepacked(x,None,None)
            self.assertIsNone(layer._cmp_moe_streams._stream)
            self.assertEqual(streams["shared"], streams["routed"])

    def test_exception_joins_side(self):
        with torch.no_grad():
            pair = CmpMoeStreams()
            x = torch.zeros(32,device="cuda")
            def side():
                torch.cuda._sleep(100000)
                x.fill_(9)
                raise ValueError("test exception")
            def main():
                return x + 1
            with self.assertRaisesRegex(ValueError,"test exception"):
                pair.run(main,side,(x,))
            result=x.clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(result,torch.full_like(x,9))

if __name__ == "__main__":
    unittest.main()
