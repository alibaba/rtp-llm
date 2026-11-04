import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

_path = Path(__file__).resolve().parents[1] / "fused_qk_rope_cat_cache_mla.py"
_spec = importlib.util.spec_from_file_location("mla_cache_kernel", _path)
kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kernel)


class MlaCacheDeviceTest(unittest.TestCase):
    def test_cache_device_contract(self):
        device = torch.device("cuda:0")
        for cache_device, pinned, valid in (
            ("cuda:0", False, True),
            ("cuda:1", False, False),
            ("cpu", True, True),
            ("cpu", False, False),
        ):
            with self.subTest(device=cache_device, pinned=pinned):
                cache = Mock(device=torch.device(cache_device), shape=(2, 64, 656))
                cache.is_pinned.return_value = pinned
                cache.stride.return_value = (41984, 656, 1)
                if valid:
                    kernel.validate_mla_cache_device(cache, device)
                else:
                    with self.assertRaisesRegex(ValueError, "MLA cache must"):
                        kernel.validate_mla_cache_device(cache, device)

    def test_fp8_constant_uses_compute_device(self):
        device = torch.device("cuda:0")

        def tensor(shape, dtype):
            return Mock(
                wraps=torch.empty(shape, dtype=dtype),
                device=device,
                dtype=dtype,
                shape=shape,
            )

        cache = torch.empty((2, 64, 656), dtype=torch.uint8)
        launch = Mock()
        dispatch = Mock()
        dispatch.__getitem__ = Mock(return_value=launch)
        with (
            patch.object(kernel, "validate_mla_cache_device"),
            patch.object(kernel, "_get_fp8_scale_min") as scale,
            patch.object(kernel, "_fused_qk_rope_cat_cache_mla_fp8_kernel", dispatch),
        ):
            kernel.fused_qk_rope_cat_cache_mla(
                tensor((64, 8, 256), torch.bfloat16),
                tensor((64, 512), torch.bfloat16),
                tensor((64, 64), torch.bfloat16),
                cache,
                tensor((64,), torch.int64),
                tensor((64,), torch.int32),
                tensor((128, 64), torch.float32),
                512, 64, False, "fp8_ds_mla",
            )
            scale.assert_called_once_with(device)
            self.assertIs(launch.call_args.args[9], scale.return_value)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_pinned_fp8_cache_graph_replay(self):
        device = torch.device("cuda", torch.cuda.current_device())
        q = torch.randn((64, 8, 256), device=device, dtype=torch.bfloat16)
        ck = torch.randn((64, 512), device=device, dtype=torch.bfloat16)
        kpe = torch.randn((64, 64), device=device, dtype=torch.bfloat16)
        positions = torch.arange(64, device=device, dtype=torch.int32)
        slots = torch.arange(64, device=device, dtype=torch.int64)
        csc = torch.randn((128, 64), device=device, dtype=torch.float32)
        for strided in (False, True):
            with self.subTest(strided=strided):
                backing = torch.zeros((4, 64, 656), dtype=torch.uint8, pin_memory=True)
                host = backing[::2] if strided else backing[:2]
                gpu = torch.zeros_like(host, device=device)
                q_host, q_gpu = q.clone(), q.clone()
                k_host, k_gpu = kpe.clone(), kpe.clone()

                def run(query, key, cache):
                    kernel.fused_qk_rope_cat_cache_mla(
                        query, ck, key, cache, slots, positions, csc,
                        512, 64, False, "fp8_ds_mla",
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(q_host, k_host, host)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(q_host, k_host, host)
                for offset in (0, 64, 0):
                    slots.copy_(torch.arange(64, device=device) + offset)
                    q_host.copy_(q)
                    q_gpu.copy_(q)
                    k_host.copy_(kpe)
                    k_gpu.copy_(kpe)
                    run(q_gpu, k_gpu, gpu)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(host, gpu.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(q_host, q_gpu, rtol=0, atol=0)
                    torch.testing.assert_close(k_host, k_gpu, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
