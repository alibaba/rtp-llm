import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.kimi_k25.test.test_kimi_compressed import owner
from rtp_llm.models_py.new_models.kimi_k25.test.test_kimi_vision import (
    check_loaded_forward,
)
from rtp_llm.models_py.weight_mapper import get_all_weights


class KimiK25VisionH20(unittest.TestCase):
    def test_routed_w4a8_repack_matches_signed_int4_cuda_reference(self):
        from rtp_kernel.w4a8_group_gemm import (
            pack_scale_fp8,
            reorder_tensor,
            unified_encode_int4b,
        )

        layer = owner(ep=2, er=1)
        checkpoint, reference = {}, {}
        for expert in range(4):
            for p, (name, n, k) in enumerate(
                (("gate", 128, 64), ("up", 128, 64), ("down", 64, 128))
            ):
                prefix = f"{expert}.{name}_proj."
                signed = (torch.arange(n * k).reshape(n, k) + expert * 3 + p) % 16 - 8
                scale = (
                    torch.arange(n * k // 32).reshape(n, k // 32) % 4 + 1
                ).bfloat16()
                offset = (signed + 8).to(torch.uint8)
                checkpoint[prefix + "weight_packed"] = (
                    (offset[:, ::2] | (offset[:, 1::2] << 4))
                    .contiguous()
                    .view(torch.int32)
                )
                checkpoint[prefix + "weight_shape"] = torch.tensor([n, k])
                checkpoint[prefix + "weight_scale"] = scale
                reference[expert, name] = (signed, scale)
        with tempfile.TemporaryDirectory() as directory:
            shard = os.path.join(directory, "experts.safetensors")
            save_file(checkpoint, shard)
            for name, value in get_all_weights([shard]):
                layer.load_weights({name: value})
        self.assertTrue(layer.checkpoint_load_complete())
        layer.quant_method.validate_runtime_device(torch.device("cuda:0"))
        layer.to("cuda:0")
        # Exercise exactly the registered conversion, independently of the
        # full fused-MoE executor (which requires the native engine).
        layer.quant_method.process_weights_after_loading(layer)
        for idx, expert in enumerate((2, 3)):
            for names, weight, scales in (
                (("up", "gate"), layer.w13[idx], layer.w13_scale[idx]),
                (("down",), layer.w2[idx], layer.w2_scale[idx]),
            ):
                signed, scale = (
                    torch.cat(t, dim=0).cuda()
                    for t in zip(*(reference[expert, name] for name in names))
                )
                nibble = signed.to(torch.int8) & 15
                signed_packed = (nibble[:, ::2] | (nibble[:, 1::2] << 4)).contiguous()
                expected = reorder_tensor(unified_encode_int4b(signed_packed))
                scale_expected = pack_scale_fp8(
                    scale.t().contiguous().to(torch.float8_e4m3fn)
                )
                torch.testing.assert_close(weight, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    scales.view(torch.uint8),
                    scale_expected.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
        self.assertFalse(hasattr(layer, "_w4a8_up_packed"))

    def test_sharded_image_and_video_grid_after_checkpoint_close(self):
        self.assertTrue(
            torch.cuda.is_available(), "H20 target requires a real CUDA GPU"
        )
        self.assertIsNone(torch.version.hip)
        for method in (
            NewLoaderLoadMethod.SCRATCH,
            NewLoaderLoadMethod.FASTSAFETENSORS,
        ):
            with self.subTest(method=method.value):
                check_loaded_forward(self, "cuda:0", method)
                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
