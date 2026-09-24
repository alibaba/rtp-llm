"""Native BF16 combine storage must preserve the FP32 shared-add boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.triton_kernels.moe.shared_expert import fused_moe_epilogue
from rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency import (
    low_latency_mxfp4_moe,
)


class MoeOutputContractTest(unittest.TestCase):
    def test_decode_block_forwards_optional_mask_only_when_present(self):
        from rtp_llm.models_py.modules.dsv4.block import Block

        x = torch.ones(2, 3, 4, 8)
        ids = torch.zeros(2, 3, dtype=torch.int64)
        mask = torch.tensor([True, True, True, False, False, False])
        hc = SimpleNamespace(
            pre_norm=lambda value, *args, **kwargs: (value, None, None),
            post=lambda value, *args: value,
        )
        ffn = Mock(return_value=x)
        block = SimpleNamespace(
            layer_id=0, tp_size=1, tp_rank=0, attn_hc=hc, ffn_hc=hc,
            attn_norm=None, ffn_norm=None, ffn=ffn,
        )
        with patch(
            "rtp_llm.models_py.modules.dsv4._record_tensor.should_record_layer",
            return_value=False,
        ):
            for value in (None, mask):
                metadata = SimpleNamespace(active_token_mask=value)
                Block.forward_decode(block, x, metadata, ids, attn_fn=lambda x: x)
                expected = {"is_decode_forward": True}
                if value is not None:
                    expected["active_token_mask"] = mask
                self.assertEqual(ffn.call_args.kwargs.keys(), expected.keys())
                if value is not None:
                    self.assertIs(ffn.call_args.kwargs["active_token_mask"], mask)

    def test_invalid_output_dtype_fails_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, "routed output"):
            low_latency_mxfp4_moe(
                None,
                None,
                None,
                None,
                None,
                None,
                num_experts=256,
                max_dispatch_tokens=128,
                expected_m=24,
                output_dtype=torch.float16,
            )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class MoeOutputGraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_native_storage_and_materialized_fp32_have_identical_epilogues(self):
        torch.manual_seed(890432)
        for batch in (1, 3, 8, 32, 128):
            for shared_dtype in (torch.bfloat16, torch.float32):
                for strided in (False, True):
                    width = 4096
                    storage_width = width * (2 if strided else 1)
                    routed = torch.empty(
                        (batch, storage_width), device="cuda", dtype=torch.bfloat16
                    )[:, :width]
                    shared = torch.empty(
                        (batch, storage_width), device="cuda", dtype=shared_dtype
                    )[:, :width]
                    actual = torch.empty_like(routed)

                    def run():
                        return fused_moe_epilogue(
                            routed, shared, torch.bfloat16, out=actual
                        )

                    routed.normal_()
                    shared.normal_()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        run()
                    torch.cuda.current_stream().wait_stream(stream)
                    address = actual.data_ptr()
                    for step in range(5):
                        routed.normal_().mul_(2.0 ** (step * 4 - 8))
                        # Cancellation plus a small FP32 residual detects any
                        # premature cast of the shared result to BF16.
                        shared.copy_(
                            -routed.float() + torch.randn_like(routed.float()) / 128
                        )
                        expected = (routed.float() + shared.float()).bfloat16()
                        materialized = fused_moe_epilogue(
                            routed.float(), shared, torch.bfloat16
                        )
                        actual.fill_(float("nan"))
                        graph.replay()
                        self.assertEqual(actual.data_ptr(), address)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(actual, materialized, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
