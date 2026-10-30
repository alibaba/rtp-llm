"""Full Prefill numerical acceptance on M890P, including native FP8 GEMMs."""

import unittest
from types import SimpleNamespace

import torch
from rtp_llm.device.device_type import is_ppu
from rtp_llm.platforms.ppu.models.dsv4.ppu_shared_phases import SharedPhases
from rtp_llm.platforms.ppu.models.dsv4.ppu_tp_shared_expert import PpuTPSharedExpert

import test_prefill_optimizations as common


class PrefillM890PTest(common.PrefillOperatorsTest):
    def setUp(self):
        if not is_ppu() or torch.cuda.get_device_capability() != (8, 9):
            raise RuntimeError("Native Prefill FP8 acceptance requires M890P (PPU 1.5)")

    @torch.inference_mode()
    def test_shared_phases_match_serial_on_nondefault_stream(self):
        torch.manual_seed(890430)

        def weight(shape):
            return (torch.randn(shape, device="cuda") / 8).to(torch.float8_e4m3fn)

        shared = PpuTPSharedExpert(
            4096,
            2048,
            {
                "w13_w": weight((1024, 4096)),
                "w13_s": torch.ones((8, 32), device="cuda"),
                "w2_w": weight((4096, 512)),
                "w2_s": torch.ones((32, 4), device="cuda"),
            },
            tp_size=4,
            tp_rank=0,
            swiglu_limit=10.0,
            platform_provider=SimpleNamespace(_bool=lambda _, default: default),
        )
        producer, aux = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            for rows in (1024, 2049):
                for _ in range(3):
                    x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
                    expected = shared(x)
                    phases = SharedPhases(shared, x, aux)
                    phases.before_up()
                    # Independent work between joins exercises stream ordering.
                    routed = x.square()
                    phases.after_up()
                    routed.add_(1)
                    phases.before_down()
                    actual = phases.output + routed
                    torch.testing.assert_close(
                        actual, expected + routed, rtol=0, atol=0
                    )
        torch.cuda.current_stream().wait_stream(producer)


if __name__ == "__main__":
    unittest.main()
