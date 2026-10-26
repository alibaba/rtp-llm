"""Indexer provider selection and cleanup after partial GPU submission."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires M890P",
)
class DecodeIndexerGpuTest(unittest.TestCase):
    @torch.inference_mode()
    def test_exception_joins_pending_work_before_pool_release(self):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_indexer import (
            prepare_decode_indexer_overlap,
        )

        outer = torch.cuda.Stream()
        producer = torch.cuda.current_stream()
        streams = {name: torch.cuda.Stream() for name in ("q", "weights")}
        x = torch.ones(8, 1, 128, device="cuda", dtype=torch.bfloat16)
        qr = torch.ones(8, 1, 1024, device="cuda", dtype=torch.bfloat16)
        positions = torch.zeros(8, device="cuda", dtype=torch.int32)
        metadata = SimpleNamespace(positions=positions)
        scratch = torch.zeros(2048, 2048, device="cuda")
        outer.wait_stream(producer)
        module = "rtp_llm.platforms.ppu.models.dsv4.ppu_decode_indexer"

        for failure in ("compressor", "weights", "q", "quantize"):
            with self.subTest(failure=failure):
                submitted = []

                def fail_at(stage):
                    if stage == failure:
                        # Keep work pending when Python raises. Synchronizing
                        # only the caller must still complete this event.
                        for _ in range(64):
                            scratch.add_(1)
                        event = torch.cuda.Event()
                        event.record()
                        submitted.append(event)
                        raise RuntimeError("injected " + stage)

                def compressor(*args, **kwargs):
                    fail_at("compressor")

                def weights(*args, **kwargs):
                    self.assertEqual(torch.cuda.current_stream(), streams["weights"])
                    fail_at("weights")
                    return torch.ones(8, 64, device="cuda", dtype=torch.bfloat16)

                def query(*args, **kwargs):
                    self.assertEqual(torch.cuda.current_stream(), streams["q"])
                    fail_at("q")
                    return torch.ones(8, 64, 128, device="cuda", dtype=torch.bfloat16)

                def quantize(*args, **kwargs):
                    fail_at("quantize")

                indexer = SimpleNamespace(
                    compressor=SimpleNamespace(forward_decode_vectorized=compressor),
                    _compute_indexer_q=query,
                    n_heads=64,
                    weights_proj=None,
                    weight_scale=1.0,
                    freqs_cis=torch.ones(1, 32, dtype=torch.complex64, device="cuda"),
                )
                outer.wait_stream(producer)
                with patch(module + ".F.linear", side_effect=weights), patch(
                    module + ".quantize_q", side_effect=quantize
                ), torch.cuda.stream(outer):
                    with self.assertRaisesRegex(RuntimeError, "injected " + failure):
                        prepare_decode_indexer_overlap(
                            indexer,
                            x,
                            qr,
                            positions,
                            positions,
                            metadata,
                            producer,
                            streams,
                        )
                outer.synchronize()
                self.assertEqual(len(submitted), 1)
                self.assertTrue(submitted[0].query())


if __name__ == "__main__":
    unittest.main()
