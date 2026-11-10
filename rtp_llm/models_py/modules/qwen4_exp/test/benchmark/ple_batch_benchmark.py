"""Manual PLE batch timing and allocator benchmark."""

import gc
import json
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.test.support.ple_gate import (
    PLESourceBf16Fixture,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PleBatchBenchmark(PLESourceBf16Fixture, unittest.TestCase):
    def test_component_timing_and_allocator_receipt(self):
        gc.collect()
        torch.cuda.current_stream().synchronize()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            layer = self.layer()
            torch.manual_seed(81500)
            hidden = torch.randn(1, 8192, 10240, device="cuda", dtype=torch.bfloat16)
            embedding = torch.randn(1, 8192, 2560, device="cuda", dtype=torch.bfloat16)
            initial = torch.randn(1, 9, 10240, device="cuda", dtype=torch.bfloat16)
            tokens = torch.zeros((1, 8194), device="cuda", dtype=torch.int64)
            # Strong local references keep every persistent fixture allocation alive.
            persistent = (layer, hidden, embedding, initial, tokens)
            receipt = {
                "scope": "unmocked complete PLE component; synthetic source-BF16; not E2E",
                "shape": [1, 8192, 10240],
                "page_size": 128,
                "state_dtype": "bfloat16",
                "ngram_context_dtype": "int64",
                "warmup_runs": 3,
                "measure_runs": 7,
                "variant_order": ["0", "1"],
                "gc_initially_enabled": gc_was_enabled,
                "gc_disabled_during_warmup_measurement": True,
                "gc_explicit_before_each_sample": True,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "variants": {},
            }
            for enabled in ("0", "1"):
                with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": enabled}):
                    gc.collect()
                    torch.cuda.current_stream().synchronize()
                    for _ in range(3):
                        output, snapshots = layer.prefill_page_snapshots(
                            hidden,
                            tokens,
                            128,
                            initial,
                            embeddings=embedding,
                            add_residual=True,
                        )
                        del output, snapshots
                    torch.cuda.current_stream().synchronize()
                    samples = []
                    for _ in range(7):
                        gc.collect()
                        torch.cuda.current_stream().synchronize()
                        before_allocated = torch.cuda.memory_allocated()
                        before_reserved = torch.cuda.memory_reserved()
                        torch.cuda.reset_peak_memory_stats()
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        output, snapshots = layer.prefill_page_snapshots(
                            hidden,
                            tokens,
                            128,
                            initial,
                            embeddings=embedding,
                            add_residual=True,
                        )
                        end.record()
                        end.synchronize()
                        after_return_allocated = torch.cuda.memory_allocated()
                        snapshot_storages = {
                            s.untyped_storage().data_ptr(): s.untyped_storage().nbytes()
                            for s in snapshots
                        }
                        sample = {
                            "cuda_event_ms": start.elapsed_time(end),
                            "allocated_after_return_bytes": after_return_allocated,
                            "retained_allocated_delta_bytes": after_return_allocated
                            - before_allocated,
                            "snapshot_payload_bytes": sum(
                                s.numel() * s.element_size() for s in snapshots
                            ),
                            "snapshot_unique_storage_bytes": sum(
                                snapshot_storages.values()
                            ),
                            "snapshot_unique_storage_count": len(snapshot_storages),
                            "allocated_before_bytes": before_allocated,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated()
                            - before_allocated,
                            "reserved_before_bytes": before_reserved,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                        self.assertEqual(len(snapshots), 64)
                        del output, snapshots
                        torch.cuda.current_stream().synchronize()
                        after_delete_allocated = torch.cuda.memory_allocated()
                        sample.update(
                            allocated_after_delete_bytes=after_delete_allocated,
                            allocated_after_delete_delta_bytes=after_delete_allocated
                            - before_allocated,
                            memory_sample_valid=(
                                after_return_allocated >= before_allocated
                                and after_delete_allocated == before_allocated
                            ),
                        )
                        samples.append(sample)
                    receipt["variants"][enabled] = samples
            self.assertEqual(len(persistent), 5)
            receipt["all_memory_samples_valid"] = all(
                s["memory_sample_valid"]
                for samples in receipt["variants"].values()
                for s in samples
            )
            print(
                "PLE_BF16_BATCH_COMPONENT_BENCH=" + json.dumps(receipt, sort_keys=True)
            )
            self.assertTrue(
                receipt["all_memory_samples_valid"],
                "Allocator receipt is contaminated or retains unexpected allocations",
            )
        finally:
            if gc_was_enabled:
                gc.enable()
            gc.collect()
            torch.cuda.current_stream().synchronize()


if __name__ == "__main__":
    unittest.main()
