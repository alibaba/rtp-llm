"""cuLA page mappings, host lifetime and nonblocking CUDA preparation."""
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import (
    prepare_cula_kda_prefill_metadata,
)


class KdaMetadataTest(unittest.TestCase):
    def prepare(self, device):
        # Two real requests surround a virtual row. Repeated checkpoint slot 2
        # must retain only its last publication; the second request reads slot 4.
        states = torch.zeros((8, 2, 128, 128), dtype=torch.float32, device=device)
        return prepare_cula_kda_prefill_metadata(
            [0, 130, 194, 259], [0, 0, 64],
            [[1, 2, 2], [0], [4, 5, 6]], 64, states,
        )

    def check_mapping(self, metadata):
        packed = metadata.packed
        self.assertIsNotNone(packed)
        self.assertEqual(packed.initial_blocks.cpu().tolist(), [-1, -1, 4])
        self.assertEqual(packed.checkpoint_blocks.cpu().tolist(), [1, -1, 2, -1, 5, 6])
        self.assertEqual(packed.cu_seqlens_cpu.tolist(), [0, 130, 194, 259])
        self.assertEqual(packed.checkpoint_offsets.cpu().tolist(), [0, 3, 4, 6])
        self.assertEqual(packed.padding_ranges, ((130, 194),))
        self.assertEqual(packed.initial_states.dtype, torch.float32)
        self.assertEqual(packed.checkpoints.shape, (1, 6, 2, 128, 128))
        self.assertEqual(len(packed.host_tensors), 4)
        self.assertTrue(all(t.device.type == "cpu" for t in packed.host_tensors))

    def test_exact_page_mapping_and_independent_round_lifetimes(self):
        first, second = self.prepare("cpu"), self.prepare("cpu")
        self.check_mapping(first)
        self.check_mapping(second)
        for a, b in zip(first.packed.host_tensors, second.packed.host_tensors):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_cuda_metadata_has_no_blocking_copy(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        with torch.device("cuda"):
            warm = self.prepare("cuda")
        torch.cuda.synchronize()
        del warm
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            with torch.profiler.record_function("k31.metadata.prepare"):
                with torch.device("cuda"):
                    actual = self.prepare("cuda")
        self.check_mapping(actual)
        self.assertTrue(all(t.is_pinned() for t in actual.packed.host_tensors))
        for event in profile.events():
            if event.name != "cudaStreamSynchronize":
                continue
            parent = event.cpu_parent
            while parent is not None:
                self.assertNotEqual(parent.name, "k31.metadata.prepare")
                parent = parent.cpu_parent


if __name__ == "__main__":
    unittest.main()
