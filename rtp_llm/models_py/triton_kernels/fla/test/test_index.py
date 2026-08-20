import struct
import unittest

import torch

from rtp_llm.models_py.triton_kernels.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_lens,
)
from rtp_llm.models_py.triton_kernels.fla.utils import (
    _CUDA_GRAPH_CACHE_HOLDS_ATTR,
)


class IndexCacheTest(unittest.TestCase):
    def test_cuda_graph_pins_warmup_cache_hit_to_input_lifetime(self):
        cu_seqlens = torch.tensor([0, 45], dtype=torch.int32, device="cuda")
        warmup_indices = prepare_chunk_indices(cu_seqlens, 64)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_indices = prepare_chunk_indices(cu_seqlens, 64)
            output = captured_indices + 1

        holds = getattr(cu_seqlens, _CUDA_GRAPH_CACHE_HOLDS_ATTR)
        self.assertTrue(any(held is warmup_indices for held in holds))
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output.tolist(), [[1, 1]])

    def test_cached_helpers_support_inference_tensors(self):
        with torch.inference_mode():
            cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [64, 64])
        self.assertEqual(
            prepare_chunk_indices(cu_seqlens, 64).tolist(), [[0, 0], [1, 0]]
        )
        self.assertEqual(
            prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 1, 2]
        )

    def test_cached_helpers_refresh_after_tensor_copy(self):
        cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [64, 64])
        self.assertEqual(
            prepare_chunk_indices(cu_seqlens, 64).tolist(), [[0, 0], [1, 0]]
        )
        self.assertEqual(
            prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 1, 2]
        )

        cu_seqlens.copy_(torch.tensor([0, 128, 192], dtype=torch.int32))

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [128, 64])
        self.assertEqual(
            prepare_chunk_indices(cu_seqlens, 64).tolist(),
            [[0, 0], [0, 1], [1, 0]],
        )
        self.assertEqual(
            prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 2, 3]
        )

    def test_cached_helpers_refresh_after_external_buffer_mutation(self):
        raw = bytearray(3 * 4)
        struct.pack_into("3i", raw, 0, 0, 64, 128)
        cu_seqlens = torch.frombuffer(raw, dtype=torch.int32)

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [64, 64])
        self.assertEqual(
            prepare_chunk_indices(cu_seqlens, 64).tolist(), [[0, 0], [1, 0]]
        )
        self.assertEqual(
            prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 1, 2]
        )

        # The C++ input gatherer mutates an externally-owned buffer without
        # touching PyTorch's version counter.  The content fingerprint must
        # still invalidate all derived FLA metadata.
        old_version = cu_seqlens._version
        struct.pack_into("3i", raw, 0, 0, 128, 192)
        self.assertEqual(cu_seqlens._version, old_version)

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [128, 64])
        self.assertEqual(
            prepare_chunk_indices(cu_seqlens, 64).tolist(),
            [[0, 0], [0, 1], [1, 0]],
        )
        self.assertEqual(
            prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 2, 3]
        )


if __name__ == "__main__":
    unittest.main()
