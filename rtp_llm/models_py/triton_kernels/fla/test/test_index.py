import unittest
import functools
import struct

import torch

from rtp_llm.models_py.triton_kernels.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_lens,
)


def _legacy_identity_tensor_cache(fn):
    cache_entries = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    key in last_kwargs and value is last_kwargs[key]
                    for key, value in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@_legacy_identity_tensor_cache
def _legacy_prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]


@_legacy_identity_tensor_cache
def _legacy_prepare_chunk_indices(cu_seqlens, chunk_size):
    indices = torch.cat(
        [
            torch.arange((n + chunk_size - 1) // chunk_size)
            for n in _legacy_prepare_lens(cu_seqlens).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@_legacy_identity_tensor_cache
def _legacy_prepare_chunk_offsets(cu_seqlens, chunk_size):
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            torch.div(
                _legacy_prepare_lens(cu_seqlens) + chunk_size - 1,
                chunk_size,
                rounding_mode="floor",
            ),
        ]
    ).cumsum(-1)


class IndexCacheTest(unittest.TestCase):
    def test_legacy_identity_cache_reproduces_stale_external_cpu_buffer(self):
        raw = bytearray(3 * 4)
        struct.pack_into("3i", raw, 0, 0, 64, 128)
        cu_seqlens = torch.frombuffer(raw, dtype=torch.int32)

        self.assertEqual(_legacy_prepare_lens(cu_seqlens).tolist(), [64, 64])
        self.assertEqual(
            _legacy_prepare_chunk_indices(cu_seqlens, 64).tolist(),
            [[0, 0], [1, 0]],
        )
        self.assertEqual(
            _legacy_prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 1, 2]
        )

        struct.pack_into("3i", raw, 0, 0, 128, 192)

        self.assertEqual(cu_seqlens.tolist(), [0, 128, 192])
        self.assertEqual((cu_seqlens[1:] - cu_seqlens[:-1]).tolist(), [128, 64])
        self.assertEqual(_legacy_prepare_lens(cu_seqlens).tolist(), [64, 64])
        self.assertEqual(
            _legacy_prepare_chunk_indices(cu_seqlens, 64).tolist(),
            [[0, 0], [1, 0]],
        )
        self.assertEqual(
            _legacy_prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 1, 2]
        )

    def test_cached_helpers_refresh_when_cu_seqlens_mutates(self):
        cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [64, 64])
        self.assertEqual(prepare_chunk_indices(cu_seqlens, 64).tolist(), [[0, 0], [1, 0]])
        self.assertEqual(prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 1, 2])

        cu_seqlens.copy_(torch.tensor([0, 128, 192], dtype=torch.int32))

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [128, 64])
        self.assertEqual(
            prepare_chunk_indices(cu_seqlens, 64).tolist(),
            [[0, 0], [0, 1], [1, 0]],
        )
        self.assertEqual(prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 2, 3])

    def test_cached_helpers_refresh_when_external_cpu_buffer_mutates(self):
        raw = bytearray(3 * 4)
        struct.pack_into("3i", raw, 0, 0, 64, 128)
        cu_seqlens = torch.frombuffer(raw, dtype=torch.int32)

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [64, 64])
        self.assertEqual(prepare_chunk_indices(cu_seqlens, 64).tolist(), [[0, 0], [1, 0]])
        self.assertEqual(prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 1, 2])

        struct.pack_into("3i", raw, 0, 0, 128, 192)

        self.assertEqual(prepare_lens(cu_seqlens).tolist(), [128, 64])
        self.assertEqual(
            prepare_chunk_indices(cu_seqlens, 64).tolist(),
            [[0, 0], [0, 1], [1, 0]],
        )
        self.assertEqual(prepare_chunk_offsets(cu_seqlens, 64).tolist(), [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
