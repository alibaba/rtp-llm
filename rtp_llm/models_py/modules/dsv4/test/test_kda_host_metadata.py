"""Exact KDA metadata and asynchronous submission with queued CUDA work."""

import unittest
from itertools import accumulate
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.forward_metadata import (
    _FORWARD_METADATA,
    cached_kda_chunk_indices,
    cached_prefill_prefix_summary,
    tensor_key,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
    BLOCK_M,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices


class TestKdaHostMetadata(unittest.TestCase):
    def test_exact_geometry_and_scope_lifetime(self):
        for dtype in (torch.int32, torch.int64):
            for lengths in ([1], [0, 0], [0, 1, 63, 64, 65, 0, 129], [100003, 8179]):
                host = torch.tensor([0, *accumulate(lengths)], dtype=dtype)
                cu = host.cuda()
                expected = {
                    chunk: prepare_chunk_indices(cu, chunk).cpu()
                    for chunk in (64, 128, 256)
                }
                for step in range(2):
                    token = _FORWARD_METADATA.set({})
                    try:
                        meta = prepare_causal_conv1d_metadata(cu, cu.device, host)
                        counts = [(n + BLOCK_M - 1) // BLOCK_M for n in lengths]
                        torch.testing.assert_close(
                            meta.batch_ptr.cpu(),
                            torch.tensor(
                                [
                                    i
                                    for i, count in enumerate(counts)
                                    for _ in range(count)
                                ]
                                + [-1],
                                dtype=torch.int32,
                            ),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            meta.token_chunk_offset_ptr.cpu(),
                            torch.tensor(
                                [j for count in counts for j in range(count)] + [-1],
                                dtype=torch.int32,
                            ),
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(meta.total, sum(counts))
                        _FORWARD_METADATA.get()[("prefill_cu_host", tensor_key(cu))] = (
                            tuple(host.tolist())
                        )
                        for chunk in (64, 128, 256):
                            actual = prepare_chunk_indices(cu, chunk)
                            self.assertIs(actual, prepare_chunk_indices(cu, chunk))
                            # Use a new reference input: FLA's legacy identity cache
                            # cannot observe mutation of the previous cu tensor.
                            if step:
                                expected[chunk] = prepare_chunk_indices(
                                    cu.clone(), chunk
                                ).cpu()
                            torch.testing.assert_close(
                                actual.cpu(), expected[chunk], rtol=0, atol=0
                            )
                    finally:
                        _FORWARD_METADATA.reset(token)
                    self.assertIsNone(cached_kda_chunk_indices(cu, 64))
                    lengths = [n + 17 for n in lengths]
                    host.copy_(torch.tensor([0, *accumulate(lengths)], dtype=dtype))
                    cu.copy_(host)

    def test_cpu_return_does_not_wait_for_prior_gpu_work(self):
        host = torch.tensor([0, 81, 161, 322], dtype=torch.int32, pin_memory=True)
        cu = host.cuda()
        prefix_host = torch.tensor([0, 8192, 128], dtype=torch.int32)
        prefix = prefix_host.cuda()
        # Warm CUDA/pinned allocators before the pending-event check.
        for attempt in range(2):
            token = _FORWARD_METADATA.set({})
            try:
                _FORWARD_METADATA.get()[("prefill_cu_host", tensor_key(cu))] = tuple(
                    host.tolist()
                )
                original_cpu = torch.Tensor.cpu

                def cpu(tensor, *args, **kwargs):
                    if tensor.is_cuda:
                        raise AssertionError("metadata performed a device-to-host read")
                    return original_cpu(tensor, *args, **kwargs)

                torch.cuda.synchronize()
                if attempt:
                    torch.cuda._sleep(300_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                with patch.object(torch.Tensor, "cpu", cpu):
                    self.assertEqual(
                        cached_prefill_prefix_summary(prefix, prefix_host),
                        (True, False),
                    )
                    self.assertEqual(
                        cached_prefill_prefix_summary(prefix), (True, False)
                    )
                    conv = prepare_causal_conv1d_metadata(cu, cu.device, host)
                    chunks = prepare_chunk_indices(cu, 64)
                if attempt:
                    self.assertFalse(pending.query(), "metadata drained prior GPU work")
                torch.cuda.synchronize()
                self.assertEqual(conv.sequence_boundaries_host, tuple(host.tolist()))
                torch.testing.assert_close(
                    chunks.cpu(), prepare_chunk_indices(cu, 64).cpu(), rtol=0, atol=0
                )
            finally:
                _FORWARD_METADATA.reset(token)

    def test_fallback_and_invalid_mirror(self):
        cu = torch.tensor([0, 17, 81], dtype=torch.int32, device="cuda")
        actual = prepare_causal_conv1d_metadata(cu, cu.device)
        expected = prepare_causal_conv1d_metadata(cu, cu.device, cu.cpu())
        torch.testing.assert_close(actual.batch_ptr, expected.batch_ptr, rtol=0, atol=0)
        token = _FORWARD_METADATA.set({})
        try:
            self.assertIsNone(cached_kda_chunk_indices(cu, 64))
            with self.assertRaises(ValueError):
                prepare_causal_conv1d_metadata(cu, cu.device, cu)
            with self.assertRaises(ValueError):
                prepare_causal_conv1d_metadata(cu, cu.device, torch.zeros(2))
        finally:
            _FORWARD_METADATA.reset(token)


if __name__ == "__main__":
    unittest.main()
