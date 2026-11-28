"""Cold-cache recompile coverage for the DSV4 CP metadata kernels."""

import unittest

import pytest
import torch

from rtp_llm.models_py.modules.dsv4 import _cp_metadata_triton
from rtp_llm.models_py.modules.dsv4._cp_metadata_triton import (
    try_build_cp_forward_metadata,
    try_build_cp_full_prefill_positions,
    try_build_cp_restore_indices,
)
from rtp_llm.models_py.modules.dsv4.cp import cp_padded_local_kv_len
from rtp_llm.models_py.modules.dsv4.test.test_cp_metadata_triton import (
    _cp_forward_inputs,
    _random_varlen_case,
    _triton_cache_size,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CpMetadataTritonRecompileTest(unittest.TestCase):
    @pytest.mark.manual
    def test_power_of_two_warmup_covers_non_power_batches_without_recompile(self):
        kernels = (
            _cp_metadata_triton._cp_restore_varlen_kernel,
            _cp_metadata_triton._cp_positions_varlen_kernel,
            _cp_metadata_triton._cp_forward_metadata_kernel,
        )
        for kernel in kernels:
            kernel.device_caches.clear()

        def launch_all(batch_size: int) -> None:
            lengths_host, prefixes_host = _random_varlen_case(
                batch_size, seed=2026082200 + batch_size
            )
            lengths = torch.tensor(lengths_host, dtype=torch.int64, device="cuda")
            prefixes = torch.tensor(prefixes_host, dtype=torch.int64, device="cuda")
            total_tokens = sum(lengths_host)
            total_local = sum(
                cp_padded_local_kv_len(length, 4, 4) for length in lengths_host
            )
            self.assertIsNotNone(
                try_build_cp_restore_indices(
                    lengths,
                    cp_size=4,
                    owner_block_size=4,
                    total_tokens=total_tokens,
                    total_local_kv=total_local,
                )
            )
            self.assertIsNotNone(
                try_build_cp_full_prefill_positions(
                    lengths, prefixes, total_tokens=total_tokens
                )
            )
            forward_inputs = _cp_forward_inputs(
                lengths_host, prefixes_host, cp_size=4, cp_rank=0
            )
            self.assertIsNotNone(
                try_build_cp_forward_metadata(
                    forward_inputs["lengths"],
                    forward_inputs["chunks"],
                    forward_inputs["prefixes"],
                    forward_inputs["padding"],
                    forward_inputs["restore"],
                    forward_inputs["shuffle"],
                    cp_size=4,
                    cp_rank=0,
                    chunk_length=forward_inputs["chunk_length"],
                    seq_len_full=forward_inputs["seq_len_full"],
                )
            )

        for batch_block in (4, 8, 16, 32, 64):
            with self.subTest(batch_block=batch_block):
                cache_before_warmup = tuple(
                    _triton_cache_size(kernel) for kernel in kernels
                )
                launch_all(batch_block)
                torch.cuda.synchronize()
                cache_after_warmup = tuple(
                    _triton_cache_size(kernel) for kernel in kernels
                )
                self.assertEqual(
                    cache_after_warmup,
                    tuple(count + 1 for count in cache_before_warmup),
                )

                production_batch = batch_block - 1
                self.assertEqual(
                    _cp_metadata_triton._batch_block(production_batch),
                    batch_block,
                )
                launch_all(production_batch)
                torch.cuda.synchronize()
                self.assertEqual(
                    tuple(_triton_cache_size(kernel) for kernel in kernels),
                    cache_after_warmup,
                )


if __name__ == "__main__":
    unittest.main()
