"""CP4 forward metadata keeps the fused path through the online batch range."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import _cp_metadata_triton as fused
from rtp_llm.models_py.modules.dsv41.cp import build_cp_context_for_forward


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CPMetadataHighBatchTest(unittest.TestCase):
    def test_fused_matches_fallback_with_varlen_prefix_and_padding(self):
        for batch, rank in (
            (64, 0),
            (65, 0),
            (65, 3),
            (128, 0),
            (128, 3),
            (129, 0),
        ):
            with self.subTest(batch=batch, rank=rank):
                rng = random.Random(20261107 + batch)
                lengths = [rng.randrange(49) for _ in range(batch)]
                lengths[0] = 1
                prefixes = torch.tensor(
                    [rng.randrange(1 << 20) for _ in lengths], dtype=torch.int64
                )
                chunks = [((length + 7) // 8) * 2 for length in lengths]
                padding = torch.cat(
                    [
                        torch.tensor(
                            [1] * length + [0] * (chunk * 4 - length),
                            dtype=torch.int32,
                        )
                        for length, chunk in zip(lengths, chunks)
                    ]
                )
                shuffle = torch.cat(
                    [
                        torch.tensor(
                            list(range(rank * (chunk // 2), (rank + 1) * (chunk // 2)))
                            + list(
                                range(
                                    chunk * 4 - (rank + 1) * (chunk // 2),
                                    chunk * 4 - rank * (chunk // 2),
                                )
                            ),
                            dtype=torch.int32,
                        )
                        for chunk in chunks
                    ]
                )
                device = torch.device("cuda")
                info = SimpleNamespace(
                    prefill_qkv_padding_mask=padding.to(device),
                    prefill_qkv_restore_indice=torch.arange(
                        padding.numel(), dtype=torch.int32, device=device
                    ),
                    prefill_qkv_padding_mask_cpu=padding,
                    prefill_qkv_restore_indice_cpu=torch.arange(
                        padding.numel(), dtype=torch.int32
                    ),
                    prefill_actual_input_lengths_cpu=torch.tensor(
                        lengths, dtype=torch.int32
                    ),
                    prefill_cp_chunk_lengths=torch.tensor(
                        chunks, dtype=torch.int32, device=device
                    ),
                    prefill_shuffle_indices=shuffle.to(device),
                )

                def build():
                    return build_cp_context_for_forward(
                        info,
                        4,
                        rank,
                        sum(chunks),
                        device,
                        prefix_lengths=prefixes.to(device),
                        prefix_lengths_host=prefixes,
                        chunk_lengths_device=info.prefill_cp_chunk_lengths,
                        kv_cache_sharded=True,
                    )

                dispatched = []
                original = fused.try_build_cp_forward_metadata

                def record_dispatch(*args, **kwargs):
                    result = original(*args, **kwargs)
                    dispatched.append(result)
                    return result

                with patch.object(
                    fused,
                    "try_build_cp_forward_metadata",
                    side_effect=record_dispatch,
                ):
                    actual = build()
                self.assertEqual(len(dispatched), 1)
                self.assertEqual(dispatched[0] is not None, batch <= 128)
                with patch.object(fused, "_MAX_FORWARD_BATCH", 0):
                    expected = build()
                for context in (actual, expected):
                    for name, dtype in (
                        ("prefix_lengths", torch.int64),
                        ("req_id_per_token", torch.int32),
                    ):
                        value = getattr(context, name)
                        self.assertEqual(value.dtype, dtype)
                        self.assertTrue(value.is_cuda)
                        self.assertTrue(value.is_contiguous())
                for name in (
                    "relative_positions",
                    "global_positions",
                    "req_id_per_token",
                    "local_is_real",
                    "unpad_restore",
                    "cu_seqlens_global",
                    "prefix_lengths",
                ):
                    torch.testing.assert_close(
                        getattr(actual, name), getattr(expected, name), rtol=0, atol=0
                    )
