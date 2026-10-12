"""Compare host-assisted prefill geometry against the original GPU builder."""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.hybrid.indexer import _build_tp_cache_shard_context


class HostGeometryTest(unittest.TestCase):
    def test_meta_matches_gpu_reference_and_shared_context(self):
        self.assertTrue(torch.cuda.is_available(), "GPU test must not skip")
        for lengths in ((1, 7, 129), (418,) * 7 + (419,) * 32):
            cu = torch.tensor(
                [0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda"
            )
            inputs = torch.tensor(lengths, dtype=torch.int32, device="cuda")
            prefix = torch.tensor(
                [8192 if i % 3 else 0 for i in range(len(lengths))],
                dtype=torch.int32,
                device="cuda",
            )
            req = torch.repeat_interleave(
                torch.arange(len(lengths), device="cuda"), inputs.long()
            ).int()
            positions = torch.cat(
                [
                    torch.arange(n, device="cuda") + prefix[i]
                    for i, n in enumerate(lengths)
                ]
            ).int()
            table = torch.ones(len(lengths), 128, dtype=torch.int32, device="cuda")
            attention = SimpleNamespace(
                input_lengths=inputs, prefix_lengths=prefix, cu_seqlens=cu
            )
            params = SimpleNamespace(positions_d=positions, batch_indice_d=req)
            token = _FORWARD_METADATA.set({})
            try:
                first = _build_tp_cache_shard_context(attention, params, 4, 0)
                second = _build_tp_cache_shard_context(attention, params, 4, 0)
                self.assertIs(first, second)
                stub = SimpleNamespace(
                    compressor=SimpleNamespace(kpool_mode=True),
                    compress_ratio=4,
                    freqs_cis=None,
                    _cp_ctx=None,
                    _kv_block_table=table,
                    _kv_eb=16,
                )
                kwargs = dict(
                    bsz=1,
                    seqlen=sum(lengths),
                    sp_int=0,
                    device=cu.device,
                    kv_block_table=table,
                    kv_eb=16,
                    use_varlen=True,
                    batch_size=len(lengths),
                    cu_seqlens=cu,
                    input_lengths=inputs,
                    prefix_lengths=prefix,
                    position_ids=positions,
                    req_id_per_token=req,
                    has_prefix=True,
                )
                with mock.patch.dict(os.environ, {"GLM53_PREFILL_HOST_GEOMETRY": "0"}):
                    reference = IndexerFP8.prepare.__wrapped__(stub, **kwargs)
                with mock.patch.dict(os.environ, {"GLM53_PREFILL_HOST_GEOMETRY": "1"}):
                    actual = IndexerFP8.prepare.__wrapped__(stub, **kwargs)
                self.assertIsNone(reference.max_request_keys)
                self.assertEqual(
                    actual.max_request_keys,
                    max((int(prefix[i]) + n) // 4 for i, n in enumerate(lengths)),
                )
                for name in (
                    "ks",
                    "ke",
                    "positions_d",
                    "cu_kv_seqlens",
                    "cu_kv_per_token",
                ):
                    torch.testing.assert_close(
                        getattr(actual, name), getattr(reference, name), rtol=0, atol=0
                    )
                for name in ("M", "T", "end_pos", "is_fresh_prefill"):
                    self.assertEqual(getattr(actual, name), getattr(reference, name))
            finally:
                _FORWARD_METADATA.reset(token)


if __name__ == "__main__":
    unittest.main()
