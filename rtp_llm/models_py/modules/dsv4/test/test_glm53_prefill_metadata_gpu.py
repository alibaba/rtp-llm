"""GPU regression for CP metadata, packed KDA consumers and batched indexer."""

import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4._cp_metadata_triton import build_fused_cp_context
from rtp_llm.models_py.modules.dsv4.cp import (
    build_cp_context,
    build_cp_full_prefill_positions,
)
from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA


def cp_info(lengths, world):
    chunks = [((n + 2 * world - 1) // (2 * world)) * 2 for n in lengths]
    total = sum(chunks)
    pad = torch.zeros(total * world, dtype=torch.int32)
    restore = torch.empty_like(pad)
    padded = local = 0
    for n, chunk in zip(lengths, chunks):
        pad[padded : padded + n] = 1
        half = chunk // 2
        for rank in range(world):
            positions = list(range(rank * half, (rank + 1) * half))
            positions += list(
                range(chunk * world - (rank + 1) * half, chunk * world - rank * half)
            )
            for t, pos in enumerate(positions):
                restore[padded + pos] = rank * total + local + t
        local += chunk
        padded += chunk * world
    return (
        SimpleNamespace(
            prefill_qkv_padding_mask=pad,
            prefill_qkv_restore_indice=restore,
            prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
            prefill_cp_chunk_lengths=torch.tensor(chunks, dtype=torch.int32),
        ),
        total,
    )


def test_cp():
    count = 0
    for world in (1, 2, 8):
        for lengths in (
            [1],
            [0, 0],
            [0, 1, 15, 16, 17],
            [17, 0, 131, 65],
            [13312] * 20,
        ):
            info, total = cp_info(lengths, world)
            prefix = torch.tensor(
                [0 if i % 2 else 117760 + i * 128 for i in range(len(lengths))]
            )
            for rank in range(world):
                reference = build_cp_context(
                    info, world, rank, total, torch.device("cpu"), prefix, True
                )
                candidate = build_fused_cp_context(
                    info, world, rank, total, torch.device("cuda"), prefix.cuda(), True
                )
                for name in (
                    "relative_positions",
                    "global_positions",
                    "req_id_per_token",
                    "local_is_real",
                    "unpad_restore",
                    "prefix_lengths",
                    "input_lengths_global",
                    "cu_seqlens_global",
                ):
                    torch.testing.assert_close(
                        getattr(candidate, name).cpu(),
                        getattr(reference, name),
                        rtol=0,
                        atol=0,
                    )
                for name in (
                    "prefix_length",
                    "seq_len_full",
                    "seq_len_total",
                    "padded_seq_len",
                ):
                    assert getattr(candidate, name) == getattr(reference, name), name
                for actual, expected in zip(
                    build_cp_full_prefill_positions(candidate, torch.device("cuda")),
                    build_cp_full_prefill_positions(reference, torch.device("cpu")),
                ):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                count += 1
    # The cache must return the identical object within a forward, then rebuild
    # for a new forward even when the prefix tensor is updated in place.
    info, total = cp_info([17, 65], 8)
    prefix = torch.tensor([128, 256], device="cuda")
    first = None
    for step in range(2):
        token = _FORWARD_METADATA.set({})
        try:
            a = build_cp_context(info, 8, 0, total, torch.device("cuda"), prefix, True)
            b = build_cp_context(info, 8, 0, total, torch.device("cuda"), prefix, True)
            assert a is b
            if first is not None:
                assert first is not a
                torch.testing.assert_close(
                    a.global_positions, first.global_positions + 128, rtol=0, atol=0
                )
            first = a
        finally:
            _FORWARD_METADATA.reset(token)
        prefix.add_(128)
    info, total = cp_info([17], 8)
    expected = build_cp_context(info, 8, 0, total, torch.device("cpu"), 128, True)
    actual = build_fused_cp_context(
        info, 8, 0, total, torch.device("cuda"), torch.tensor(128, device="cuda"), True
    )
    torch.testing.assert_close(
        actual.global_positions.cpu(), expected.global_positions, rtol=0, atol=0
    )
    print("CP_PASS", count + 1, flush=True)


def test_kda():
    from rtp_llm.models_py.distributed.glm53_collective_gemm import (
        packed_kda_projections,
    )
    from rtp_llm.models_py.triton_kernels.causal_conv1d import (
        causal_conv1d_fn,
        prepare_causal_conv1d_metadata,
    )

    torch.manual_seed(20260917)
    widths = (3072, 8, 128, 128)
    weights = [
        torch.randn(4096, n, device="cuda", dtype=torch.bfloat16) * 0.01 for n in widths
    ]
    packed_w = torch.zeros((4096, 3456), device="cuda", dtype=torch.bfloat16)
    packed_w[:, : sum(widths)].copy_(torch.cat(weights, 1))
    # No collective state is configured: the packed GEMM also covers the
    # ordinary-AG path below the communication/GEMM overlap threshold.
    for lengths in ([1], [127], [128], [129], [33, 127, 256], [1664] * 20):
        m = sum(lengths)
        x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
        expected = [x @ w for w in weights]
        actual = packed_kda_projections(x, packed_w, widths)
        errors = []
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0.02, atol=0.02)
            errors.append((got.float() - ref.float()).abs().max().item())
        qkv = actual[0]
        assert qkv.stride() == (3456, 1)
        if m > 1:
            assert not qkv.is_contiguous()
        cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
        prefix = torch.zeros(len(lengths), dtype=torch.int32, device="cuda")
        meta = prepare_causal_conv1d_metadata(cu, x.device)
        conv_w = torch.randn(3072, 4, device="cuda", dtype=torch.float32) * 0.1
        kwargs = dict(
            weight=conv_w,
            bias=None,
            conv_states=None,
            query_start_loc=cu,
            block_map=None,
            seq_size_per_block=128,
            prefix_lengths=prefix,
            metadata=meta,
            output_groups=3,
        )
        out = causal_conv1d_fn(qkv.T, **kwargs)
        reference = causal_conv1d_fn(qkv.contiguous().T, **kwargs)
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        print(
            "KDA_PASS",
            json.dumps({"m": m, "max_abs_error": errors, "stride": list(qkv.stride())}),
            flush=True,
        )


def test_mqa():
    from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

    module = importlib.import_module("rtp_llm.models_py.modules.dsv4.fp8.indexer")
    torch.manual_seed(91753)
    runner = SimpleNamespace(
        index_topk=2048,
        compress_ratio=4,
        compressor=SimpleNamespace(kpool_mode=True),
        prefill_topk_backend="legacy",
    )
    os.environ["DSV4_INDEXER_TOPK_CANONICALIZE"] = "1"
    os.environ["DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS"] = "16384"
    for batch, uneven, large in (
        (1, False, "tiny"),
        (1, False, "short"),
        (1, False, False),
        (2, False, False),
        (5, True, False),
        (20, True, False),
        (20, False, True),
    ):
        q_lens = [
            (
                1
                if large == "tiny"
                else (
                    16
                    if large == "short"
                    else (
                        1664 if large else 128 + (i % 3) * 32 + (i % 2 if uneven else 0)
                    )
                )
            )
            for i in range(batch)
        ]
        k_lens = [
            (
                1
                if large == "tiny"
                else (
                    32
                    if large == "short"
                    else (
                        32768
                        if large
                        else 4096 + (i % 3) * 128 + (i % 2 if uneven else 0)
                    )
                )
            )
            for i in range(batch)
        ]
        m, n = sum(q_lens), sum(k_lens)
        q = (torch.randn(m, 32, 128, device="cuda") * 0.3).to(torch.float8_e4m3fn)
        k = (torch.randn(n, 128, device="cuda") * 0.3).to(torch.float8_e4m3fn)
        scale = torch.rand(n, device="cuda") * 1.5 + 0.25
        w = torch.rand(m, 32, device="cuda")
        starts, ends, segments = [], [], []
        qo = ko = 0
        for qn, kn in zip(q_lens, k_lens):
            segments.append((qo, qo + qn, ko, ko + kn))
            starts += [ko] * qn
            ends += [ko + kn - qn + t for t in range(qn)]
            qo += qn
            ko += kn
        ks = torch.tensor(starts, dtype=torch.int32, device="cuda")
        ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
        meta = SimpleNamespace(
            M=m,
            T=n,
            ks=ks,
            ke=ke,
            score_segments=segments,
            score_relative_ks=torch.zeros_like(ks),
            score_relative_ke=ke - ks,
        )
        runner.compressor.kpool_mode = False
        ref = IndexerFP8._prefill_score_topk(runner, q, w, k, scale, meta)
        runner.compressor.kpool_mode = True
        token = _FORWARD_METADATA.set({})
        try:
            for budget in (512, 8192) if large else (32, 512):
                with mock.patch.object(
                    module, "score_workspace_budget", return_value=budget << 20
                ):
                    for layer in range(2):
                        actual = IndexerFP8._prefill_score_topk(
                            runner, q, w, k, scale, meta
                        )
                        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        finally:
            _FORWARD_METADATA.reset(token)
        print("MQA_PASS", batch, m, n, flush=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53PrefillMetadataGPUTest(unittest.TestCase):
    test_cp = staticmethod(test_cp)
    test_kda = staticmethod(test_kda)
    test_mqa = staticmethod(test_mqa)

    def test_mqa_empty_rows_keys_and_insufficient_budget(self):
        module = importlib.import_module("rtp_llm.models_py.modules.dsv4.fp8.indexer")
        runner = SimpleNamespace(
            index_topk=2048,
            compress_ratio=4,
            compressor=SimpleNamespace(kpool_mode=True),
            prefill_topk_backend="legacy",
        )
        q = torch.zeros(3, 32, 128, device="cuda", dtype=torch.float8_e4m3fn)
        weights = torch.ones(3, 32, device="cuda")
        key = torch.zeros(1, 128, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, device="cuda")
        meta = SimpleNamespace(M=0, T=0)
        run = module.IndexerFP8._prefill_score_topk
        self.assertEqual(
            run(runner, q[:0], weights[:0], key, scale, meta).shape, (0, 2048)
        )
        token = _FORWARD_METADATA.set({})
        try:
            meta.M = 3
            meta.score_segments = [(0, 3, 0, 0)]
            self.assertTrue(
                bool((run(runner, q, weights, key, scale, meta) == -1).all())
            )
            meta.T = 1
            meta.score_segments = [(0, 3, 0, 1)]
            meta.ks = torch.zeros(3, device="cuda", dtype=torch.int32)
            meta.ke = torch.ones_like(meta.ks)
            with mock.patch.object(module, "score_workspace_budget", return_value=1):
                with self.assertRaisesRegex(RuntimeError, "one query row"):
                    run(runner, q, weights, key, scale, meta)
        finally:
            _FORWARD_METADATA.reset(token)


if __name__ == "__main__":
    unittest.main()
