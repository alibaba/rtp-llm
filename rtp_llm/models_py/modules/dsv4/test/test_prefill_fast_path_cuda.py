import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_MAIN,
    CPContext,
    combine_topk_swa_indices_cp_varlen,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
    CompressorFP8,
)
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.prefill import forward as prefill_forward
from rtp_llm.models_py.modules.dsv4.test.test_prefill_fast_path import (
    _FakeLayer,
    _FakeV4,
    _PrefillForwardTestBase,
    _RealMetaAttention,
)
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


class _PagedPrefillCache:
    """Python-owned GPU pools exposing native framework LayerKVCache objects."""

    def __init__(self, device, ratio):
        self.layers = {}
        self.block_tables = {}
        self.specs = {}
        for tag, entries, tokens_per_block, columns in (
            (SWA_KV, 4, 2048, 1),
            (CSA_KV if ratio == 4 else HCA_KV, 16, 16 * max(ratio, 1), 8),
        ):
            base = torch.arange((2 * columns + 1) * entries * 8, device=device)
            base = base.to(torch.uint8).reshape(2 * columns + 1, entries * 8)
            self.layers[tag] = LayerKVCache(
                base, tokens_per_block, 0, len(self.layers), tag
            )
            self.block_tables[tag] = torch.arange(
                1, 2 * columns + 1, dtype=torch.int32, device=device
            ).reshape(2, columns)
            self.specs[tag] = (torch.bfloat16, 4)

    def get_layer_cache(self, layer_id, tag):
        assert layer_id == 0
        return self.layers[tag]

    def get_seq_size_per_block(self, tag):
        return self.layers[tag].seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag):
        return self.get_seq_size_per_block(tag)


class _MetadataLayer(_FakeLayer):
    def forward_prefill_fast(self, h, input_ids, positions, cu_seqlens, **_kwargs):
        offsets = torch.arange(input_ids.numel(), device=input_ids.device)
        requests = torch.searchsorted(
            cu_seqlens[1:].to(device=input_ids.device), offsets, right=True
        )
        return h + (positions + requests * 10).reshape(-1, 1, 1)


class _CsaPrefillCache(_PagedPrefillCache):
    """Native KV and state pools for the complete ratio-4 metadata chain."""

    def __init__(self, device):
        self.layers = {}
        self.block_tables = {}
        self.specs = {}
        for tag, dtype, width, entries, tokens_per_block, columns in (
            (SWA_KV, torch.uint8, KV_ENTRY_BYTES, 4, 2048, 1),
            (CSA_KV, torch.uint8, KV_ENTRY_BYTES, 16, 64, 8),
            (INDEXER_KV, torch.uint8, INDEXER_ENTRY_BYTES, 16, 64, 8),
            (CSA_STATE, torch.float32, 4 * KV_HEAD_DIM, 256, 2048, 1),
            (INDEXER_STATE, torch.float32, 4 * INDEXER_HEAD_DIM, 256, 2048, 1),
        ):
            base = torch.zeros(
                (2 * columns + 1, entries * width * dtype.itemsize),
                dtype=torch.uint8,
                device=device,
            )
            if tag == INDEXER_KV:
                # Each block stores all FP8 K rows followed by FP32 scales.
                base[:, : entries * INDEXER_HEAD_DIM] = torch.arange(
                    base.shape[0], device=device
                ).to(torch.uint8)[:, None]
                base[:, entries * INDEXER_HEAD_DIM :].view(torch.float32).fill_(1)
            self.layers[tag] = LayerKVCache(
                base, tokens_per_block, 0, len(self.layers), tag
            )
            self.block_tables[tag] = torch.arange(
                1, 2 * columns + 1, dtype=torch.int32, device=device
            ).reshape(2, columns)
            self.specs[tag] = (dtype, width)


def _compressor_weights(head_dim, device):
    return {
        "ape": torch.zeros((4, 2 * head_dim), device=device),
        "wkv": torch.zeros((2 * head_dim, 128), dtype=torch.bfloat16, device=device),
        "wgate": torch.zeros((2 * head_dim, 128), dtype=torch.bfloat16, device=device),
        "norm": torch.ones(head_dim, dtype=torch.bfloat16, device=device),
    }


class PrefillFastPathCudaTest(_PrefillForwardTestBase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated GPU target")
        if torch.cuda.get_device_capability()[0] < 10:
            raise AssertionError("SM100 or newer is required by this GPU target")

    def test_prefill_cu_seqlens_warmup_is_cuda_graph_safe(self):
        device = torch.device("cuda", torch.cuda.current_device())
        existing = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device=device)
        self.assertIs(
            prefill_forward._resolve_prefill_cu_seqlens(existing, None, device),
            existing,
        )

        placeholder = torch.empty(0, dtype=torch.int32, device=device)
        input_lengths = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
        prefill_forward._resolve_prefill_cu_seqlens(placeholder, input_lengths, device)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            resolved = prefill_forward._resolve_prefill_cu_seqlens(
                placeholder, input_lengths, device
            )
        self.assertEqual(resolved.device, device)
        self.assertEqual(resolved.dtype, torch.int32)
        self.assertTrue(resolved.is_contiguous())
        for lengths in ([1, 3, 1], [0, 2, 3]):
            with self.subTest(lengths=lengths):
                input_lengths.copy_(torch.tensor(lengths, device=device))
                resolved.fill_(-1)
                graph.replay()
                torch.cuda.synchronize()
                expected = [0]
                for length in lengths:
                    expected.append(expected[-1] + length)
                torch.testing.assert_close(
                    resolved,
                    torch.tensor(expected, dtype=torch.int32, device=device),
                )

    def test_forward_prefill_moves_cuda_cu_seqlens_to_host(self):
        attn = SimpleNamespace(
            cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
            cu_seqlens_device=None,
            combo_position_ids=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            input_lengths=None,
            input_lengths_device=None,
        )

        forwarded = self._forwarded_cu_seqlens(self._run_forward_prefill_with(attn))
        self.assertEqual(forwarded.device.type, "cpu")
        torch.testing.assert_close(
            forwarded, torch.tensor([0, 2, 4], dtype=torch.int32)
        )

    def test_full_forward_prefill_is_cuda_graph_capture_safe(self):
        for host_mirror_present in (False, True):
            with self.subTest(host_mirror_present=host_mirror_present):
                self._check_forward_prefill_graph(host_mirror_present)

    def _check_forward_prefill_graph(self, host_mirror_present):
        device = torch.device("cuda", torch.cuda.current_device())
        v4 = _FakeV4()
        v4.layers = [_MetadataLayer(0, v4.calls)]
        input_ids = torch.tensor([3, 4, 5, 6], dtype=torch.long, device=device)
        attn = SimpleNamespace(
            cu_seqlens=torch.tensor(
                [0, 2, 4] if host_mirror_present else [], dtype=torch.int32
            ),
            cu_seqlens_device=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
            input_lengths=torch.tensor([2, 2], dtype=torch.int32),
            input_lengths_device=torch.tensor([2, 2], dtype=torch.int32, device=device),
            prefix_lengths=torch.tensor([5, 100], dtype=torch.int32),
            prefix_lengths_device=torch.tensor(
                [5, 100], dtype=torch.int32, device=device
            ),
            combo_position_ids=torch.empty(0, dtype=torch.long, device=device),
        )
        inputs = SimpleNamespace(attention_inputs=attn, input_ids=input_ids)

        patches = (
            patch.object(prefill_forward, "set_cp_info"),
            patch.object(
                prefill_forward, "primary_attention_inputs", return_value=attn
            ),
            patch.object(
                prefill_forward, "build_block_tables_batched", return_value={}
            ),
            patch.object(
                prefill_forward,
                "synchronized_moe_chunk_plan",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    prefill_forward.forward_prefill(v4, None, None, inputs)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                graph_output = prefill_forward.forward_prefill(
                    v4, None, None, inputs
                ).hidden_states

            input_ids.copy_(torch.tensor([9, 8, 7, 6], device=device))
            for lengths in ([0, 4], [3, 1], [4, 0]):
                with self.subTest(lengths=lengths):
                    # The host mirrors deliberately retain the capture-time
                    # batch. Replay updates only the existing device tensors.
                    attn.cu_seqlens_device.copy_(
                        torch.tensor([0, lengths[0], 4], device=device)
                    )
                    attn.input_lengths_device.copy_(
                        torch.tensor(lengths, device=device)
                    )
                    expected_offsets = [
                        prefix + offset + request * 10
                        for request, (prefix, length) in enumerate(
                            zip([5, 100], lengths)
                        )
                        for offset in range(length)
                    ]
                    expected = (
                        v4.embed(input_ids)
                        + 100
                        + torch.tensor(expected_offsets, device=device).unsqueeze(1)
                    )
                    with torch.inference_mode():
                        graph_output.fill_(float("nan"))
                        graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(graph_output, expected)
            torch.testing.assert_close(
                attn.input_lengths, torch.tensor([2, 2], dtype=torch.int32)
            )

    def test_real_attention_meta_builder_is_cuda_graph_capture_safe(self):
        device = torch.device("cuda", torch.cuda.current_device())
        attn = _RealMetaAttention.__new__(_RealMetaAttention)
        nn.Module.__init__(attn)
        attn.freqs_cis = torch.arange(32, device=device).reshape(16, 2)
        x = torch.empty((2, 4), dtype=torch.bfloat16, device=device)
        cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        input_lengths = torch.tensor([1, 1], dtype=torch.int32, device=device)
        prefix_lengths = torch.tensor([5, 0], dtype=torch.int32, device=device)
        position_ids = torch.tensor([0, 1], dtype=torch.long, device=device)
        req_id_per_token = torch.tensor([0, 1], dtype=torch.int32, device=device)
        sp_per_req = torch.tensor([5, 0], dtype=torch.long, device=device)
        topk_idxs = torch.zeros((2, 1), dtype=torch.long, device=device)
        topk_length = torch.ones((2,), dtype=torch.int32, device=device)

        def build_meta():
            return attn._build_shared_prefill_meta(
                x,
                0,
                sp_per_req=sp_per_req,
                cu_seqlens=cu_seqlens,
                batch_size=2,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=1,
                any_cont=True,
            )

        with patch.object(
            _swa_ops_triton,
            "compute_window_topk_and_length_varlen",
            return_value=(topk_idxs, topk_length),
        ), _profiler.disable_record_function_ranges():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    build_meta()
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                meta = build_meta()

            position_ids.copy_(torch.tensor([2, 3], device=device))
            graph.replay()
            torch.cuda.synchronize()

        self.assertTrue(meta.any_cont)
        torch.testing.assert_close(
            meta.freqs_cis, attn.freqs_cis.index_select(0, position_ids)
        )

    def test_cp_continuation_paged_swa_capture_and_replay(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for cp_size in (2, 4):
            for rank in (0, cp_size - 1):
                with self.subTest(cp_size=cp_size, rank=rank):
                    cache = _PagedPrefillCache(device, 0)
                    attn = AttentionFP8.__new__(AttentionFP8)
                    nn.Module.__init__(attn)
                    attn.layer_id = 0
                    attn.compress_ratio = 0
                    attn.window_size = 4
                    attn._kv_cache = cache
                    attn._block_tables_by_type = cache.block_tables
                    attn._pool_spec = cache.specs
                    global_length = 4 * cp_size
                    offsets = [
                        rank * 2,
                        rank * 2 + 1,
                        global_length - (rank + 1) * 2,
                        global_length - (rank + 1) * 2 + 1,
                    ]
                    local_offsets = torch.tensor(offsets * 2, device=device)
                    requests = torch.tensor([0] * 4 + [1] * 4, device=device)
                    prefixes = torch.tensor([1, 0], dtype=torch.int32, device=device)
                    positions = local_offsets + prefixes[requests]
                    lengths = torch.tensor([4, 4], dtype=torch.int32, device=device)
                    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
                    attn._cp_ctx = CPContext(
                        cp_size=cp_size,
                        cp_rank=rank,
                        chunk_length=8,
                        padded_seq_len=2 * global_length,
                        seq_len_full=2 * global_length,
                        relative_positions=local_offsets + requests * global_length,
                        prefix_length=1,
                        global_positions=positions,
                        local_is_real=torch.ones(8, dtype=torch.bool, device=device),
                        unpad_restore=torch.arange(2 * global_length, device=device),
                        seq_len_total=global_length + 1,
                        cp_info=None,
                        req_id_per_token=requests,
                        prefix_lengths=prefixes,
                        input_lengths_global=lengths * cp_size,
                        cu_seqlens_global=cu * cp_size,
                    )

                    def build():
                        meta = attn._build_swa_prefill_meta_varlen(
                            seqlen=8,
                            device=device,
                            any_cont=True,
                            batch_size=2,
                            cu_seqlens=cu,
                            input_lengths=lengths,
                            prefix_lengths=prefixes,
                            position_ids=positions,
                            req_id_per_token=requests,
                        )
                        slots = meta.cache_slot_mapping
                        pool = cache.layers[SWA_KV].kv_cache_base.reshape(-1, 8)
                        gathered = torch.where(
                            (slots >= 0).unsqueeze(-1), pool[slots.clamp_min(0)], 0
                        )
                        return meta, gathered

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream), torch.inference_mode():
                        for _ in range(3):
                            build()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph), torch.inference_mode():
                        captured, captured_kv = build()

                    for new_prefixes in ([129, 260], [0, 2], [0, 0]):
                        prefixes.copy_(torch.tensor(new_prefixes, device=device))
                        positions.copy_(local_offsets + prefixes[requests])
                        graph.replay()
                        torch.cuda.synchronize()
                        eager, eager_kv = build()
                        torch.testing.assert_close(
                            captured.slot_mapping, eager.slot_mapping
                        )
                        width = eager.cache_slot_mapping.shape[1]
                        torch.testing.assert_close(
                            captured.cache_slot_mapping[:, :width],
                            eager.cache_slot_mapping,
                        )
                        torch.testing.assert_close(captured_kv[:, :width], eager_kv)
                        self.assertTrue(torch.all(captured_kv[:, width:] == 0))
                        indices, lens = combine_topk_swa_indices_cp_varlen(
                            topk_indices=torch.empty(
                                (8, 0), dtype=torch.int32, device=device
                            ),
                            global_positions=positions,
                            sp_int=0,
                            window_size=4,
                            compress_ratio=1,
                            topk=0,
                            M=captured.M,
                            N=0,
                            req_id_per_token=requests,
                            prefix_lengths=prefixes,
                        )
                        torch.testing.assert_close(captured.combined_indices, indices)
                        torch.testing.assert_close(captured.combined_lens, lens)

    def test_paged_swa_csa_hca_metadata_capture_and_replay(self):
        # Do not override either metadata builder or mock its Triton kernels.
        # Exercise real device block tables/pools, including prefixes that grow
        # after capture and empty requests in a fixed-size token batch.
        device = torch.device("cuda", torch.cuda.current_device())
        for ratio in (0, 4, 128):
            with self.subTest(ratio=ratio):
                cache = _PagedPrefillCache(device, ratio)
                attn = AttentionFP8.__new__(AttentionFP8)
                nn.Module.__init__(attn)
                attn.layer_id = 0
                attn.compress_ratio = ratio
                attn.window_size = 4
                attn._cp_ctx = None
                attn._kv_cache = cache
                attn._block_tables_by_type = cache.block_tables
                attn._pool_spec = cache.specs
                lengths = torch.tensor([4, 4], dtype=torch.int32, device=device)
                prefixes = torch.tensor([1, 0], dtype=torch.int32, device=device)
                cu = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
                positions = torch.tensor([1, 2, 3, 4, 0, 1, 2, 3], device=device)
                requests = torch.tensor(
                    [0] * 4 + [1] * 4, dtype=torch.int32, device=device
                )

                def build():
                    common = dict(
                        seqlen=8,
                        device=device,
                        batch_size=2,
                        cu_seqlens=cu,
                        input_lengths=lengths,
                        prefix_lengths=prefixes,
                        position_ids=positions,
                        req_id_per_token=requests,
                    )
                    swa = attn._build_swa_prefill_meta_varlen(any_cont=True, **common)
                    workspace = (
                        attn._build_workspace_meta(
                            sp_int=0,
                            with_dense_cmp_topk=ratio == 128,
                            use_varlen=True,
                            **common,
                        )
                        if ratio
                        else None
                    )
                    slots = (
                        workspace.swa_cache_slot_mapping
                        if workspace
                        else swa.cache_slot_mapping
                    )
                    pool = cache.layers[SWA_KV].kv_cache_base.reshape(-1, 8)
                    gathered = pool[slots.clamp_min(0)]
                    gathered = torch.where((slots >= 0).unsqueeze(-1), gathered, 0)
                    return swa, workspace, slots, gathered

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), torch.inference_mode():
                    for _ in range(3):
                        build()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.inference_mode():
                    captured_swa, captured_workspace, captured_slots, captured_kv = (
                        build()
                    )
                for lens, prefs in (
                    ([3, 5], [129, 260]),
                    ([0, 8], [0, 2]),
                    ([8, 0], [130, 0]),
                    ([4, 4], [0, 0]),
                ):
                    with self.subTest(lengths=lens, prefixes=prefs):
                        lengths.copy_(torch.tensor(lens, device=device))
                        prefixes.copy_(torch.tensor(prefs, device=device))
                        cu.copy_(torch.tensor([0, lens[0], 8], device=device))
                        requests.copy_(
                            torch.tensor([0] * lens[0] + [1] * lens[1], device=device)
                        )
                        positions.copy_(
                            torch.tensor(
                                [
                                    prefix + offset
                                    for length, prefix in zip(lens, prefs)
                                    for offset in range(length)
                                ],
                                device=device,
                            )
                        )
                        graph.replay()
                        torch.cuda.synchronize()
                        eager_swa, eager_workspace, eager_slots, eager_kv = build()
                        torch.testing.assert_close(
                            captured_swa.slot_mapping, eager_swa.slot_mapping
                        )
                        width = eager_slots.shape[1]
                        torch.testing.assert_close(
                            captured_slots[:, :width], eager_slots
                        )
                        self.assertTrue(torch.all(captured_slots[:, width:] == -1))
                        torch.testing.assert_close(captured_kv[:, :width], eager_kv)
                        self.assertTrue(torch.all(captured_kv[:, width:] == 0))

                        meta = captured_workspace or captured_swa
                        base = captured_workspace.N if captured_workspace else 0
                        expected_slots = [
                            request * meta.M + base + min(prefix, 3) + offset
                            for request, (length, prefix) in enumerate(zip(lens, prefs))
                            for offset in range(length)
                        ]
                        slots = (
                            meta.new_k_slot_in_flat
                            if captured_workspace
                            else meta.slot_in_flat
                        )
                        torch.testing.assert_close(
                            slots,
                            torch.tensor(expected_slots, device=device),
                            check_dtype=False,
                        )
                        if captured_workspace:
                            torch.testing.assert_close(
                                captured_workspace.cmp_seq_lens,
                                eager_workspace.cmp_seq_lens,
                            )

    def test_cp_sharded_indexer_metadata_rejects_cuda_graph_capture(self):
        device = torch.device("cuda", torch.cuda.current_device())
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                indexer = IndexerFP8.__new__(IndexerFP8)
                nn.Module.__init__(indexer)
                indexer.compress_ratio = 4
                indexer.freqs_cis = torch.ones(
                    (32, 32), dtype=torch.complex64, device=device
                )
                lengths = torch.full((batch_size,), 2, dtype=torch.int32, device=device)
                prefixes = torch.zeros_like(lengths)
                cu = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * 2
                positions = torch.arange(2, device=device).repeat(batch_size)
                requests = torch.arange(
                    batch_size, device=device, dtype=torch.int32
                ).repeat_interleave(2)
                indexer._cp_ctx = SimpleNamespace(
                    cp_size=2,
                    cp_rank=0,
                    kv_cache_sharded=True,
                    input_lengths_global=lengths * 2,
                )
                block_table = torch.ones(
                    (batch_size, 1), dtype=torch.int32, device=device
                )
                marker = torch.zeros(1, device=device)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.inference_mode():
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "CUDA Graph capture is not supported for CP-sharded indexer",
                    ):
                        indexer.prepare(
                            bsz=1,
                            seqlen=batch_size * 2,
                            sp_int=0,
                            device=device,
                            kv_block_table=block_table,
                            kv_eb=4,
                            use_varlen=True,
                            batch_size=batch_size,
                            cu_seqlens=cu,
                            input_lengths=lengths,
                            prefix_lengths=prefixes,
                            position_ids=positions,
                            req_id_per_token=requests,
                            max_seqlen_q=2,
                            has_prefix=False,
                        )
                    marker.add_(1)
                # Reject before a host sync or invalid pool read poisons capture.
                marker.zero_()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(marker, torch.ones_like(marker))

    def test_full_csa_metadata_and_indexer_gather_capture_and_replay(self):
        device = torch.device("cuda", torch.cuda.current_device())
        cache = _CsaPrefillCache(device)
        attn = AttentionFP8.__new__(AttentionFP8)
        nn.Module.__init__(attn)
        attn.layer_id = 0
        attn.compress_ratio = 4
        attn.rope_head_dim = 64
        attn.window_size = 4
        attn._cp_ctx = None
        attn._kv_cache = cache
        attn._block_tables_by_type = cache.block_tables
        attn._pool_spec = cache.specs
        attn.freqs_cis = torch.ones((512, 32), dtype=torch.complex64, device=device)
        attn.compressor = CompressorFP8(
            dim=128,
            head_dim=KV_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            max_batch_size=2,
            cp_role=_CP_ROLE_MAIN,
            compressor_weights=_compressor_weights(KV_HEAD_DIM, device),
        )
        inner_weights = _compressor_weights(INDEXER_HEAD_DIM, device)
        attn.indexer = IndexerFP8(
            dim=128,
            q_lora_rank=128,
            index_n_heads=1,
            index_head_dim=INDEXER_HEAD_DIM,
            rope_head_dim=64,
            index_topk=4,
            compress_ratio=4,
            max_batch_size=2,
            max_seq_len=512,
            layer_weights={
                W.v4_indexer_wq_b_w: torch.zeros(
                    (128, 128), dtype=torch.uint8, device=device
                ).view(torch.float8_e4m3fn),
                W.v4_indexer_wq_b_s: torch.ones((1, 1), device=device),
                W.v4_indexer_weights_proj_w: torch.zeros(
                    (1, 128), dtype=torch.bfloat16, device=device
                ),
                W.v4_indexer_compressor_ape: inner_weights["ape"],
                W.v4_indexer_compressor_wkv: inner_weights["wkv"],
                W.v4_indexer_compressor_wgate: inner_weights["wgate"],
                W.v4_indexer_compressor_norm: inner_weights["norm"],
            },
        )
        x = torch.empty((4, 128), dtype=torch.bfloat16, device=device)
        lengths = torch.tensor([2, 2], dtype=torch.int32, device=device)
        prefixes = torch.tensor([1, 0], dtype=torch.int32, device=device)
        cu = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
        positions = torch.tensor([1, 2, 0, 1], device=device)
        requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        def build_and_gather():
            # No builder, compressor, pool accessor, or kernel is mocked.
            meta = attn._build_shared_prefill_meta(
                x,
                0,
                sp_per_req=prefixes,
                cu_seqlens=cu,
                batch_size=2,
                input_lengths=lengths,
                prefix_lengths=prefixes,
                position_ids=positions,
                req_id_per_token=requests,
                max_seqlen_q=4,
                any_cont=True,
            )
            indexer_meta = meta.csa_meta.indexer_meta
            keys = torch.zeros(
                (indexer_meta.T, INDEXER_HEAD_DIM), dtype=torch.uint8, device=device
            ).view(torch.float8_e4m3fn)
            scales = torch.zeros((indexer_meta.T, 4), dtype=torch.uint8, device=device)
            if indexer_meta.T:
                attn._set_compressor_pool_context()
                try:
                    attn.indexer._gather_prefill_k_cache(indexer_meta, keys, scales)
                finally:
                    attn._clear_compressor_pool_context()
            return meta.csa_meta, keys.view(torch.uint8), scales

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                build_and_gather()
            # Warm the gather kernel as well (the initial batch has no K rows).
            prefixes.fill_(4)
            build_and_gather()
            prefixes.copy_(torch.tensor([1, 0], device=device))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            captured, captured_keys, captured_scales = build_and_gather()
        self.assertEqual(captured.indexer_meta.T, 2 * 512 // 4)

        for lens, prefs in (
            ([3, 1], [129, 260]),
            ([0, 4], [0, 2]),
            ([4, 0], [130, 0]),
            ([2, 2], [0, 0]),
        ):
            with self.subTest(lengths=lens, prefixes=prefs):
                lengths.copy_(torch.tensor(lens, device=device))
                prefixes.copy_(torch.tensor(prefs, device=device))
                cu.copy_(torch.tensor([0, lens[0], 4], device=device))
                requests.copy_(
                    torch.tensor([0] * lens[0] + [1] * lens[1], device=device)
                )
                positions.copy_(
                    torch.tensor(
                        [
                            prefix + offset
                            for length, prefix in zip(lens, prefs)
                            for offset in range(length)
                        ],
                        device=device,
                    )
                )
                graph.replay()
                torch.cuda.synchronize()
                eager, eager_keys, eager_scales = build_and_gather()
                for name in (
                    "cu_kv_seqlens",
                    "ks",
                    "ke",
                    "positions_d",
                    "freqs_cis_slice",
                ):
                    torch.testing.assert_close(
                        getattr(captured.indexer_meta, name),
                        getattr(eager.indexer_meta, name),
                    )
                for captured_cmp, eager_cmp in (
                    (captured.compressor_meta, eager.compressor_meta),
                    (
                        captured.indexer_meta.compressor_meta,
                        eager.indexer_meta.compressor_meta,
                    ),
                ):
                    self.assertIsNotNone(captured_cmp.state_slots)
                    self.assertIsNotNone(captured_cmp.kv_slots)
                    for name in (
                        "positions",
                        "b_idx",
                        "state_slots",
                        "kv_slots",
                        "token_to_req",
                    ):
                        torch.testing.assert_close(
                            getattr(captured_cmp, name), getattr(eager_cmp, name)
                        )
                live = eager.indexer_meta.T
                torch.testing.assert_close(captured_keys[:live], eager_keys)
                torch.testing.assert_close(captured_scales[:live], eager_scales)
                self.assertTrue(torch.all(captured_keys[live:] == 0))
                self.assertTrue(torch.all(captured_scales[live:] == 0))


if __name__ == "__main__":
    unittest.main()
