"""Exact shared metadata contracts; parent owns all GPU execution."""

import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.fp8.decode._v41_sm90_prepare_meta import (
    Sm90DecodeMetadataUpdater,
)
from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
    allocate_decode_metadata_fp8,
    update_decode_metadata_in_place_fp8,
)
from rtp_llm.models_py.modules.dsv41.fp8.decode import decode_fmha_impl as fmha_module
from rtp_llm.models_py.modules.dsv41.kv_cache_utils import TaggedAttentionInputs

SPECS = {
    "swa_kv": (128, 1024, 5),
    "decoder_swa_kv": (256, 2048, 3),
    "global_kv_2": (64, 128, 33),
    "global_kv_1": (128, 128, 33),
    "indexer_kv": (128, 128, 33),
    "indexer_state": (2, 1024, 2),
}
ENTRIES = {tag: spec[0] for tag, spec in SPECS.items()}
TOKENS = {tag: spec[1] for tag, spec in SPECS.items()}


def _metadata(bucket, q):
    return allocate_decode_metadata_fp8(
        max_batch_size=bucket,
        q_len=q,
        window_size=128,
        head_dim=512,
        max_seq_len=4096,
        compress_ratios=[0],
        index_topk=512,
        device=torch.device("cuda", torch.cuda.current_device()),
        paged_pool_specs=SPECS,
    )


def _tables(rows, *, narrow=False, strided=False):
    result = {}
    for i, (tag, (_, _, cols)) in enumerate(SPECS.items()):
        cols = max(1, cols - 1) if narrow else cols
        t = torch.arange(rows * cols, device="cuda", dtype=torch.int32).reshape(
            rows, cols
        )
        t = t + 1 + i * 1000
        t[:, ::3] = 0
        if strided:
            storage = torch.full(
                (rows, cols * 2 + 3), -777, device="cuda", dtype=torch.int32
            )
            storage[:, 1 : cols * 2 + 1 : 2].copy_(t)
            t = storage[:, 1 : cols * 2 + 1 : 2]
        result[tag] = t
    return result


def _tensors(meta):
    result = {}
    for name, value in vars(meta).items():
        if isinstance(value, torch.Tensor):
            result[name] = value
        elif isinstance(value, dict):
            for key, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    result[f"{name}.{key}"] = tensor
    child = meta.decoder_swa_metadata
    if child is not None:
        result["decoder.swa_global_slots"] = child.swa_global_slots
    return result


def _legacy(meta, inputs, tables, full=False):
    update_decode_metadata_in_place_fp8(
        meta,
        inputs,
        forbid_realloc=True,
        paged_block_tables=tables,
        paged_pool_entries_per_block=ENTRIES,
        paged_pool_tokens_per_block=TOKENS,
        capture_full_width_lengths=full,
    )


class Sm90PrepareMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("Dedicated target requires a parent-locked SM90 GPU")

    def _equal(self, actual, expected):
        for name, value in _tensors(actual).items():
            torch.testing.assert_close(
                value, _tensors(expected)[name], rtol=0, atol=0, msg=name
            )
        self.assertEqual(actual.batch_size, expected.batch_size)
        self.assertEqual(actual.total_tokens, expected.total_tokens)

    def _independent_positions(self, meta, raw, full=False):
        # CPU integer oracle is independent of both Triton implementations.
        starts = raw.cpu().to(torch.int32).clamp(0, 4096 - meta.q_len_per_req)
        bs = starts.numel()
        positions = starts[:, None] + torch.arange(
            meta.q_len_per_req, dtype=torch.int32
        )
        torch.testing.assert_close(meta.start_pos[:bs].cpu(), starts, rtol=0, atol=0)
        torch.testing.assert_close(
            meta.position_ids[: positions.numel()].cpu(),
            positions.flatten(),
            rtol=0,
            atol=0,
        )
        lengths = (
            torch.full_like(starts, 128)
            if full
            else (starts + meta.q_len_per_req).clamp(max=128)
        )
        torch.testing.assert_close(
            meta.swa_topk_length[:bs].cpu(), lengths, rtol=0, atol=0
        )
        for tag, read in (
            ("swa_kv", meta.swa_global_slots),
            ("decoder_swa_kv", meta.decoder_swa_metadata.swa_global_slots),
        ):
            table = meta.pool_block_tables[tag].cpu()
            expected = torch.full((positions.numel(), 128), -1, dtype=torch.int32)
            for r in range(bs):
                for q in range(meta.q_len_per_req):
                    pos = int(positions[r, q])
                    for w in range(128):
                        token = max(0, pos - 127) + w
                        column = token // TOKENS[tag]
                        if token <= pos and column < table.shape[1]:
                            bid = int(table[r, column])
                            if bid > 0:
                                expected[r * meta.q_len_per_req + q, w] = (
                                    bid * ENTRIES[tag] + token % ENTRIES[tag]
                                )
            torch.testing.assert_close(
                read[: positions.numel()].cpu(), expected, rtol=0, atol=0
            )

    def test_exact_fields_bucket_tails_and_dynamic_tables(self):
        for bucket in (1, 4, 17):
            for q in (1, 3):
                with self.subTest(bucket=bucket, q=q):
                    actual, reference = _metadata(bucket, q), _metadata(bucket, q)
                    updater = Sm90DecodeMetadataUpdater(actual, ENTRIES, TOKENS)
                    self.assertTrue(updater.enabled)
                    pointers = {k: t.data_ptr() for k, t in _tensors(actual).items()}
                    for step, bs in enumerate((bucket, 1, bucket)):
                        raw = torch.tensor(
                            [
                                (-11, 0, 1, 127, 128, 1023, 2047, 4095, 9999)[
                                    (r + step * 3) % 9
                                ]
                                for r in range(bs)
                            ],
                            device="cuda",
                            dtype=torch.int32,
                        )
                        tables = _tables(bs, narrow=step == 1, strided=step == 2)
                        full = step == 0
                        self.assertTrue(
                            updater.update(
                                raw,
                                paged_block_tables=tables,
                                capture_full_width_lengths=full,
                            )
                        )
                        _legacy(reference, raw, tables, full)
                        self._equal(actual, reference)
                        self._independent_positions(actual, raw, full)
                        self.assertEqual(
                            pointers,
                            {k: t.data_ptr() for k, t in _tensors(actual).items()},
                        )

    def test_device_mirrors_cpu_ownership_and_target_verify(self):
        for q, target, prefill in (
            (1, False, False),
            (3, True, False),
            (3, False, True),
        ):
            with self.subTest(q=q, target=target, prefill=prefill):
                meta = _metadata(4, q)
                updater = Sm90DecodeMetadataUpdater(meta, ENTRIES, TOKENS)
                cpu = torch.tensor([0, 127, 2047, 4095], dtype=torch.int32).pin_memory()
                original = cpu.clone()
                dev = cpu.cuda()
                inputs = SimpleNamespace(
                    is_target_verify=target,
                    is_prefill=prefill,
                    sequence_lengths=cpu,
                    sequence_lengths_plus_1_device=dev + 1,
                    prefix_lengths=cpu,
                    prefix_lengths_device=dev,
                )
                self.assertTrue(updater.update(inputs, paged_block_tables=_tables(4)))
                self._independent_positions(meta, original)
                torch.testing.assert_close(cpu, original, rtol=0, atol=0)
                # Bare CPU path owns a synchronous staging copy, so immediate
                # reuse of pinned storage cannot change pending kernel inputs.
                self.assertTrue(updater.update(cpu, paged_block_tables=_tables(4)))
                cpu.fill_(3)
                self._independent_positions(meta, original)

    def test_graph_changed_lengths_and_mappings(self):
        for q in (1, 3):
            with self.subTest(q=q):
                actual, reference = _metadata(4, q), _metadata(4, q)
                updater = Sm90DecodeMetadataUpdater(actual, ENTRIES, TOKENS)
                source = torch.zeros(4, device="cuda", dtype=torch.int32)
                tables = _tables(4)
                for _ in range(3):
                    self.assertTrue(updater.update(source, paged_block_tables=tables))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.assertTrue(updater.update(source, paged_block_tables=tables))
                for step in range(4):
                    source.copy_(
                        torch.tensor(
                            [0, 126 + step, 2046 + step, 4095],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    for table in tables.values():
                        table.add_(step)
                    graph.replay()
                    _legacy(reference, source, tables)
                    self._equal(actual, reference)
                    self._independent_positions(actual, source)

    def test_unsupported_is_no_write(self):
        meta = _metadata(4, 1)
        updater = Sm90DecodeMetadataUpdater(meta, ENTRIES, TOKENS)
        before = {k: t.clone() for k, t in _tensors(meta).items()}
        tables = _tables(4)
        cases = (
            (torch.ones(5, device="cuda", dtype=torch.int32), tables),
            (torch.ones(4, device="cuda"), tables),
            (torch.ones(4, device="cuda", dtype=torch.int32), {}),
            (
                torch.ones(4, device="cuda", dtype=torch.int32),
                {k: v for k, v in tables.items() if k != "swa_kv"},
            ),
            (torch.ones(4, device="cuda", dtype=torch.int32), meta.pool_block_tables),
        )
        for source, mapping in cases:
            self.assertFalse(updater.update(source, paged_block_tables=mapping))
            for k, t in _tensors(meta).items():
                torch.testing.assert_close(t, before[k], rtol=0, atol=0)
        # Never intercept legacy compressed-ratio metadata (4/128 or 1/2).
        for ratio in (1, 2, 4, 128):
            meta.compressed_lens[ratio] = torch.ones(
                4, device="cuda", dtype=torch.int32
            )
            unsupported = Sm90DecodeMetadataUpdater(meta, ENTRIES, TOKENS)
            self.assertFalse(unsupported.enabled)
            self.assertFalse(
                unsupported.update(
                    torch.ones(4, device="cuda", dtype=torch.int32),
                    paged_block_tables=tables,
                )
            )
            meta.compressed_lens.clear()

        for tag in (
            "swa_kv",
            "decoder_swa_kv",
            "global_kv_2",
            "global_kv_1",
            "indexer_kv",
        ):
            with self.subTest(missing_write_destination=tag):
                destination = meta.pool_write_slot_mappings.pop(tag)
                unsupported = Sm90DecodeMetadataUpdater(meta, ENTRIES, TOKENS)
                self.assertFalse(unsupported.enabled)
                self.assertFalse(
                    unsupported.update(
                        torch.ones(4, device="cuda", dtype=torch.int32),
                        paged_block_tables=tables,
                    )
                )
                meta.pool_write_slot_mappings[tag] = destination
                for name, tensor in _tensors(meta).items():
                    torch.testing.assert_close(
                        tensor, before[name], rtol=0, atol=0, msg=name
                    )

    def test_public_fmha_hook_tagged_inputs_and_graph_consumer(self):
        for q in (1, 3):
            with self.subTest(q=q):
                bucket = 4

                def tagged(raw, tables):
                    host = raw.to(dtype=torch.int32, device="cpu").pin_memory()
                    device = host.cuda()
                    return TaggedAttentionInputs(
                        {
                            tag: SimpleNamespace(
                                is_target_verify=q > 1,
                                is_prefill=False,
                                sequence_lengths=host,
                                sequence_lengths_plus_1_device=device + 1,
                                prefix_lengths=host,
                                prefix_lengths_device=device,
                                kv_cache_kernel_block_id_device=table,
                            )
                            for tag, table in tables.items()
                        }
                    )

                tables = _tables(bucket)
                inputs = tagged(torch.tensor([0, 1, 126, 2047]), tables)
                config = fmha_module.DSv4DecodeFmhaImplConfigFP8(
                    max_batch_size=bucket,
                    q_len=q,
                    window_size=128,
                    head_dim=512,
                    max_seq_len=4096,
                    compress_ratios=[0],
                    index_topk=512,
                    paged_pool_specs=SPECS,
                    group_tags=list(SPECS),
                )
                # Initialization must retain the legacy full-width capture
                # preparation. Only replay preparation is patched below.
                impl = fmha_module.DSv4DecodeFmhaImplFP8(
                    config,
                    torch.device("cuda", torch.cuda.current_device()),
                    inputs,
                )
                self.assertIsNotNone(impl.metadata.decoder_swa_metadata)
                self.assertIsNotNone(impl._sm90_metadata_updater)
                self.assertTrue(impl._sm90_metadata_updater.enabled)
                reference = _metadata(bucket, q)
                _legacy(reference, inputs, tables, full=True)
                self._equal(impl.metadata, reference)
                self.assertTrue(bool((impl.metadata.swa_topk_length == 128).all()))
                persistent = _tensors(impl.metadata)
                pointers = {name: t.data_ptr() for name, t in persistent.items()}

                # Capture a consumer of every persistent tensor, not the
                # prepare hook itself: production prepares outside its graph.
                consumed = {name: torch.empty_like(t) for name, t in persistent.items()}

                def consume():
                    for name, tensor in persistent.items():
                        consumed[name].copy_(tensor)

                consume()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    consume()

                with mock.patch.object(
                    fmha_module,
                    "update_decode_metadata_in_place_fp8",
                    side_effect=AssertionError(
                        "supported replay unexpectedly used legacy metadata"
                    ),
                ) as forbidden_legacy:
                    for step, bs in enumerate((bucket, 1, bucket)):
                        raw = torch.tensor([0, 127, 2048, 4095][:bs], dtype=torch.int32)
                        raw += step
                        # New source addresses, column geometry and contents:
                        # graph must consume owned snapshots, never old sources.
                        tables = _tables(bs, narrow=step == 1, strided=step == 2)
                        for table in tables.values():
                            table.add_(step)
                        inputs = tagged(raw, tables)
                        host_before = inputs.sequence_lengths.clone()
                        impl.prepare_cuda_graph(inputs if step != 1 else dict(inputs))
                        _legacy(reference, inputs, tables)
                        graph.replay()
                        self._equal(impl.metadata, reference)
                        self._independent_positions(impl.metadata, raw)
                        for name, tensor in consumed.items():
                            torch.testing.assert_close(
                                tensor,
                                _tensors(reference)[name],
                                rtol=0,
                                atol=0,
                                msg=name,
                            )
                        torch.testing.assert_close(
                            inputs.sequence_lengths, host_before, rtol=0, atol=0
                        )
                        self.assertEqual(
                            pointers,
                            {
                                name: t.data_ptr()
                                for name, t in _tensors(impl.metadata).items()
                            },
                        )
                    forbidden_legacy.assert_not_called()

                # A missing typed source is unsupported by the fused helper,
                # but valid for the original partial-snapshot path. Verify
                # actual fallback execution, not just a mocked return value.
                tables = {
                    tag: table
                    for tag, table in tables.items()
                    if tag != "indexer_state"
                }
                inputs = tagged(torch.tensor([7, 129, 2050, 4090]), tables)
                with mock.patch.object(
                    fmha_module,
                    "update_decode_metadata_in_place_fp8",
                    wraps=update_decode_metadata_in_place_fp8,
                ) as fallback:
                    impl.prepare_cuda_graph(inputs)
                    fallback.assert_called_once()
                _legacy(reference, inputs, tables)
                graph.replay()
                self._equal(impl.metadata, reference)
                for name, tensor in consumed.items():
                    torch.testing.assert_close(
                        tensor, _tensors(reference)[name], rtol=0, atol=0, msg=name
                    )

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_META_BENCH") == "1", "opt-in timing"
    )
    def test_complete_prepare_latency(self):
        for bs in (1, 8):
            actual, reference = _metadata(bs, 1), _metadata(bs, 1)
            updater = Sm90DecodeMetadataUpdater(actual, ENTRIES, TOKENS)
            cpu = torch.full((bs,), 2047, dtype=torch.int32).pin_memory()
            device = cpu.cuda()
            inputs = SimpleNamespace(
                sequence_lengths=cpu, sequence_lengths_plus_1_device=device + 1
            )
            tables = _tables(bs)
            calls = {
                "legacy": lambda: _legacy(reference, inputs, tables),
                "candidate": lambda: updater.update(inputs, paged_block_tables=tables),
            }
            for call in calls.values():
                for _ in range(5):
                    call()
            self._equal(actual, reference)
            results = {}
            for name, call in calls.items():
                wall = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(30):
                        call()
                    torch.cuda.synchronize()
                    wall.append((time.perf_counter() - start) * 1e6 / 30)
                # Isolate the COMPLETE GPU preparation chain, not one fused
                # leaf. Graph capture needs device lengths for the legacy arm.
                gpu_call = (
                    (lambda: _legacy(reference, device, tables))
                    if name == "legacy"
                    else (lambda: updater.update(device, paged_block_tables=tables))
                )
                for _ in range(3):
                    gpu_call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(10):
                        gpu_call()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                begin.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                results[name] = dict(
                    host_end_to_end_us=wall,
                    complete_gpu_graph_us=begin.elapsed_time(end) * 1000 / 200,
                )
            self._equal(actual, reference)
            print(
                "SM90_META_BENCH "
                + json.dumps(
                    dict(
                        bs=bs,
                        results=results,
                        scope="all shared metadata preparation; no profiler; excludes C++ prepareInputs",
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
