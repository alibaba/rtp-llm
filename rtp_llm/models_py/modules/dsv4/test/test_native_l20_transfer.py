"""Native-shaped L20 views and maintained PP frame/HC interfaces.

Fixtures have real opaque storage and native metadata, but do not instantiate
the C++ allocator or the model's pipeline execution loop.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.hc.pp_state import imported_delayed_pre_mix
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
    native_l20_cache_views,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_pp_delayed_state import fixture
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Contract,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_pp import (
    V41PPFrame,
    import_v41_pp_frame,
    pack_v41_pp_frame,
)

TAGS = ("hca_kv", "indexer_kv")
WIRE_ROWS = ("v41_l20_global_delta", "v41_l20_index_delta")
TEXT = json.loads(
    (
        Path(__file__).parents[4] / "config/test/testdata/deepseek_v41_flash.json"
    ).read_text()
)["text_config"]
CONTRACT = V41L20Contract("a" * 64, "b" * 64, "c" * 64, 512)


def native_fixture(rank, device="cpu", *, seeded=False):
    layouts = dsv41_page_layouts(0)
    sizes = {tag: 6 * (layouts[tag].stride_bytes + 64) for tag in TAGS}
    if seeded:
        backing = torch.randint(
            0,
            256,
            (sum(sizes.values()),),
            dtype=torch.uint8,
            device=device,
            generator=torch.Generator(device=device).manual_seed(41020),
        )
    else:
        backing = torch.full(
            (sum(sizes.values()),), 165, device=device, dtype=torch.uint8
        )
    layers, offset = {}, 0
    local = 20 if rank == 0 else 0
    for tag in TAGS:
        width = layouts[tag].stride_bytes
        raw = backing[offset : offset + sizes[tag]].view(6, width + 64)[:, :width]
        offset += sizes[tag]
        layers[local, tag] = SimpleNamespace(
            tag=tag,
            layer_id=local,
            seq_size_per_block=256,
            kv_cache_base=raw,
            kv_scale_base=None,
            read_replica_source_layer_id=None if rank == 0 else 20,
        )
    native = SimpleNamespace(
        layer_count=21 if rank == 0 else 19,
        get_layer_cache=lambda layer, tag: layers[layer, tag],
        get_seq_size_per_block=lambda tag: 256,
        get_kernel_seq_size_per_block=lambda tag: 256,
        layers=layers,
    )
    return native, backing


def frame_for(rows):
    count = rows.shape[0]
    topk = torch.full((count, 512), -1, dtype=torch.int32)
    candidates = torch.full((count, 2048), -1, dtype=torch.int32)
    counts = torch.zeros(count, dtype=torch.int32)
    for row, (_, position, valid) in enumerate(rows.tolist()):
        if valid:
            topk[row, : position + 1] = torch.arange(position + 1)
            counts[row] = position // 8 + 1
            candidates[row, : counts[row]] = torch.arange(position // 8 + 1)
    return V41PPFrame(
        torch.randn(count, 4, 5120).bfloat16(),
        torch.rand(count, 4),
        rows.clone(),
        topk,
        candidates,
        counts,
    )


class NativeL20TransferTest(unittest.TestCase):
    def test_views_preserve_native_storage_and_reject_wrong_stage_metadata(self):
        for rank in (0, 1):
            native, backing = native_fixture(rank)
            views = native_l20_cache_views(TEXT, native, rank)
            for tag, view in views.items():
                self.assertEqual(
                    view.raw.untyped_storage().data_ptr(), backing.data_ptr()
                )
                self.assertGreater(view.raw.stride(0), view.layout.stride_bytes)
                self.assertEqual(
                    view.raw.data_ptr(),
                    native.get_layer_cache(
                        20 if rank == 0 else 0, tag
                    ).kv_cache_base.data_ptr(),
                )
            native.layer_count += 1
            with self.assertRaisesRegex(ValueError, "layer count"):
                native_l20_cache_views(TEXT, native, rank)
        for rank in (-1, 2, True):
            with self.assertRaises(ValueError):
                native_l20_cache_views(TEXT, native_fixture(0)[0], rank)

    def test_geometry_replica_identity_and_overlap_fail_before_writes(self):
        changes = (
            ("tag", "csa_kv"),
            ("layer_id", 1),
            ("seq_size_per_block", 128),
            ("read_replica_source_layer_id", None),
            ("read_replica_source_layer_id", 19),
            ("kv_scale_base", torch.zeros(1)),
            ("kv_cache_base", torch.zeros(2, 73728)),
            ("kv_cache_base", torch.empty(0, 73728, dtype=torch.uint8)),
        )
        for name, value in changes:
            native, backing = native_fixture(1)
            with self.subTest(name=name):
                setattr(native.layers[0, "hca_kv"], name, value)
                with self.assertRaises(ValueError):
                    NativeL20ReplicaReceiver(TEXT, native, CONTRACT, 1, backend="eager")
                self.assertTrue(bool((backing == 165).all()))
        for method in ("get_seq_size_per_block", "get_kernel_seq_size_per_block"):
            native, _ = native_fixture(1)
            setattr(native, method, lambda tag: 128)
            with self.assertRaisesRegex(ValueError, "256/256"):
                native_l20_cache_views(TEXT, native, 1)
        native, _ = native_fixture(1)
        native.layers[0, "indexer_kv"].kv_cache_base = native.layers[
            0, "hca_kv"
        ].kv_cache_base[:, :17408]
        with self.assertRaisesRegex(ValueError, "overlap"):
            native_l20_cache_views(TEXT, native, 1)
        native, _ = native_fixture(0)
        native.layers[20, "hca_kv"].read_replica_source_layer_id = 20
        with self.assertRaises(ValueError):
            native_l20_cache_views(TEXT, native, 0)

    def test_reserved_page_zero_is_rejected_on_export_admit_and_extend(self):
        identity = V41CacheIdentity(41, 1, 0)
        source, _ = native_fixture(0, seeded=True)
        target, backing = native_fixture(1)
        receiver = NativeL20ReplicaReceiver(TEXT, target, CONTRACT, 1, backend="eager")
        good = {tag: (1,) for tag in TAGS}
        for page in (0, -1, True):
            bad = {"hca_kv": (page,), "indexer_kv": (1,)}
            with self.assertRaisesRegex(ValueError, "reserved page zero"):
                receiver.begin_request(identity, bad)
            with self.assertRaisesRegex(ValueError, "reserved page zero"):
                export_native_l20_delta(
                    TEXT,
                    source,
                    CONTRACT,
                    1,
                    0,
                    [V41L20Span(identity, 0, 0, 1)],
                    {identity: bad},
                    backend="eager",
                )
        receiver.begin_request(identity, good)
        with self.assertRaises(ValueError):
            receiver.extend_request(identity, {tag: (1, 0) for tag in TAGS})
        self.assertEqual(receiver.block_tables(identity), good)
        self.assertTrue(bool((backing == 165).all()))

    def verify_roundtrip(self, device, backend):
        source, source_backing = native_fixture(0, device, seeded=True)
        target, backing = native_fixture(1, device)
        reference, expected = native_fixture(1)
        receiver = NativeL20ReplicaReceiver(TEXT, target, CONTRACT, 1, backend=backend)
        first, second = V41CacheIdentity(41, 1, 2), V41CacheIdentity(42, 2, 3)
        producer = {
            first: {"hca_kv": (3, 1), "indexer_kv": (1, 3)},
            second: {"hca_kv": (2,), "indexer_kv": (4,)},
        }
        replica = {
            first: {"hca_kv": (1, 4), "indexer_kv": (3, 4)},
            second: {"hca_kv": (2,), "indexer_kv": (2,)},
        }
        for identity in (first, second):
            receiver.begin_request(identity, replica[identity])
        source_snapshot = source_backing.clone()
        for chunk, start, count, other_start, other_count in (
            (0, 0, 255, 0, 2),
            (1, 255, 2, 2, 1),
        ):
            spans = [
                V41L20Span(first, chunk, start, count),
                V41L20Span(second, chunk, other_start, other_count),
            ]
            message = export_native_l20_delta(
                TEXT, source, CONTRACT, 1, chunk, spans, producer, backend=backend
            )
            snapshot = backing.clone()
            bad = dict(message)
            bad[WIRE_ROWS[1]] = message[WIRE_ROWS[1]][:, :-1].contiguous()
            with self.assertRaises(ValueError):
                receiver.apply(bad, spans)
            torch.testing.assert_close(backing, snapshot, rtol=0, atol=0)
            receiver.validate(message, spans)
            torch.testing.assert_close(backing, snapshot, rtol=0, atol=0)
            for tag, key in zip(TAGS, WIRE_ROWS):
                layout = dsv41_page_layouts(0)[tag]
                source_raw = source.layers[20, tag].kv_cache_base.cpu()
                target_raw = reference.layers[0, tag].kv_cache_base
                rows = []
                for span in spans:
                    for position in range(span.start, span.start + span.count):
                        entry = position % 256
                        src_page = producer[span.identity][tag][position // 256]
                        dst_page = replica[span.identity][tag][position // 256]
                        payload = slice(
                            entry * layout.payload_bytes_per_entry,
                            (entry + 1) * layout.payload_bytes_per_entry,
                        )
                        scale = slice(
                            layout.scale_plane_offset
                            + entry * layout.scale_bytes_per_entry,
                            layout.scale_plane_offset
                            + (entry + 1) * layout.scale_bytes_per_entry,
                        )
                        rows.append(
                            torch.cat(
                                (
                                    source_raw[src_page, payload],
                                    source_raw[src_page, scale],
                                )
                            )
                        )
                        target_raw[dst_page, payload] = source_raw[src_page, payload]
                        target_raw[dst_page, scale] = source_raw[src_page, scale]
                torch.testing.assert_close(
                    message[key].cpu(), torch.stack(rows), rtol=0, atol=0
                )
            receiver.apply(message, spans)
            for span in spans:
                receiver.require_ready(span.identity, span.start + span.count, chunk)
            torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                receiver.apply(message, spans)
            torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(source_backing, source_snapshot, rtol=0, atol=0)
        for identity in (first, second):
            receiver.release(identity)

    def test_relocated_page_boundary_preserves_reserved_pages_and_storage_gaps(self):
        self.verify_roundtrip("cpu", "eager")

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "SM120 native L20 bridge requires its own GPU lease",
    )
    def test_sm120_relocated_page_boundary_with_actual_transfer_kernels(self):
        self.verify_roundtrip("cuda", "triton")

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "SM120 native L20 capture fence requires its own GPU lease",
    )
    def test_sm120_receiver_constructed_eager_cannot_mutate_during_capture(self):
        source, _ = native_fixture(0, "cuda", seeded=True)
        target, backing = native_fixture(1, "cuda")
        identity = V41CacheIdentity(41, 1, 0)
        newer = V41CacheIdentity(42, 2, 0)
        blocks = {tag: (1,) for tag in TAGS}
        span = V41L20Span(identity, 0, 0, 1)
        receiver = NativeL20ReplicaReceiver(TEXT, target, CONTRACT, 1)
        receiver.begin_request(identity, blocks)
        message = export_native_l20_delta(
            TEXT, source, CONTRACT, 1, 0, [span], {identity: blocks}
        )
        operations = (
            lambda: native_l20_cache_views(TEXT, target, 1),
            lambda: export_native_l20_delta(
                TEXT, source, CONTRACT, 1, 0, [span], {identity: blocks}
            ),
            lambda: receiver.validate(message, [span]),
            lambda: receiver.apply(message, [span]),
            lambda: receiver.begin_request(newer, {tag: (2,) for tag in TAGS}),
            lambda: receiver.extend_request(identity, {tag: (1, 2) for tag in TAGS}),
            lambda: receiver.cancel(identity),
            lambda: receiver.release(identity),
        )
        scratch = torch.zeros(1, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scratch.add_(1)
            for operation in operations:
                with self.assertRaisesRegex(NotImplementedError, "eager execution"):
                    operation()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(bool((backing == 165).all()))
        self.assertEqual(receiver.block_tables(identity), blocks)
        receiver.apply(message, [span])
        receiver.require_ready(identity, 1, 0)
        receiver.release(identity)

    def test_empty_round_cancellation_and_recycled_identity_fence_stale_data(self):
        source, _ = native_fixture(0, seeded=True)
        target, backing = native_fixture(1)
        receiver = NativeL20ReplicaReceiver(TEXT, target, CONTRACT, 1, backend="eager")
        blocks = {tag: (1,) for tag in TAGS}
        old = V41CacheIdentity(41, 5, 0)
        receiver.begin_request(old, blocks)
        empty = export_native_l20_delta(
            TEXT, source, CONTRACT, 1, 0, [], {}, backend="eager"
        )
        receiver.apply(empty, [])
        self.assertTrue(bool((backing == 165).all()))
        receiver.release(old)
        current = V41CacheIdentity(41, 6, 1)
        receiver.begin_request(current, blocks)
        stale_span = V41L20Span(old, 0, 0, 1)
        stale = export_native_l20_delta(
            TEXT, source, CONTRACT, 1, 1, [stale_span], {old: blocks}, backend="eager"
        )
        with self.assertRaisesRegex(ValueError, "stale replica identity"):
            receiver.apply(stale, [stale_span])
        receiver.cancel(current)
        span = V41L20Span(current, 0, 0, 1)
        message = export_native_l20_delta(
            TEXT, source, CONTRACT, 1, 1, [span], {current: blocks}, backend="eager"
        )
        with self.assertRaises(ValueError):
            receiver.apply(message, [span])
        self.assertTrue(bool((backing == 165).all()))
        receiver.release(current)
        newer = V41CacheIdentity(41, 7, 2)
        receiver.begin_request(newer, blocks)
        span = V41L20Span(newer, 0, 0, 1)
        message = export_native_l20_delta(
            TEXT, source, CONTRACT, 1, 1, [span], {newer: blocks}, backend="eager"
        )
        receiver.apply(message, [span])
        receiver.require_ready(newer, 1, 0)
        receiver.release(newer)

    def test_complete_frame_import_precedes_actual_delayed_hc_stage_readout(self):
        source, _ = native_fixture(0, seeded=True)
        target, backing = native_fixture(1)
        identity = V41CacheIdentity(41, 1, 0)
        spans = [V41L20Span(identity, 0, 0, 4)]
        rows = torch.tensor([[0, 1, 1], [-1, -1, 0], [0, 3, 1]])
        frame = frame_for(rows)
        delta = export_native_l20_delta(
            TEXT,
            source,
            CONTRACT,
            1,
            0,
            spans,
            {identity: {tag: (1,) for tag in TAGS}},
            backend="eager",
        )
        message = pack_v41_pp_frame(delta, frame, spans, rows)
        receiver = NativeL20ReplicaReceiver(TEXT, target, CONTRACT, 1, backend="eager")
        receiver.begin_request(identity, {tag: (2,) for tag in TAGS})
        bad = dict(message)
        bad["v41_row_map"] = rows.flip(0).contiguous()
        with self.assertRaisesRegex(ValueError, "row map"):
            import_v41_pp_frame(bad, receiver, spans, rows)
        self.assertTrue(bool((backing == 165).all()))
        imported = import_v41_pp_frame(message, receiver, spans, rows)
        unit, _, _ = fixture(dim=5120, rows=3)
        _, expected_post, expected_comb = unit.pre(imported.hidden_states)
        current = unit.pre_mix_out.clone()
        expected = (
            (imported.hidden_states.float() * imported.prev_pre_mix.unsqueeze(-1))
            .sum(1)
            .bfloat16()
        )
        with imported_delayed_pre_mix(
            unit, imported.hidden_states, imported.prev_pre_mix
        ):
            imported.prev_pre_mix.zero_()
            actual, post, comb = unit.pre(imported.hidden_states)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(post, expected_post, rtol=0, atol=0)
            torch.testing.assert_close(comb, expected_comb, rtol=0, atol=0)
            torch.testing.assert_close(unit.pre_mix_out, current, rtol=0, atol=0)
        self.assertIsNone(unit._previous_ref)
        receiver.release(identity)


if __name__ == "__main__":
    unittest.main()
