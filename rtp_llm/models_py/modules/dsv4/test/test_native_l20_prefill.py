"""Short-forward publication contracts before maintained owner integration."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.native_l20_prefill import (
    IMPORT_KEY,
    imported_native_l20_prefill,
    prepare_native_l20_prefill_source,
)
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
    native_l20_cache_views,
)
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import (
    CONTRACT,
    TAGS,
    TEXT,
    frame_for,
    native_fixture,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import logical_to_physical_slots
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_pp import (
    V41PPFrame,
    import_v41_pp_frame,
    pack_v41_pp_frame,
)


def setup(device="cpu"):
    backend = "eager" if device == "cpu" else "triton"
    native, _ = native_fixture(0, device)
    replica, backing = native_fixture(1, device)
    identities = (V41CacheIdentity(310, 1, 0), V41CacheIdentity(311, 2, 1))
    spans = [V41L20Span(identity, 0, 0, n) for identity, n in zip(identities, (4, 3))]
    producer = {
        identities[0]: {"hca_kv": (1,), "indexer_kv": (2,)},
        identities[1]: {"hca_kv": (2,), "indexer_kv": (1,)},
    }
    destination = {
        identities[0]: {"hca_kv": (3,), "indexer_kv": (4,)},
        identities[1]: {"hca_kv": (4,), "indexer_kv": (3,)},
    }
    caches = native_l20_cache_views(TEXT, native, 0)
    expected = []
    for index, span in enumerate(spans):
        pair = []
        for tag, dim, dtype in (
            ("hca_kv", 512, torch.bfloat16),
            ("indexer_kv", 128, torch.float32),
        ):
            values = (
                (
                    torch.arange(span.count * dim, device=device).reshape(
                        span.count, dim
                    )
                    + index * 17
                )
                % 71
                / 19
                - 1
            ).bfloat16()
            logical = torch.arange(span.count, device=device)
            slots = logical_to_physical_slots(
                torch.tensor(producer[span.identity][tag], device=device),
                logical,
                tag,
                1,
            )
            caches[tag].write(values, slots, backend=backend)
            pair.append(caches[tag].read(slots, out_dtype=dtype, backend=backend))
        expected.append(tuple(pair))
    rows = torch.tensor(
        [[i, p, 1] for i, span in enumerate(spans) for p in range(span.count)]
    )
    frame = frame_for(rows)
    if device != "cpu":
        frame = V41PPFrame(
            **{
                name: value if name == "row_map" else value.to(device)
                for name, value in frame.__dict__.items()
            }
        )
    delta = export_native_l20_delta(
        TEXT, native, CONTRACT, 1, 0, spans, producer, backend=backend
    )
    receiver = NativeL20ReplicaReceiver(TEXT, replica, CONTRACT, 1, backend=backend)
    for span in spans:
        receiver.begin_request(span.identity, destination[span.identity])
    imported = import_v41_pp_frame(
        pack_v41_pp_frame(delta, frame, spans, rows), receiver, spans, rows
    )
    attention = SimpleNamespace(
        layer_id=21,
        kv_source_layer_id=20,
        index_source_layer_id=20,
        is_kv_source=False,
        is_index_source=False,
        tp_size=1,
        _kv_cache=replica,
        _block_tables_by_type={
            tag: torch.tensor(
                [destination[s.identity][tag] for s in spans],
                device=device,
                dtype=torch.int32,
            )
            for tag in TAGS
        },
    )
    attention._shared_attention = {
        "layers": {21: attention},
        "global": {},
        "topk": {},
        "candidates": None,
    }
    common = SimpleNamespace(
        cp_on=False,
        input_lengths=torch.tensor([4, 3], device=device),
        prefix_lengths=torch.tensor([0, 0], device=device),
        position_ids=rows[:, 1].to(device),
        req_id_per_token=rows[:, 0].to(device),
    )
    return attention, common, receiver, imported, rows, spans, expected, backing


class NativeL20PrefillTest(unittest.TestCase):
    def test_relocated_tags_publish_preserved_values_without_a_fake_source_owner(self):
        attention, common, receiver, frame, rows, spans, expected, backing = setup()
        source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
        prior = backing.clone()
        with imported_native_l20_prefill(attention, source):
            source.publish(attention, common)
            shared = attention._shared_attention
            self.assertEqual(list(shared["layers"]), [21])
            self.assertEqual(list(shared["global"]), [20])
            for actual, reference in zip(shared["global"][20], expected):
                for a, b in zip(actual, reference):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(shared["topk"][20], frame.topk, rtol=0, atol=0)
            self.assertIsNone(shared["candidates"])
        self.assertNotIn(IMPORT_KEY, shared)
        self.assertNotIn(20, shared["global"])
        self.assertNotIn(20, shared["topk"])
        torch.testing.assert_close(backing, prior, rtol=0, atol=0)

    def test_completed_receive_buffer_reuse_preserves_current_selection_and_rows(self):
        attention, common, receiver, frame, rows, spans, _, _ = setup()
        source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
        before = frame.topk.clone()
        frame.topk.zero_()
        frame.row_map.fill_(99)
        with imported_native_l20_prefill(attention, source):
            source.publish(attention, common)
            torch.testing.assert_close(
                attention._shared_attention["topk"][20], before, rtol=0, atol=0
            )

    def test_reset_and_current_dispatch_validation_precede_shared_mutation(self):
        for name, value in (
            ("input_lengths", [3, 4]),
            ("prefix_lengths", [1, 0]),
            ("position_ids", [3, 2, 1, 0, 0, 1, 2]),
            ("req_id_per_token", [1, 1, 1, 1, 0, 0, 0]),
        ):
            attention, common, receiver, frame, rows, spans, _, _ = setup()
            source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
            setattr(common, name, torch.tensor(value))
            with imported_native_l20_prefill(attention, source):
                with self.assertRaisesRegex(ValueError, "dispatch disagrees"):
                    source.publish(attention, common)
                self.assertEqual(attention._shared_attention["global"], {})
                self.assertEqual(attention._shared_attention["topk"], {})
        attention, common, receiver, frame, rows, spans, _, _ = setup()
        source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
        attention._shared_attention["global"] = {20: None}
        with imported_native_l20_prefill(attention, source):
            with self.assertRaisesRegex(ValueError, "forward reset"):
                source.publish(attention, common)
            self.assertEqual(attention._shared_attention["global"], {20: None})

    def test_scope_clears_after_failure_and_rejects_nested_bindings(self):
        attention, common, receiver, frame, rows, spans, _, _ = setup()
        source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
        with self.assertRaisesRegex(RuntimeError, "owner failure"):
            with imported_native_l20_prefill(attention, source):
                with self.assertRaisesRegex(ValueError, "already has"):
                    with imported_native_l20_prefill(attention, source):
                        pass
                self.assertIs(attention._shared_attention[IMPORT_KEY], source)
                source.publish(attention, common)
                raise RuntimeError("owner failure")
        self.assertNotIn(IMPORT_KEY, attention._shared_attention)
        self.assertEqual(attention._shared_attention["global"], {})
        self.assertEqual(attention._shared_attention["topk"], {})

    def test_changed_native_page_tables_reject_before_publication(self):
        for tag in TAGS:
            attention, common, receiver, frame, rows, spans, _, backing = setup()
            source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
            before = backing.clone()
            attention._block_tables_by_type[tag] = (
                attention._block_tables_by_type[tag].flip(0).contiguous()
            )
            with imported_native_l20_prefill(attention, source):
                with self.assertRaisesRegex(ValueError, "dispatch disagrees.*pages"):
                    source.publish(attention, common)
                self.assertEqual(attention._shared_attention["global"], {})
                self.assertEqual(attention._shared_attention["topk"], {})
            torch.testing.assert_close(backing, before, rtol=0, atol=0)

    def test_wrong_owner_cache_cancellation_and_recycled_admission_cannot_publish(self):
        attention, common, receiver, frame, rows, spans, _, _ = setup()
        source = prepare_native_l20_prefill_source(frame, receiver, spans, rows)
        with imported_native_l20_prefill(attention, source):
            attention._kv_cache = native_fixture(1)[0]
            with self.assertRaisesRegex(ValueError, "admitted replica"):
                source.publish(attention, common)
            attention._kv_cache = receiver._native_cache
            attention._shared_attention["layers"][20] = object()
            with self.assertRaises(NotImplementedError):
                source.publish(attention, common)
            del attention._shared_attention["layers"][20]
            receiver.cancel(spans[0].identity)
            with self.assertRaises(ValueError):
                source.publish(attention, common)
            self.assertEqual(attention._shared_attention["global"], {})
        receiver.release(spans[0].identity)
        receiver.begin_request(
            V41CacheIdentity(310, 3, 2),
            {tag: (3 if tag == "hca_kv" else 4,) for tag in TAGS},
        )
        with self.assertRaisesRegex(ValueError, "stale replica identity"):
            with imported_native_l20_prefill(attention, source):
                pass
        self.assertNotIn(IMPORT_KEY, attention._shared_attention)


if __name__ == "__main__":
    unittest.main()
