"""Actual maintained L21 queries against relocated native-shaped L20 replicas.

This qualifies the eager attention handoff, without a scheduler/allocation lease
or a PP transformer loop. The reference uses the ordinary local L20 owner.
"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings
from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    make_layer,
    prepare,
)
from rtp_llm.models_py.modules.dsv4.native_l20_prefill import (
    IMPORT_KEY,
    imported_native_l20_prefill,
    prepare_native_l20_prefill_source,
)
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
)
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_prefill import (
    NativeL20PrefillTest as NativeL20PrefillTest,
)
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import (
    CONTRACT,
    TEXT,
    frame_for,
    native_fixture,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_pp import (
    V41PPFrame,
    import_v41_pp_frame,
    pack_v41_pp_frame,
)


def owners():
    shared = {}
    producer = make_layer(20, 1, shared, gen=0, dim=5120)
    consumer = make_layer(21, 1, shared, source=20, gen=0, dim=5120)
    producer.cache_layer_id = 20
    consumer.cache_layer_id = 0
    ordinary = NativeTaggedCache([producer, consumer], gen=0, pages=6)
    ordinary.layer_count = 21
    replica, backing = native_fixture(1, "cuda")
    swa = NativeTaggedCache([consumer], gen=0, pages=6)
    replica.layers[0, "swa_kv"] = swa.groups[0, "swa_kv"]
    replica.layouts = swa.layouts
    imported_shared = {
        "layers": {21: consumer},
        "cache_bindings": {
            (b.source_layer_id, b.tag): b for b in dsv41_cache_bindings(TEXT, 2, 1)
        },
    }
    identities = (V41CacheIdentity(901, 1, 0), V41CacheIdentity(902, 2, 1))
    producer_pages = {
        identities[0]: {"hca_kv": (1,), "indexer_kv": (2,)},
        identities[1]: {"hca_kv": (2,), "indexer_kv": (1,)},
    }
    replica_pages = {
        identities[0]: {"hca_kv": (3,), "indexer_kv": (4,)},
        identities[1]: {"hca_kv": (4,), "indexer_kv": (3,)},
    }
    receiver = NativeL20ReplicaReceiver(TEXT, replica, CONTRACT, 0)
    for identity in identities:
        receiver.begin_request(identity, replica_pages[identity])

    def dispatch(cache, pages):
        tables = {
            tag: torch.tensor([[1], [2]], device="cuda", dtype=torch.int32)
            for tag in cache.layouts
        }
        for tag in ("hca_kv", "indexer_kv"):
            tables[tag] = torch.tensor(
                [pages[identity][tag] for identity in identities],
                device="cuda",
                dtype=torch.int32,
            )
        return tables

    return (
        producer,
        consumer,
        shared,
        imported_shared,
        ordinary,
        replica,
        backing,
        receiver,
        identities,
        producer_pages,
        dispatch(ordinary, producer_pages),
        dispatch(replica, replica_pages),
    )


def transfer(producer, ordinary, receiver, identities, pages, counts, prefixes, chunk):
    rows = torch.tensor(
        [
            [request, position, 1]
            for request, (count, prefix) in enumerate(zip(counts, prefixes))
            for position in range(prefix, prefix + count)
        ],
        dtype=torch.int64,
    )
    control = frame_for(rows)
    frame = V41PPFrame(
        **{
            name: value if name == "row_map" else value.cuda()
            for name, value in control.__dict__.items()
        }
    )
    frame = replace(frame, topk=producer._shared_attention["topk"][20].clone())
    spans = [
        V41L20Span(identity, chunk, prefix, count)
        for identity, prefix, count in zip(identities, prefixes, counts)
    ]
    delta = export_native_l20_delta(TEXT, ordinary, CONTRACT, 0, chunk, spans, pages)
    frame = import_v41_pp_frame(
        pack_v41_pp_frame(delta, frame, spans, rows), receiver, spans, rows
    )
    return prepare_native_l20_prefill_source(frame, receiver, spans, rows), frame


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "Actual L21 attention handoff requires a leased SM120 GPU",
)
class NativeL20PrefillSM120Test(unittest.TestCase):
    @torch.no_grad()
    def test_actual_l21_matches_local_owner_across_changed_forward_and_swa_wrap(self):
        torch.manual_seed(4121)
        (
            producer,
            consumer,
            shared,
            imported_shared,
            ordinary,
            replica,
            backing,
            receiver,
            identities,
            pages,
            tables,
            imported_tables,
        ) = owners()
        prefixes = [0, 0]
        for chunk, counts in enumerate(([4, 3], [3, 4], [249, 249])):
            consumer._shared_attention = shared
            x = torch.randn(sum(counts), 5120, device="cuda", dtype=torch.bfloat16)
            query = torch.randn_like(x)
            positions, _, _ = prepare(
                producer, x, ordinary, counts, prefixes, tables=tables
            )
            producer(x, positions, ordinary, tables)
            keys = shared["global"][20]
            selection = shared["topk"][20].clone()
            positions, _, _ = prepare(
                consumer, query, ordinary, counts, prefixes, tables=tables
            )
            reference = consumer(query, positions, ordinary, tables)
            source, frame = transfer(
                producer,
                ordinary,
                receiver,
                identities,
                pages,
                counts,
                prefixes,
                chunk,
            )
            for actual, expected in zip(source.globals_by_req, keys):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(source.topk, selection, rtol=0, atol=0)
            # Completed receive buffers can be reused before the owner executes.
            frame.topk.zero_()
            frame.row_map.fill_(99)
            consumer._shared_attention = imported_shared
            before = backing.clone()
            try:
                with imported_native_l20_prefill(consumer, source):
                    positions, _, _ = prepare(
                        consumer,
                        query,
                        replica,
                        counts,
                        prefixes,
                        tables=imported_tables,
                    )
                    actual = consumer(query, positions, replica, imported_tables)
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    self.assertEqual(list(imported_shared["layers"]), [21])
                    self.assertFalse(hasattr(consumer, "global_wkv"))
                    torch.testing.assert_close(backing, before, rtol=0, atol=0)
                    consumer._set_prefill_meta_shared(None)
                    self.assertEqual(imported_shared["global"], {})
                    self.assertEqual(imported_shared["topk"], {})
            finally:
                consumer._set_prefill_meta_shared(None)
                consumer._shared_attention = shared
                producer._set_prefill_meta_shared(None)
            self.assertNotIn(IMPORT_KEY, imported_shared)
            torch.testing.assert_close(
                replica.layers[0, "swa_kv"].kv_cache_base,
                ordinary.groups[0, "swa_kv"].kv_cache_base,
                rtol=0,
                atol=0,
            )
            prefixes = [p + n for p, n in zip(prefixes, counts)]

    @torch.no_grad()
    def test_actual_entry_rejects_reordered_rows_and_cleans_up_query_failure(self):
        (
            producer,
            consumer,
            shared,
            imported_shared,
            ordinary,
            replica,
            backing,
            receiver,
            identities,
            pages,
            tables,
            imported_tables,
        ) = owners()
        counts, prefixes = [4, 3], [0, 0]
        x = torch.randn(7, 5120, device="cuda", dtype=torch.bfloat16)
        positions, _, _ = prepare(producer, x, ordinary, counts, tables=tables)
        producer(x, positions, ordinary, tables)
        source, _ = transfer(
            producer, ordinary, receiver, identities, pages, counts, prefixes, 0
        )
        consumer._shared_attention = imported_shared
        try:
            for invalid_rows in (True, False):
                before = backing.clone()
                swa_before = replica.layers[0, "swa_kv"].kv_cache_base.clone()
                with imported_native_l20_prefill(consumer, source):
                    positions, _, meta = prepare(
                        consumer, x, replica, counts, tables=imported_tables
                    )
                    if invalid_rows:
                        consumer._set_prefill_meta_shared(
                            meta._replace(position_ids=meta.position_ids.flip(0))
                        )
                    with patch.object(
                        consumer,
                        "_prefill_compute_qkv",
                        side_effect=RuntimeError("injected query failure"),
                    ) as compute:
                        with self.assertRaisesRegex(
                            ValueError if invalid_rows else RuntimeError,
                            "dispatch disagrees" if invalid_rows else "query failure",
                        ):
                            consumer(x, positions, replica, imported_tables)
                        self.assertEqual(compute.call_count, 0 if invalid_rows else 1)
                    consumer._set_prefill_meta_shared(None)
                    self.assertEqual(imported_shared["global"], {})
                    self.assertEqual(imported_shared["topk"], {})
                self.assertNotIn(IMPORT_KEY, imported_shared)
                torch.testing.assert_close(backing, before, rtol=0, atol=0)
                torch.testing.assert_close(
                    replica.layers[0, "swa_kv"].kv_cache_base,
                    swa_before,
                    rtol=0,
                    atol=0,
                )
            # A missing imported scope must still be rejected by the normal gate.
            with bind_attn_cache(consumer, replica, imported_tables):
                with self.assertRaisesRegex(NotImplementedError, "imported PP source"):
                    consumer._require_qualified_prefill(
                        torch.tensor(counts, device="cuda", dtype=torch.int32),
                        torch.tensor(prefixes, device="cuda", dtype=torch.int32),
                        cp_on=False,
                    )
        finally:
            consumer._set_prefill_meta_shared(None)
            consumer._shared_attention = shared
            producer._set_prefill_meta_shared(None)


if __name__ == "__main__":
    unittest.main()
