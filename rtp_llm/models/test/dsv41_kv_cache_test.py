import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from rtp_llm.models.dsv41_kv_cache import (
    dsv41_cache_binding_for_layer,
    dsv41_cache_bindings,
    dsv41_page_layouts,
)


def checkpoint_text():
    path = Path(__file__).parents[2] / "config/test/testdata/deepseek_v41_flash.json"
    return json.loads(path.read_text())["text_config"]


class V41CacheTest(unittest.TestCase):
    def test_page_geometry_and_generation_slack(self):
        expected = {
            "csa_kv": (128, 32768, 36864),
            "hca_kv": (256, 65536, 73728),
            "indexer_kv": (256, 16384, 17408),
            "csa_state": (2, 8192, 8192),
            "swa_kv": (128, 65536, 67584),
        }
        for tag, layout in dsv41_page_layouts(0).items():
            self.assertEqual(
                (layout.entries, layout.scale_plane_offset, layout.stride_bytes),
                expected[tag],
            )
        for gen in (1, 2, 3, 8):
            layouts = dsv41_page_layouts(gen)
            self.assertGreaterEqual(layouts["csa_state"].entries, 2 + gen)
            self.assertGreaterEqual(layouts["swa_kv"].entries, 128 + gen)
            self.assertEqual(layouts["swa_kv"].entries % 2, 0)
            self.assertEqual(layouts["swa_kv"].stride_bytes % 512, 0)
        self.assertEqual(dsv41_page_layouts(1)["swa_kv"].stride_bytes, 69120)
        self.assertEqual(dsv41_page_layouts(1)["csa_state"].stride_bytes, 16384)

    def test_planes_do_not_overlap_or_touch_padding(self):
        for tag, layout in dsv41_page_layouts(1).items():
            with self.subTest(tag=tag):
                storage = bytearray([255]) * (2 * layout.stride_bytes)
                for page in range(2):
                    base = page * layout.stride_bytes
                    for entry in range(layout.entries):
                        payload, scales = layout.entry_offsets(entry)
                        storage[
                            base
                            + payload : base
                            + payload
                            + layout.payload_bytes_per_entry
                        ] = (bytes([page + 1]) * layout.payload_bytes_per_entry)
                        storage[
                            base + scales : base + scales + layout.scale_bytes_per_entry
                        ] = (bytes([page + 3]) * layout.scale_bytes_per_entry)
                for page in range(2):
                    base = page * layout.stride_bytes
                    self.assertEqual(
                        storage[base : base + layout.scale_plane_offset],
                        bytes([page + 1]) * layout.scale_plane_offset,
                    )
                    self.assertEqual(
                        storage[
                            base + layout.scale_plane_offset : base + layout.used_bytes
                        ],
                        bytes([page + 3])
                        * (layout.used_bytes - layout.scale_plane_offset),
                    )
                    self.assertEqual(
                        storage[base + layout.used_bytes : base + layout.stride_bytes],
                        bytes([255]) * (layout.stride_bytes - layout.used_bytes),
                    )
                for entry in (-1, layout.entries):
                    with self.assertRaises(IndexError):
                        layout.entry_offsets(entry)

    def test_pp_producers_cover_decode_exactly_once(self):
        text = checkpoint_text()
        decode = dsv41_cache_bindings(text)
        stages = [dsv41_cache_bindings(text, 2, rank) for rank in range(2)]
        exported = [
            (b.source_layer_id, b.tag)
            for s in stages
            for b in s
            if not b.is_read_replica
        ]
        self.assertEqual(
            Counter(exported), Counter((b.source_layer_id, b.tag) for b in decode)
        )
        self.assertEqual(len(exported), len(set(exported)))
        self.assertEqual(
            Counter(b.tag for b in stages[0]),
            {"swa_kv": 21, "csa_kv": 3, "hca_kv": 1, "indexer_kv": 4, "csa_state": 3},
        )
        self.assertEqual(
            Counter(b.tag for b in stages[1]),
            {"swa_kv": 19, "hca_kv": 1, "indexer_kv": 1},
        )
        replicas = [b for b in stages[1] if b.is_read_replica]
        self.assertEqual(
            {(b.source_layer_id, b.storage_layer_id) for b in replicas}, {(20, 0)}
        )
        layouts = dsv41_page_layouts(1)
        self.assertEqual(sum(layouts[b.tag].stride_bytes for b in replicas), 91136)

    def test_consumers_resolve_kv_producer_not_selection_producer(self):
        text = checkpoint_text()
        for layer in range(21, 40):
            for tag in ("hca_kv", "indexer_kv"):
                binding = dsv41_cache_binding_for_layer(text, layer, tag, 2, 1)
                self.assertEqual(binding.source_layer_id, 20)
                self.assertEqual(binding.storage_layer_id, 0)
                self.assertTrue(binding.is_read_replica)
        self.assertEqual(
            dsv41_cache_binding_for_layer(text, 19, "indexer_kv").source_layer_id, 14
        )
        self.assertEqual(
            dsv41_cache_binding_for_layer(text, 39, "swa_kv", 2, 1).storage_layer_id, 18
        )

    def test_invalid_bindings_and_model_geometry(self):
        text = checkpoint_text()
        for layer, tag, rank in (
            (20, "hca_kv", 1),
            (21, "swa_kv", 0),
            (21, "csa_kv", 1),
            (24, "csa_state", 1),
            (0, "indexer_kv", 0),
        ):
            with self.subTest(layer=layer, tag=tag, rank=rank), self.assertRaises(
                ValueError
            ):
                dsv41_cache_binding_for_layer(text, layer, tag, 2, rank)
        for key, value in (
            ("head_dim", 576),
            ("kv_source_layer_ids", [2, 8, 14, 24]),
            ("sliding_window", 256),
        ):
            changed = copy.deepcopy(text)
            changed[key] = value
            with self.assertRaises(ValueError):
                dsv41_cache_bindings(changed)
        for size, rank in ((2, -1), (2, 2), (4, 0)):
            with self.assertRaises(ValueError):
                dsv41_cache_bindings(text, size, rank)

    def test_unsupported_page_modes(self):
        for args in ((0, 512, 256), (0, 256, 128), (0, 256, 256, True), (-1,), (True,)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                dsv41_page_layouts(*args)


if __name__ == "__main__":
    unittest.main()
