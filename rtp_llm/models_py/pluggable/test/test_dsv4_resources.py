"""Exercise actual cache tensors against native DSv4 descriptor metadata."""

import copy
import unittest
from types import SimpleNamespace

from rtp_llm.models.dsv4.resources import (
    cache_geometry_snapshot,
    opaque_cache_layouts,
    validate_bound_cache,
)

try:
    import torch
except ImportError:
    torch = None


class CacheGeometryTest(unittest.TestCase):
    def test_engine_overrides_and_generation_are_frozen(self):
        model = SimpleNamespace(attn_config=SimpleNamespace(tokens_per_block=256))
        engine = SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                seq_size_per_block=64, kernel_seq_size_per_block=0
            ),
            sp_config=SimpleNamespace(gen_num_per_cycle=1),
        )
        first = cache_geometry_snapshot(model, engine, speculative=False)
        self.assertEqual(
            first,
            dict(
                physical_tokens_per_block=256,
                kernel_tokens_per_block=256,
                gen_num_per_cycle=0,
            ),
        )
        engine.kv_cache_config.seq_size_per_block = 512
        engine.kv_cache_config.kernel_seq_size_per_block = 256
        engine.sp_config.gen_num_per_cycle = 3
        self.assertEqual(
            cache_geometry_snapshot(model, engine, speculative=True),
            dict(
                physical_tokens_per_block=512,
                kernel_tokens_per_block=256,
                gen_num_per_cycle=3,
            ),
        )
        self.assertEqual(first["gen_num_per_cycle"], 0)


@unittest.skipIf(torch is None, "requires Torch and native cache descriptors")
class CacheResourceTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models.dsv4.specs import cache_description_snapshot
        from rtp_llm.models.dsv4_kv_cache import (
            Dsv4IndexerCacheMode,
            build_dsv4_kv_cache_spec_descs,
        )

        descs = build_dsv4_kv_cache_spec_descs(
            3,
            [0, 4, 128],
            True,
            512,
            128,
            indexer_cache_mode=Dsv4IndexerCacheMode.FP8,
        )
        self.metadata = dict(
            num_layers=3,
            cp_enabled=False,
            cache_geometry=dict(
                physical_tokens_per_block=256,
                kernel_tokens_per_block=256,
                gen_num_per_cycle=0,
            ),
            cache_descriptions=cache_description_snapshot(descs),
        )

    def cache(self, metadata=None):
        metadata = self.metadata if metadata is None else metadata
        layouts = opaque_cache_layouts(metadata)
        tags = list(dict.fromkeys(tag for layer in layouts for tag in layer))
        rows = []
        for layer_id, layer in enumerate(layouts):
            views = []
            for tag, layout in layer.items():
                dtype = {"TYPE_UINT8": torch.uint8, "TYPE_FP32": torch.float32}[
                    layout["dtype"]
                ]
                tensor = torch.zeros(
                    (
                        2 * layout["blocks_per_physical"],
                        layout["stride_bytes"] // dtype.itemsize,
                    ),
                    dtype=dtype,
                )
                views.append(
                    SimpleNamespace(
                        tag=tag,
                        layer_id=layer_id,
                        group_id=tags.index(tag),
                        seq_size_per_block=layout["seq_size_per_block"],
                        kv_cache_base=tensor,
                        kv_scale_base=None,
                    )
                )
            rows.append(views)
        all_layouts = {tag: spec for layer in layouts for tag, spec in layer.items()}
        return SimpleNamespace(
            group_tags=tags,
            layer_count=len(rows),
            rows=rows,
            get_layer_cache_groups=lambda layer: rows[layer],
            get_seq_size_per_block=lambda tag: all_layouts[tag][
                "physical_tokens_per_block"
            ],
            get_kernel_seq_size_per_block=lambda tag: all_layouts[tag][
                "kernel_tokens_per_block"
            ],
        )

    def validate(self, cache, metadata=None):
        return validate_bound_cache(
            cache, self.metadata if metadata is None else metadata, device="cpu"
        )

    def test_native_layout_sizes_and_real_tensors(self):
        report = self.validate(self.cache())
        self.assertTrue(report["bound"])
        self.assertEqual(report["layers"], 3)
        self.assertEqual(len(report["groups"]), 7)
        expected = {
            "swa_kv": 74880,
            "csa_kv": 37440,
            "hca_kv": 1728,
            "indexer_kv": 8448,
            "csa_state": 65536,
            "hca_state": 524288,
            "indexer_state": 16384,
        }
        self.assertEqual(
            {tag: spec["stride_bytes"] for tag, spec in report["groups"].items()},
            expected,
        )

    def test_missing_extra_reordered_or_wrong_layer_groups_fail(self):
        for mutation in (
            lambda c: c.rows[1].pop(),
            lambda c: c.rows[1].append(c.rows[0][0]),
            lambda c: c.group_tags.reverse(),
            lambda c: setattr(c.rows[1][0], "layer_id", 0),
            lambda c: setattr(c.rows[1][0], "group_id", 99),
        ):
            cache = self.cache()
            mutation(cache)
            with self.assertRaises(ValueError):
                self.validate(cache)

    def test_wrong_dtype_stride_device_and_scale_fail(self):
        for mutation in (
            lambda v: setattr(v, "kv_cache_base", v.kv_cache_base.float()),
            lambda v: setattr(v, "kv_cache_base", v.kv_cache_base[:, :-1]),
            lambda v: setattr(v, "kv_cache_base", v.kv_cache_base[:, ::2]),
            lambda v: setattr(v, "kv_cache_base", v.kv_cache_base.to("meta")),
            lambda v: setattr(v, "kv_scale_base", torch.ones(1)),
        ):
            cache = self.cache()
            mutation(cache.rows[0][0])
            with self.assertRaises(ValueError):
                self.validate(cache)

    def test_subdivided_kernel_blocks_and_generation_ring(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["cache_geometry"].update(
            physical_tokens_per_block=512, gen_num_per_cycle=3
        )
        report = self.validate(self.cache(metadata), metadata)
        self.assertEqual(report["groups"]["csa_kv"]["shape"][0], 4)
        self.assertEqual(report["groups"]["csa_kv"]["seq_size_per_block"], 256)
        self.assertEqual(report["groups"]["csa_state"]["seq_size_per_block"], 512)
        self.assertEqual(report["groups"]["csa_state"]["entries_per_view"], 12)
        with self.assertRaises(ValueError):
            self.validate(self.cache(metadata))

    def test_no_kv_repeat_and_resource_replacement_do_not_retain_tensors(self):
        self.assertFalse(self.validate(None)["bound"])
        cache = self.cache()
        report = self.validate(cache)
        self.assertEqual(self.validate(cache), report)
        self.assertEqual(self.validate(self.cache()), report)
        import json

        json.dumps(report)

    def test_incompatible_declaration_fails_before_binding(self):
        for path, value in (("cp_enabled", True), ("num_layers", 4)):
            metadata = copy.deepcopy(self.metadata)
            metadata[path] = value
            with self.assertRaises(ValueError):
                self.validate(None, metadata)
