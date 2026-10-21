"""CPU contracts for decode pool ownership discovery (no CUDA execution)."""

import unittest
from collections import Counter
from types import SimpleNamespace

from rtp_llm.models_py.modules.dsv41.attn_type import (
    CSA_KV,
    DECODER_SWA_KV,
    HCA_KV,
    INDEXER_KV,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv41.decode.forward import build_paged_pool_specs


class _Cache:
    def __init__(self, owners, *, declared=True, group_tags=None):
        self.owners = owners
        self.group_tags = (
            tuple(dict.fromkeys(tag for tags in owners.values() for tag in tags))
            if group_tags is None
            else group_tags
        )
        self.owner_queries = Counter()
        self.probes = []
        self.strict = declared
        if not declared:
            self.get_layer_cache_groups = None

    def get_layer_cache_groups(self, layer_id):
        self.owner_queries[layer_id] += 1
        return [SimpleNamespace(tag=tag) for tag in self.owners[layer_id]]

    def get_layer_cache(self, layer_id, tag):
        if tag not in self.owners[layer_id]:
            raise RuntimeError("layer does not own tag")
        return SimpleNamespace(tag=tag)

    def get_kernel_seq_size_per_block(self, tag):
        return 128

    def get_seq_size_per_block(self, tag):
        return 256


class _Attn:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self._kv_cache = object()
        self._swa_cache_region = SWA_KV
        self.fail_tag = None

    def _pool_entries_per_block(self, tag):
        cache = self._kv_cache
        resolved = self._swa_cache_region if tag == SWA_KV else tag
        cache.probes.append((self.layer_id, resolved))
        if resolved == self.fail_tag:
            raise ValueError("intentional pool geometry failure")
        if cache.strict:
            assert resolved in cache.owners[self.layer_id], "non-owner probe"
        return cache.owners[self.layer_id].get(resolved, 0)


def _model(*layer_ids):
    return SimpleNamespace(layers=[SimpleNamespace(attn=_Attn(i)) for i in layer_ids])


class DecodePoolSpecsTest(unittest.TestCase):
    def test_only_declared_owners_are_probed_and_each_layer_is_queried_once(self):
        cache = _Cache(
            {
                3: {SWA_KV: 32},
                7: {SWA_KV: 32, HCA_KV: 64},
                11: {SWA_KV: 32, INDEXER_KV: 16, CSA_KV: 8},
            }
        )
        model = _model(3, 7, 11)
        bindings = [layer.attn._kv_cache for layer in model.layers]
        self.assertEqual(
            build_paged_pool_specs(cache, model, 4096),
            {
                SWA_KV: (32, 256, 33),
                HCA_KV: (64, 128, 33),
                INDEXER_KV: (16, 128, 33),
                CSA_KV: (8, 128, 33),
            },
        )
        self.assertEqual(cache.owner_queries, {3: 1, 7: 1, 11: 1})
        self.assertEqual(len(cache.probes), 4)
        for layer, binding in zip(model.layers, bindings):
            self.assertIs(layer.attn._kv_cache, binding)
            self.assertEqual(layer.attn._swa_cache_region, SWA_KV)

    def test_encoder_decoder_swa_and_legacy_cache_compatibility(self):
        for declared in (False, True):
            with self.subTest(declared=declared):
                cache = _Cache(
                    {0: {SWA_KV: 128}, 21: {DECODER_SWA_KV: 140}},
                    declared=declared,
                )
                model = _model(0, 21)
                self.assertEqual(
                    build_paged_pool_specs(cache, model, 4096),
                    {SWA_KV: (128, 256, 33), DECODER_SWA_KV: (140, 256, 33)},
                )
                self.assertEqual(cache.probes, [(0, SWA_KV), (21, DECODER_SWA_KV)])
                for layer in model.layers:
                    self.assertEqual(layer.attn._swa_cache_region, SWA_KV)

    def test_legacy_pool_probe_can_skip_non_owners(self):
        cache = _Cache({0: {SWA_KV: 32}, 1: {HCA_KV: 64}}, declared=False)
        self.assertEqual(
            build_paged_pool_specs(cache, _model(0, 1), 4096),
            {SWA_KV: (32, 256, 33), HCA_KV: (64, 128, 33)},
        )
        self.assertIn((0, HCA_KV), cache.probes)

    def test_missing_global_tag_is_not_probed(self):
        cache = _Cache({0: {SWA_KV: 32, HCA_KV: 64}}, group_tags=(SWA_KV,))
        self.assertEqual(
            build_paged_pool_specs(cache, _model(0), 4096),
            {SWA_KV: (32, 256, 33)},
        )
        self.assertEqual(cache.probes, [(0, SWA_KV)])

    def test_declared_tag_without_allocated_owner_is_omitted(self):
        cache = _Cache({0: {SWA_KV: 32}}, group_tags=(SWA_KV, HCA_KV))
        self.assertEqual(
            build_paged_pool_specs(cache, _model(0), 4096),
            {SWA_KV: (32, 256, 33)},
        )
        self.assertEqual(cache.probes, [(0, SWA_KV)])

    def test_bindings_are_restored_after_geometry_error(self):
        cache = _Cache({0: {SWA_KV: 32}, 1: {DECODER_SWA_KV: 140}})
        model = _model(0, 1)
        bindings = [layer.attn._kv_cache for layer in model.layers]
        model.layers[1].attn.fail_tag = DECODER_SWA_KV
        with self.assertRaisesRegex(ValueError, "intentional pool geometry failure"):
            build_paged_pool_specs(cache, model, 4096)
        for layer, binding in zip(model.layers, bindings):
            self.assertIs(layer.attn._kv_cache, binding)
            self.assertEqual(layer.attn._swa_cache_region, SWA_KV)

    def test_ownership_and_geometry_are_refreshed_on_each_call(self):
        cache = _Cache({0: {SWA_KV: 32}, 1: {HCA_KV: 64}})
        model = _model(0, 1)
        first = build_paged_pool_specs(cache, model, 4096)
        cache.owners = {0: {SWA_KV: 32, HCA_KV: 96}, 1: {}}
        second = build_paged_pool_specs(cache, model, 8192)
        self.assertEqual(first[HCA_KV], (64, 128, 33))
        self.assertEqual(second[HCA_KV], (96, 128, 65))
        self.assertEqual(cache.owner_queries[0], 2)

    def test_absent_cache_or_layers_is_empty(self):
        self.assertEqual(build_paged_pool_specs(None, _model(0), 4096), {})
        self.assertEqual(build_paged_pool_specs(_Cache({}), _model(), 4096), {})


if __name__ == "__main__":
    unittest.main()
