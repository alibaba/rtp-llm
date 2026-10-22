"""Behavioral cache routing contracts, runnable without native/CUDA modules."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CACHE = _load("v41_cache_contract", Path(__file__).parents[1] / "kv_cache_utils.py")


class CacheTagsTest(unittest.TestCase):
    def test_group_metadata_is_not_silently_taken_from_primary_pool(self):
        swa = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32)
        global_kv = torch.tensor([[7], [9]], dtype=torch.int32)
        inputs = {
            "swa_kv": SimpleNamespace(
                kv_cache_kernel_block_id_device=swa, is_prefill=True
            ),
            "global_kv_2": SimpleNamespace(
                kv_cache_kernel_block_id_device=global_kv, is_prefill=True
            ),
        }
        view = CACHE.as_tagged_attention_inputs(inputs)
        self.assertTrue(view.is_prefill)
        tables = CACHE.build_block_tables(None, view, batch_offset=1)
        self.assertEqual(tables["swa_kv"].tolist(), [[2, 4]])
        self.assertEqual(tables["global_kv_2"].tolist(), [[9]])
        self.assertEqual(
            set(CACHE.build_block_tables_for_tags(None, view, ["global_kv_2"])),
            {"global_kv_2"},
        )

    def test_host_mirrors_keep_group_order_and_invalid_padding(self):
        cache = SimpleNamespace(group_tags=["global_kv_2", "decoder_swa_kv"])
        inputs = {
            "decoder_swa_kv": SimpleNamespace(
                kv_cache_block_id=torch.tensor([[6, 8]], dtype=torch.int32)
            ),
            "global_kv_2": SimpleNamespace(
                kv_cache_block_id=torch.tensor([[9]], dtype=torch.int32)
            ),
        }
        self.assertEqual(
            CACHE.host_block_tables(cache, inputs).tolist(), [[[9, -1]], [[6, 8]]]
        )

    def test_swa_owner_lookup_uses_declared_groups_without_missing_tag_probe(self):
        def rejected_probe(*args):
            self.fail("A declared group lookup must not probe nonexistent tags")

        cache = SimpleNamespace(
            group_tags=["swa_kv", "decoder_swa_kv"],
            get_layer_cache_groups=lambda layer: [
                SimpleNamespace(tag="decoder_swa_kv" if layer == 1 else "swa_kv")
            ],
            get_layer_cache=rejected_probe,
        )
        self.assertEqual(CACHE.swa_region_for_layer(cache, 0), "swa_kv")
        self.assertEqual(CACHE.swa_region_for_layer(cache, 1), "decoder_swa_kv")

    def test_bounded_swa_routing_is_layer_specific(self):
        def get(layer, tag):
            if (layer < 21) != (tag == "swa_kv"):
                raise RuntimeError("layer does not own tag")
            return object()

        cache = SimpleNamespace(
            group_tags=["decoder_swa_kv", "swa_kv"], get_layer_cache=get
        )
        self.assertEqual(CACHE.swa_region_for_layer(cache, 0), "swa_kv")
        self.assertEqual(CACHE.swa_region_for_layer(cache, 21), "decoder_swa_kv")
        owner = SimpleNamespace()
        for layer in (0, 21, 14, 39, 21, 0):
            self.assertEqual(
                CACHE.cached_swa_region(owner, cache, layer),
                "swa_kv" if layer < 21 else "decoder_swa_kv",
            )
        replacement = SimpleNamespace(group_tags=["swa_kv"])
        self.assertEqual(CACHE.cached_swa_region(owner, replacement, 21), "swa_kv")
        tables = {"swa_kv": torch.tensor([1]), "decoder_swa_kv": torch.tensor([8])}
        self.assertEqual(
            CACHE.bind_swa_table(tables, "decoder_swa_kv")["swa_kv"].item(), 8
        )
        self.assertEqual(tables["swa_kv"].item(), 1)


if __name__ == "__main__":
    unittest.main()
