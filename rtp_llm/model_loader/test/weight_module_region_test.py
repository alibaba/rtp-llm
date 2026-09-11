"""The weight pipeline, not only its final .to(), must run in the region."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_loader.weight_module import AtomicWeight


class WeightModuleRegionTest(unittest.TestCase):
    def test_load_and_update_cover_allocation_stages_and_honor_skip(self):
        for skip in (False, True):
            for operation in ("load", "update"):
                with self.subTest(skip=skip, operation=operation):
                    active = False
                    stages = []

                    @contextmanager
                    def region():
                        nonlocal active
                        active = True
                        try:
                            yield
                        finally:
                            active = False

                    def stage(name, result):
                        stages.append((name, active))
                        return result

                    raw = torch.ones(2, 2)
                    weight = AtomicWeight("test", [], skip_weights_region=skip)
                    config = SimpleNamespace(merge_lora=True)
                    source = SimpleNamespace(get_database=lambda: None)
                    with (
                        patch(
                            "rtp_llm.model_loader.weight_module.weights_region", region
                        ),
                        patch.object(
                            weight,
                            "_load_raw_tensor",
                            side_effect=lambda *a: stage("load", raw),
                        ),
                        patch.object(
                            weight,
                            "_merge_lora",
                            side_effect=lambda *a: stage("lora", raw),
                        ),
                        patch.object(
                            weight,
                            "_split",
                            side_effect=lambda *a: stage("split", raw.clone()),
                        ),
                        patch.object(
                            weight,
                            "_postprocess",
                            side_effect=lambda *a: stage(
                                "postprocess", {"nested": {"test": a[0].clone()}}
                            ),
                        ),
                    ):
                        if operation == "load":
                            result = weight.load(source, None, "cpu", config)
                        else:
                            result = weight.update(raw, "cpu", config)
                    expected = (
                        ["load", "lora", "split", "postprocess"]
                        if operation == "load"
                        else ["split", "postprocess"]
                    )
                    self.assertEqual(stages, [(name, not skip) for name in expected])
                    self.assertFalse(active)
                    torch.testing.assert_close(result["test"], raw)


if __name__ == "__main__":
    unittest.main()
