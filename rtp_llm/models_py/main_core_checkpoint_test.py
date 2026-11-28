"""Real CPU checkpoint reads through the main-carrier NewLoader core.

Tiny direct-parameter owners intentionally need no inference/native extension.
These tests are not transformer inference, GPU, multi-card or golden smoke.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import ReadSpec, make_sharded_read_spec
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _Rows(RtpModule):
    def __init__(self, rows, rank):
        super().__init__()
        self.rows = rows
        self.rank = rank
        self.weight = torch.nn.Parameter(
            torch.empty(3 if rows is None else len(rows), 3)
        )

    def checkpoint_read_spec(self, name, shape):
        if self.rows is not None:
            return ReadSpec(name, name, shape, row_indices=self.rows)
        return make_sharded_read_spec(name, shape, 0, 2, self.rank)


@register_model("_main_core_checkpoint_test")
class _StageOwner(RtpModule):
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def __init__(self, config, load):
        super().__init__()
        self.owned_ids = (0,) if config.stage == 0 else (1, 2)
        self.layers = torch.nn.ModuleDict(
            {str(i): _Rows(None, load.tp_rank) for i in self.owned_ids}
        )
        if config.stage == 1:
            self.head = _Rows((load.tp_rank, load.tp_rank + 4), load.tp_rank)
        self.postprocessed = False

    def checkpoint_weight_name_filter(self):
        def owns(name):
            name = name.removeprefix("model.")
            if name.startswith("layers."):
                return int(name.split(".")[1]) in self.owned_ids
            return name != "head.weight" or hasattr(self, "head")

        return owns

    def process_weights_after_loading(self):
        self.postprocessed = True

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))


def payload():
    result = {
        f"model.layers.{i}.weight": torch.arange(18, dtype=torch.float32).reshape(6, 3)
        + i * 100
        for i in range(3)
    }
    result["model.head.weight"] = (
        torch.arange(18, dtype=torch.float32).reshape(6, 3) + 500
    )
    return result


class MainCoreCheckpointTest(unittest.TestCase):
    def _load(self, directory, stage=0, rank=0):
        config = SimpleNamespace(model_type="_main_core_checkpoint_test", stage=stage)
        load = NewLoaderConfig(
            device="cpu",
            load_method="scratch",
            tp_size=2,
            tp_rank=rank,
            compute_dtype=torch.float32,
        )
        loader = NewModelLoader(config, load, model_path=directory)
        model = loader.load()
        self.assertEqual(loader.loaded_method, NewLoaderLoadMethod.SCRATCH)
        self.assertTrue(model.postprocessed)
        self.assertFalse(model.training)
        return model

    def test_safetensors_and_bin_load_owned_rows_directly(self):
        values = payload()
        for suffix in ("safetensors", "bin"):
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(
                    directory,
                    (
                        "model.safetensors"
                        if suffix == "safetensors"
                        else "pytorch_model.bin"
                    ),
                )
                if suffix == "safetensors":
                    save_file(values, path)
                else:
                    torch.save(values, path)
                for stage in (0, 1):
                    for rank in (0, 1):
                        with self.subTest(format=suffix, stage=stage, rank=rank):
                            model = self._load(directory, stage, rank)
                            self.assertEqual(
                                list(model.layers), ["0"] if stage == 0 else ["1", "2"]
                            )
                            for layer_id, layer in model.layers.items():
                                torch.testing.assert_close(
                                    layer.weight,
                                    values[f"model.layers.{layer_id}.weight"].chunk(2)[
                                        rank
                                    ],
                                )
                                self.assertEqual(layer.weight.device.type, "cpu")
                            if stage == 1:
                                torch.testing.assert_close(
                                    model.head.weight,
                                    values["model.head.weight"][[rank, rank + 4]],
                                )

    def test_missing_owned_parameter_fails_before_publication(self):
        values = payload()
        del values["model.layers.0.weight"]
        with tempfile.TemporaryDirectory() as directory:
            save_file(values, os.path.join(directory, "model.safetensors"))
            with self.assertRaisesRegex(
                (RuntimeError, ValueError),
                "Checkpoint filter|not loaded|Missing|missing|loaded",
            ):
                self._load(directory)

    def test_failed_retry_does_not_report_a_successful_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.safetensors")
            values = payload()
            save_file(values, path)
            loader = NewModelLoader(
                SimpleNamespace(model_type="_main_core_checkpoint_test", stage=0),
                NewLoaderConfig(device="cpu", load_method="scratch", tp_size=2),
                model_path=directory,
            )
            self.assertIsNone(loader.loaded_method)
            previous = loader.load()
            self.assertEqual(loader.loaded_method, NewLoaderLoadMethod.SCRATCH)
            del values["model.layers.0.weight"]
            save_file(values, path)
            with self.assertRaises((ValueError, RuntimeError)):
                loader.load()
            self.assertIsNone(loader.loaded_method)
            torch.testing.assert_close(
                previous.layers["0"].weight, payload()["model.layers.0.weight"][:3]
            )

    def test_unknown_weight_is_not_silently_dropped(self):
        values = payload()
        values["unexpected.weight"] = torch.ones(2)
        with tempfile.TemporaryDirectory() as directory:
            save_file(values, os.path.join(directory, "model.safetensors"))
            with self.assertRaises(RuntimeError):
                self._load(directory)

    def test_main_runtime_capacity_config_is_validated(self):
        marker = object()
        config = NewLoaderConfig(
            device="cpu", max_generate_batch_size=13, py_hw_kernel_config=marker
        )
        self.assertIs(config.py_hw_kernel_config, marker)
        self.assertEqual(config.max_generate_batch_size, 13)
        for invalid in (-1, True, 1.5):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                NewLoaderConfig(device="cpu", max_generate_batch_size=invalid)


if __name__ == "__main__":
    unittest.main()
