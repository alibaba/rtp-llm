"""Resolved reader notification through real HF/FT checkpoint loading."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


@register_model("_checkpoint_source_notification_test")
class SourceOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 3))
        self.source = None
        self.events = []
        self.reject = config.reject

    def set_checkpoint_source(self, method):
        if self.reject:
            raise RuntimeError("auxiliary reader rejected")
        self.source = method
        self.events.append(("selected", method))

    def load_weights(self, weights):
        if self.source is None:
            raise RuntimeError("notification must precede weight dispatch")
        self.events.append(("read", self.source))
        return super().load_weights(weights)

    def load_ft_checkpoint(self, checkpoint):
        if self.source != checkpoint.read_backend:
            raise RuntimeError("notification must precede FT reader")
        self.load_weights({"weight": checkpoint.read("global.weight")})

    def process_weights_after_loading(self):
        self.events.append(("processed", self.source))


class CheckpointSourceTest(unittest.TestCase):
    def load(self, *, ft=False, requested="scratch", reject=False):
        with tempfile.TemporaryDirectory() as root:
            name = "rank_0_0_0.global.weight" if ft else "weight"
            value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            save_file({name: value}, os.path.join(root, "model.safetensors"))
            if ft:
                with open(
                    os.path.join(root, "model.safetensors.index.json"), "w"
                ) as stream:
                    json.dump(
                        {
                            "is_ft_style_weight": True,
                            "weight_map": {name: "model.safetensors"},
                        },
                        stream,
                    )
            loader = NewModelLoader(
                SimpleNamespace(
                    model_type="_checkpoint_source_notification_test", reject=reject
                ),
                NewLoaderConfig(
                    device="cpu", load_method=requested, compute_dtype=torch.float32
                ),
                model_path=root,
            )
            model = loader.load()
            torch.testing.assert_close(model.weight, value)
            return loader, model

    def test_hf_and_ft_auto_publish_actual_reader_before_dispatch(self):
        for ft in (False, True):
            for requested in ("auto", "scratch"):
                with self.subTest(ft=ft, requested=requested):
                    loader, model = self.load(ft=ft, requested=requested)
                    self.assertEqual(
                        model.events,
                        [
                            ("selected", "scratch"),
                            ("read", "scratch"),
                            ("processed", "scratch"),
                        ],
                    )
                    self.assertEqual(loader._selected_load_method.value, "scratch")

    def test_hook_failure_aborts_before_any_read(self):
        for ft in (False, True):
            with self.subTest(ft=ft), self.assertRaisesRegex(
                RuntimeError, "auxiliary reader rejected"
            ):
                self.load(ft=ft, reject=True)

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "requires native FST"
    )
    def test_ft_explicit_native_fst_notifies_fst(self):
        import fastsafetensors.cpp as cpp

        self.assertTrue(cpp.__file__.endswith(".so"))
        loader, model = self.load(ft=True, requested="fastsafetensors")
        self.assertEqual(
            model.events,
            [
                ("selected", "fastsafetensors"),
                ("read", "fastsafetensors"),
                ("processed", "fastsafetensors"),
            ],
        )
        self.assertEqual(loader._selected_load_method.value, "fastsafetensors")


if __name__ == "__main__":
    unittest.main()
