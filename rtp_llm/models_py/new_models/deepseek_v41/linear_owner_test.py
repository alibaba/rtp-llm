"""Real CPU checkpoint staging tests. Runtime execution requires Blackwell CUDA."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v41.linear_owner import Dsv41MXFP8Linear
from rtp_llm.models_py.weight_mapper import get_all_weights


class MXFP8ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = str(Path(self.folder.name) / "model.safetensors")
        self.weights = {
            "weight": (torch.arange(128 * 64).reshape(128, 64) % 17).to(
                torch.float8_e4m3fn
            ),
            "scale": torch.arange(120, 128, dtype=torch.uint8)
            .view(torch.float8_e8m0fnu)
            .reshape(4, 2),
        }
        save_file(self.weights, self.path)

    def test_projection_tiles_read_before_dispatch(self):
        for mode, dim in (("replicated", None), ("column", 0), ("row", 1)):
            for rank in (0, 1):
                with self.subTest(mode=mode, rank=rank):
                    leaf = Dsv41MXFP8Linear(
                        64, 128, tp_size=2, tp_rank=rank, partition=mode
                    )
                    leaf.load_weights(
                        get_all_weights(
                            [self.path], read_spec_resolver=leaf.checkpoint_read_spec
                        )
                    )
                    leaf.validate_weights_loaded()
                    self.assertEqual(
                        set(dict(leaf.named_parameters())), {"weight", "scale"}
                    )
                    for name, value in self.weights.items():
                        expected = value if dim is None else value.chunk(2, dim)[rank]
                        self.assertTrue(
                            torch.equal(
                                getattr(leaf, name).view(torch.uint8),
                                expected.view(torch.uint8),
                            )
                        )

    def test_incomplete_or_invalid_layout_does_not_pass(self):
        for name, value, error in (
            ("scale", torch.ones(1, 1).to(torch.float8_e8m0fnu), ValueError),
            ("scale", self.weights["scale"].view(torch.uint8), TypeError),
            ("weight", self.weights["weight"].to(torch.float8_e5m2), TypeError),
            ("scale", None, RuntimeError),
        ):
            tensors = dict(self.weights)
            if value is None:
                del tensors[name]
            else:
                tensors[name] = value
            save_file(tensors, self.path)
            leaf = Dsv41MXFP8Linear(64, 128)
            with self.subTest(name=name, error=error), self.assertRaises(error):
                leaf.load_weights(
                    get_all_weights(
                        [self.path], read_spec_resolver=leaf.checkpoint_read_spec
                    )
                )
                leaf.validate_weights_loaded()

    def test_rejects_partial_tiles_and_other_rank_shard(self):
        with self.assertRaisesRegex(ValueError, "scale block"):
            Dsv41MXFP8Linear(96, 128, tp_size=2, partition="row")
        with self.assertRaisesRegex(ValueError, "align"):
            Dsv41MXFP8Linear(65, 128)
        left = Dsv41MXFP8Linear(64, 128, tp_size=2, tp_rank=0, partition="column")
        right = Dsv41MXFP8Linear(64, 128, tp_size=2, tp_rank=1, partition="column")
        with self.assertRaisesRegex(ValueError, "Wrong.*ReadSpec"):
            left.load_weights(
                get_all_weights(
                    [self.path], read_spec_resolver=right.checkpoint_read_spec
                )
            )

    def test_no_implicit_cpu_or_dequant_execution(self):
        leaf = Dsv41MXFP8Linear(64, 128)
        leaf.load_weights(get_all_weights([self.path]))
        leaf.validate_weights_loaded()
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            leaf.process_weights_after_loading()
        with self.assertRaisesRegex(RuntimeError, "prepared"):
            leaf(torch.ones(1, 64, dtype=torch.bfloat16))

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "requires native FST"
    )
    def test_native_fst_reader_preserves_e8m0_scale(self):
        import fastsafetensors.cpp as cpp

        self.assertTrue(cpp.__file__.endswith(".so"))
        loader = NewModelLoader(
            SimpleNamespace(model_type="_v41_leaf"),
            NewLoaderConfig(device="cpu", load_method="fastsafetensors"),
            model_path=self.folder.name,
        )
        leaf = Dsv41MXFP8Linear(64, 128)
        leaf.load_weights(
            loader._fastsafetensors_weights_iterator(
                [self.path], "cpu", None, None, leaf.checkpoint_read_spec
            )
        )
        leaf.validate_weights_loaded()
        for name, source in self.weights.items():
            self.assertTrue(
                torch.equal(
                    getattr(leaf, name).view(torch.uint8), source.view(torch.uint8)
                )
            )


if __name__ == "__main__":
    unittest.main()
