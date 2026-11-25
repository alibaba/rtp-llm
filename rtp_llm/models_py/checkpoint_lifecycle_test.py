import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule


class CheckpointLifecycleTest(unittest.TestCase):
    def test_read_indices_do_not_treat_bool_as_int(self):
        from rtp_llm.models_py.read_spec import ReadSpec

        for index in (False, True):
            with self.subTest(index=index), self.assertRaises(TypeError):
                ReadSpec("w", "w", (2, 3), slices=(index, slice(None)))
        for axis in (False, True):
            with self.subTest(axis=axis), self.assertRaises(TypeError):
                ReadSpec("w", "w", (2,), slices=(slice(None),), shard_dim=axis)
        self.assertEqual(ReadSpec("w", "w", (2,), slices=(0,)).slices, (0,))

    def test_real_checkpoint_stage_order_and_reader_failure(self):
        from rtp_llm.models_py import weight_mapper

        class Owner(RtpModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2), requires_grad=False)

            def process_weights_after_loading(self):
                events.append("post_load_hook")
                if fail == "post_load":
                    raise RuntimeError("fixture post_load failure")

        with tempfile.TemporaryDirectory() as root:
            save_file(
                {"weight": torch.tensor([2.0, 3.0])},
                str(Path(root) / "model.safetensors"),
            )
            for fail in (None, "close", "post_load"):
                events = []
                original = weight_mapper.get_all_weights

                def reader(*args, **kwargs):
                    try:
                        yield from original(*args, **kwargs)
                    finally:
                        events.append("reader_close")
                        if fail == "close":
                            raise RuntimeError("fixture close failure")

                loader = NewModelLoader(
                    {"model_type": "deepseek_v41"},
                    NewLoaderConfig(device="cpu", load_method="scratch"),
                    model_path=root,
                )
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(loader, "_create_model", side_effect=Owner)
                    )
                    stack.enter_context(
                        patch.object(
                            weight_mapper, "get_all_weights", side_effect=reader
                        )
                    )
                    if fail is not None:
                        with self.assertRaisesRegex(
                            RuntimeError, f"fixture {fail} failure"
                        ):
                            loader.load()
                        expected = ["reader_close"]
                        if fail == "post_load":
                            expected.append("post_load_hook")
                        self.assertEqual(events, expected)
                        self.assertIsNone(loader.loaded_method)
                    else:
                        loaded = loader.load()
                        self.assertEqual(loaded.weight.tolist(), [2.0, 3.0])
                        self.assertEqual(events, ["reader_close", "post_load_hook"])


if __name__ == "__main__":
    unittest.main()
