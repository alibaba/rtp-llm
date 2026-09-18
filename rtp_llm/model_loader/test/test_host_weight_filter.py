import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.utils.database import CkptDatabase


def _file(name, tensor_names):
    return SimpleNamespace(file_name=name, get_tensor_names=lambda: tensor_names)


class HostWeightFilterTest(unittest.TestCase):
    def test_host_tables_excluded_before_reader_allocation_in_mixed_shard(self):
        database = CkptDatabase(None)
        database.pretrain_file_list = [
            _file("gpu.safetensors", ["layers.0.attn.wq_a.weight"]),
            _file("host.safetensors", ["layers.1.engram.embed.weight"]),
            _file(
                "mixed.safetensors",
                ["layers.2.attn.wq_a.weight", "layers.14.engram.embed.weight"],
            ),
        ]
        loader = Mock()
        loader.iterate_weights.return_value = iter(())
        auto_loader = Mock(return_value=loader)
        package = SimpleNamespace(AutoLoader=auto_loader, SingleGroup=Mock)
        source_filter = lambda name: ".engram." not in name
        with patch.dict(sys.modules, {"fastsafetensors": package}), patch(
            "torch.distributed.is_initialized", return_value=False
        ):
            list(
                database.fastsafetensors_weights_iterator(
                    "cpu", False, source_tensor_filter=source_filter
                )
            )
        self.assertEqual(
            auto_loader.call_args.args[1],
            ["gpu.safetensors", "mixed.safetensors"],
        )
        reader_filter = auto_loader.call_args.kwargs["tensor_filter"]
        self.assertIs(reader_filter, source_filter)
        self.assertFalse(reader_filter("layers.14.engram.embed.weight"))
        self.assertTrue(reader_filter("layers.2.attn.wq_a.weight"))
        loader.close.assert_called_once()

    def test_other_models_keep_existing_reader_arguments(self):
        database = CkptDatabase(None)
        database.pretrain_file_list = [_file("gpu.safetensors", ["weight"])]
        loader = Mock()
        loader.iterate_weights.return_value = iter(())
        auto_loader = Mock(return_value=loader)
        package = SimpleNamespace(AutoLoader=auto_loader, SingleGroup=Mock)
        with patch.dict(sys.modules, {"fastsafetensors": package}), patch(
            "torch.distributed.is_initialized", return_value=False
        ):
            list(database.fastsafetensors_weights_iterator("cpu", False))
        self.assertNotIn("tensor_filter", auto_loader.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
