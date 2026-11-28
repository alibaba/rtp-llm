"""The direct loader must deliver shared sources to all local parameter owners."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.module_base import RtpModule


class SharedRow:
    def handles_safetensor(self, name):
        return name == "shared"

    def expand_safetensor(self, name, shape):
        return [("first", 0), ("second", 0)]


class TwoOwners(RtpModule):
    def __init__(self, incomplete):
        super().__init__()
        self.first = torch.nn.Parameter(torch.empty(3))
        self.second = torch.nn.Parameter(torch.empty(3))
        if incomplete:
            self.missing = torch.nn.Parameter(torch.empty(3))


def check_shared_source(case, native, incomplete):
    expected = torch.tensor([[3.0, 5.0, 7.0]])
    with tempfile.TemporaryDirectory() as temp:
        path = str(Path(temp) / "model.safetensors")
        save_file({"shared": expected, "unused": torch.ones(1)}, path)
        opts = {"name_filter": lambda key: key == "shared"}
        if native:
            from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors

            rows = iter_fst_range_tensors([path], slice_expander=SharedRow(), **opts)
        else:
            rows = weight_mapper.get_all_weights(
                [path], safetensor_slice_expander=SharedRow(), **opts
            )
        delivered = []

        def observe():
            for key, tensor in rows:
                delivered.append(key)
                yield key, tensor

        model = TwoOwners(incomplete)
        model.load_weights(observe())
        case.assertEqual(delivered, ["first", "second"])
        for param in (model.first, model.second):
            torch.testing.assert_close(param, expected[0])
        if incomplete:
            with case.assertRaisesRegex(RuntimeError, "missing"):
                model.validate_weights_loaded()
        else:
            model.validate_weights_loaded()


class DirectFanoutTest(unittest.TestCase):
    def test_shared_and_incomplete(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                check_shared_source(self, False, incomplete)

    def test_budget_counts_actual_quantization_storage_and_shared_owners(self):
        # Online conversion owners hold BF16 inputs; offline owners hold packed
        # inputs. Count actual registered storage, including mixed expert ranks.
        for bf16_rows in (0, 2, 4):
            with self.subTest(bf16_rows=bf16_rows):
                owner = torch.nn.Module()
                owner.register_buffer(
                    "online", torch.empty(bf16_rows, 8, dtype=torch.bfloat16)
                )
                owner.register_buffer(
                    "offline", torch.empty(4 - bf16_rows, 8, dtype=torch.uint8)
                )
                owner.register_buffer("tied", owner.online)
                size = NewModelLoader._model_storage_bytes(owner)
                self.assertEqual(size, 32 + 8 * bf16_rows)
                for shard, bounce in ((12, 4), (4, 12)):
                    self.assertEqual(
                        NewModelLoader.fst_staging_required_bytes(size, shard, bounce),
                        size + 36,
                    )


class NativeFanoutTest(unittest.TestCase):
    def test_real_fst_shared_and_incomplete(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                check_shared_source(self, True, incomplete)


if __name__ == "__main__":
    unittest.main()
