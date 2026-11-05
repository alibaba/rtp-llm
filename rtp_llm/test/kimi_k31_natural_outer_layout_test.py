"""Real CPU Torch tests; importing these files never initializes RTP ops."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / "models_py/modules/hybrid"
PATH = HYBRID / "natural_outer_layout.py"


def load_hybrid(name):
    package = "k31_cpu_hybrid"
    if package not in sys.modules:
        module = ModuleType(package)
        module.__path__ = [str(HYBRID)]
        sys.modules[package] = module
    fullname = package + "." + name
    if fullname not in sys.modules:
        spec = importlib.util.spec_from_file_location(fullname, HYBRID / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    return sys.modules[fullname]


def fixture(rank, *, padded=True, holes=False, buffers=None, all_valid_multi=False):
    # Literal inverse of [[0,1,14,15,16,23], [2,3,12,13,17,22],
    # [4,5,10,11,18,21], [6,7,8,9,19,20]].
    if padded:
        restore = [0, 1, 6, 7, 12, 13, 18, 19, 20, 21, 14, 15,
                   8, 9, 2, 3, 4, 10, 16, 22, 23, 17, 11, 5]
        mask = [True] * 11 + [False] * 5 + [True] * 5 + [False] * 3
        lengths, prefixes = [11, 5], [100, 700]
        if all_valid_multi:
            mask, lengths = [True] * 24, [16, 8]
    else:
        restore = [0, 1, 4, 5, 8, 9, 12, 13, 14, 15, 10, 11, 6, 7, 2, 3]
        mask, lengths, prefixes = [True] * 16, [16], [100]
    info = SimpleNamespace(
        prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
        prefill_prefix_lengths_cpu=torch.tensor(prefixes, dtype=torch.int32),
        prefill_qkv_restore_indice=torch.tensor(restore),
        prefill_qkv_padding_mask=torch.tensor(mask),
    )
    valid = None
    if holes:
        valid = torch.ones(16, dtype=torch.bool)
        valid[[2, 12]] = False
    return load_hybrid("query_layout").QueryLayout(info, 4, rank, "cpu", valid, buffers)


class NaturalOuterLayoutTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), "natural outer layout adapter is missing")
        self.layout_type = load_hybrid("natural_outer_layout").NaturalOuterLayout

    def test_all_valid_reuses_temporal_metadata_and_preserves_multiple_requests(self):
        for rank in range(4):
            original = fixture(rank, all_valid_multi=True)
            layout = self.layout_type(original)
            self.assertTrue(layout.all_valid)
            for name in ("cu_host", "cu_device", "lengths_host", "lengths_device",
                         "padding_mask", "natural_valid"):
                self.assertEqual(getattr(layout, name).data_ptr(), getattr(original, name).data_ptr(), name)
            self.assertEqual(layout.restore.tolist(), list(range(24)))
            self.assertEqual(layout.natural_to_rankmajor.data_ptr(), layout.restore.data_ptr())
            self.assertEqual(layout.cu_device.tolist(), [0, 16, 24])
            self.assertEqual(layout.rankmajor_requests.tolist(), [0] * 16 + [1] * 8)
            expected_positions = list(range(100, 116)) + list(range(700, 708))
            self.assertEqual(layout.rankmajor_positions.tolist(), expected_positions)
            self.assertEqual(layout.local_positions.tolist(), expected_positions[rank * 6:(rank + 1) * 6])
            layout.prepare_last_hidden_rows()
            self.assertEqual(layout.last_local_offsets.tolist(), [3, 5])
            self.assertEqual(layout.last_is_mine.tolist(), [rank == 2, rank == 3])
            token = object()
            self.assertIsNot(layout.natural_attention(token), original.natural_attention(token))

    def test_all_valid_aliases_without_projection_or_output_reservation(self):
        # Catches the accidental full projected/output copies on the common path.
        original = fixture(1, padded=False)
        layout = self.layout_type(original)
        layout.prepare_kda_buffers(17, 9, quantized=True)
        self.assertEqual(layout._buffers, {})
        projected = torch.arange(16 * 17).reshape(16, 17).to(torch.bfloat16)
        scales = torch.ones((16, 1), dtype=torch.int32)
        self.assertIs(layout.gather_natural(projected), projected)
        values, actual_scales = layout.scatter_rankmajor(projected, scales)
        self.assertIs(values, projected)
        self.assertIs(actual_scales, scales)
        self.assertEqual(layout.local_positions.tolist(), [104, 105, 106, 107])
        self.assertEqual(original.local_positions.tolist(), [102, 103, 112, 113])

    def test_padding_holes_positions_and_request_boundaries(self):
        expected_valid = [True, True, False] + [True] * 8 + [False] * 5
        expected_valid += [True, False, True, True, True] + [False] * 3
        expected_positions = [100, 101, -1] + list(range(103, 111)) + [-1] * 5
        expected_positions += [700, -1, 702, 703, 704] + [-1] * 3
        for rank in range(4):
            original = fixture(rank, holes=True)
            before = original.restore.clone()
            layout = self.layout_type(original)
            self.assertFalse(layout.all_valid)
            self.assertEqual(layout.physical_rows, 24)
            self.assertEqual(layout.restore.tolist(), list(range(24)))
            self.assertEqual(layout.local_valid.tolist(), expected_valid[rank * 6:(rank + 1) * 6])
            self.assertEqual(layout.local_positions.tolist(), expected_positions[rank * 6:(rank + 1) * 6])
            self.assertEqual(layout.rankmajor_requests.tolist(), [0] * 16 + [1] * 5 + [0] * 3)
            self.assertTrue(torch.equal(original.restore, before))
            self.assertNotEqual(layout.padding_mask.data_ptr(), original.padding_mask.data_ptr())
            self.assertNotEqual(layout.natural_valid.data_ptr(), original.natural_valid.data_ptr())

    def test_compaction_expansion_initializes_padding_and_reuses_capacity(self):
        buffers = {}
        layout = self.layout_type(fixture(0, holes=True), buffers=buffers)
        layout.prepare_kda_buffers(3, 2, quantized=True)
        physical = torch.arange(72).reshape(24, 3).to(torch.bfloat16)
        gathered = layout.gather_natural(physical)
        expected_rows = list(range(11)) + list(range(16, 21))
        self.assertTrue(torch.equal(gathered, physical[expected_rows]))
        # Explicit valid holes retain compact positions and are masked by attention.
        values = torch.arange(32).reshape(16, 2).to(torch.bfloat16)
        scales = torch.full((16, 1), 9, dtype=torch.int32)
        out, restored = layout.scatter_rankmajor(values, scales)
        self.assertTrue(torch.equal(out[expected_rows], values))
        padding = [11, 12, 13, 14, 15, 21, 22, 23]
        self.assertEqual(out[padding].count_nonzero().item(), 0)
        self.assertEqual(restored[padding].flatten().tolist(), [0x7F7F7F7F] * 8)
        pointers = {key: value.data_ptr() for key, value in buffers.items()}
        layout.prepare_kda_buffers(3, 2, quantized=True)
        self.assertEqual(pointers, {key: value.data_ptr() for key, value in buffers.items()})
        out.fill_(42)
        out2, _ = layout.scatter_rankmajor(values)
        self.assertEqual(out2[padding].count_nonzero().item(), 0)
        fp8 = values.to(torch.float8_e4m3fn)
        floating_scales = torch.full((16, 1), 0.5)
        fp8_out, floating_restored = layout.scatter_rankmajor(fp8, floating_scales)
        self.assertTrue(torch.equal(fp8_out.view(torch.uint8)[expected_rows], fp8.view(torch.uint8)))
        self.assertEqual(fp8_out.view(torch.uint8)[padding].count_nonzero().item(), 0)
        self.assertEqual(floating_restored[padding].flatten().tolist(), [1.0] * 8)

    def test_explicit_holes_without_padding_keep_masked_path_and_stable_view(self):
        original = fixture(0, padded=False, holes=True)
        layout = self.layout_type(original)
        self.assertFalse(layout.all_valid)
        token = object()
        view = layout.natural_attention(token)
        self.assertIs(view, layout.natural_attention(token))
        self.assertIsNot(view, original.natural_attention(token))
        self.assertEqual(view.cu_seqlens.tolist(), [0, 16])
        self.assertEqual(view.valid_token_mask.tolist(), original.natural_valid.tolist())
        layout.prepare_last_hidden_rows()
        self.assertEqual(layout.last_local_offsets.tolist(), [3])
        self.assertEqual(layout.last_is_mine.tolist(), [False])


if __name__ == "__main__":
    unittest.main()
