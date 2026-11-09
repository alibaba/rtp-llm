"""Packed commit rows retain request order and never write padding to KV."""

import ast
import itertools
import unittest
from pathlib import Path

import torch


def load_layout():
    source = Path(__file__).resolve().parents[1] / "metadata/ppu_decode_metadata.py"
    tree = ast.parse(source.read_text())
    definition = next(
        node for node in tree.body if getattr(node, "name", None) == "_MtpCommitLayout"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["_MtpCommitLayout"]


class MtpCommitLayoutTest(unittest.TestCase):
    def test_variable_acceptance_preserves_packed_rows_and_masks_padding(self):
        layout_class = load_layout()
        for width in (3, 4):
            lengths = torch.zeros(3, dtype=torch.int32)
            layout = layout_class(lengths, 3, width)
            pointers = [
                tensor.data_ptr()
                for tensor in (
                    layout.padded_to_compact,
                    layout.compact_to_padded,
                    layout.valid_rows,
                )
            ]
            for accepted in itertools.product(range(width + 1), repeat=3):
                lengths.copy_(torch.tensor(accepted, dtype=torch.int32))
                layout.update()
                expected_valid = [
                    column < count for count in accepted for column in range(width)
                ]
                self.assertEqual(layout.valid_rows.tolist(), expected_valid)
                packed = torch.arange(3 * width)
                padded = packed.index_select(0, layout.padded_to_compact)
                cursor = 0
                for request, count in enumerate(accepted):
                    self.assertEqual(
                        padded[request * width : request * width + count].tolist(),
                        list(range(cursor, cursor + count)),
                    )
                    cursor += count
                restored = padded.index_select(0, layout.compact_to_padded)
                self.assertEqual(restored[:cursor].tolist(), list(range(cursor)))
                slots = torch.arange(3 * width, dtype=torch.int32)
                layout.mask_writes(slots)
                self.assertEqual(
                    slots.tolist(),
                    [
                        index if valid else -1
                        for index, valid in enumerate(expected_valid)
                    ],
                )
                self.assertEqual(
                    pointers,
                    [
                        tensor.data_ptr()
                        for tensor in (
                            layout.padded_to_compact,
                            layout.compact_to_padded,
                            layout.valid_rows,
                        )
                    ],
                )


if __name__ == "__main__":
    unittest.main()
