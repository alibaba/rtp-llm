import os
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from standalone_load import load_component

_inputs = load_component("rtp_v41_inputs_test", "models_py/modules/dsv41/inputs.py")
_engram = load_component(
    "rtp_v41_input_engram_test", "models_py/modules/dsv41/engram.py"
)
V41ModelRows = _inputs.V41ModelRows
V41GraphInputBuffers = _inputs.V41GraphInputBuffers


class V41InputsTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))

    def rows(self, *, image=True):
        tokens = [0, 1, 3, 129264, 129264, 129264, 129264, 1, 3, 0, 1]
        types = [-1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1]
        if not image:
            tokens = [0, 1, 3]
            types = [-1] * 3
        history = [[0] * 3 for _ in tokens]
        history_valid = [[False] * 3 for _ in tokens]
        for row in range(len(tokens)):
            for slot, predecessor in enumerate(range(row - 3, row)):
                if predecessor >= 0:
                    history[row][slot] = tokens[predecessor]
                    history_valid[row][slot] = types[predecessor] == -1
        return V41ModelRows(
            torch.tensor(tokens, dtype=torch.int32, device=self.device),
            torch.tensor(types, dtype=torch.int32, device=self.device),
            torch.ones(len(tokens), dtype=torch.bool, device=self.device),
            torch.tensor(history, dtype=torch.int32, device=self.device),
            torch.tensor(history_valid, dtype=torch.bool, device=self.device),
        )

    def test_per_row_hashes_match_canonical_full_sequence_after_reorder(self):
        config = SimpleNamespace(
            text={
                "engram_compressed_vocab_size": 4,
                "engram_pad_token_id": 2,
                "engram_max_ngram_size": 4,
                "engram_layer_ids": [1, 14],
                "engram_num_embeddings": [56, 180],
                "engram_n_heads": 2,
                "engram_vocab_size": 3,
            }
        )
        hasher = _engram.EngramHash(config, [0, 1, 2, 3]).to(self.device)
        rows = self.rows()
        expected = hasher(
            rows.token_ids.to(torch.int64)[None],
            torch.zeros((1, 3), dtype=torch.int64, device=self.device),
            torch.zeros((1, 3), dtype=torch.bool, device=self.device),
            rows.text_mask[None],
        )[0]
        self.assertTrue(torch.equal(rows.engram_hashes(hasher), expected))
        order = torch.tensor([10, 0, 7, 3, 9, 8, 4, 1, 6, 5, 2], device=self.device)
        shuffled = V41ModelRows(
            *(
                getattr(rows, name).index_select(0, order)
                for name in (
                    "token_ids",
                    "token_types",
                    "valid",
                    "history_ids",
                    "history_valid",
                )
            )
        )
        self.assertTrue(
            torch.equal(shuffled.engram_hashes(hasher), expected.index_select(0, order))
        )

    def test_graph_storage_clears_image_history_and_unused_rows(self):
        buffers = V41GraphInputBuffers(16, device=self.device)
        pointers = [
            getattr(buffers.rows, name).data_ptr()
            for name in (
                "token_ids",
                "token_types",
                "valid",
                "history_ids",
                "history_valid",
            )
        ]
        buffers.update(self.rows())
        self.assertEqual(buffers.rows.image_mask.sum().item(), 4)
        buffers.update(self.rows(image=False))
        self.assertEqual(buffers.rows.image_mask.sum().item(), 0)
        self.assertEqual(buffers.rows.valid.sum().item(), 3)
        self.assertEqual(buffers.rows.history_ids[3:].abs().sum().item(), 0)
        self.assertFalse(buffers.rows.history_valid[3:].any().item())
        self.assertEqual(buffers.rows.token_types.unique().tolist(), [-1])
        self.assertEqual(
            pointers,
            [
                getattr(buffers.rows, name).data_ptr()
                for name in (
                    "token_ids",
                    "token_types",
                    "valid",
                    "history_ids",
                    "history_valid",
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "capacity"):
            V41GraphInputBuffers(2, device=self.device).update(self.rows(image=False))

    def test_shape_and_dtype_validation_rejects_inconsistent_rows(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "shape"):
            replace(rows, history_ids=rows.history_ids[:, :2].contiguous()).validate()
        with self.assertRaisesRegex(ValueError, "dtype"):
            replace(rows, history_valid=rows.history_valid.to(torch.int32)).validate()
        with self.assertRaisesRegex(ValueError, "contiguity"):
            replace(rows, history_ids=rows.history_ids.T.contiguous().T).validate()

    def test_cuda_graph_replay_observes_text_switch_and_empty_padding(self):
        self.assertEqual(
            self.device.type, "cuda", "this test requires a real CUDA Graph"
        )
        buffers = V41GraphInputBuffers(16, device=self.device)
        buffers.update(self.rows())

        def consume():
            return (
                buffers.rows.image_mask.clone(),
                torch.where(buffers.rows.history_valid, buffers.rows.history_ids, 0),
                buffers.rows.valid.clone(),
            )

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                consume()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = consume()
        graph.replay()
        self.assertEqual(actual[0].sum().item(), 4)
        buffers.update(self.rows(image=False))
        graph.replay()
        expected = consume()
        for captured, eager in zip(actual, expected):
            self.assertTrue(torch.equal(captured, eager))
        self.assertEqual(actual[0].sum().item(), 0)
        self.assertEqual(actual[2].sum().item(), 3)
        self.assertEqual(actual[1][3:].abs().sum().item(), 0)
        graph.reset()


if __name__ == "__main__":
    unittest.main()
