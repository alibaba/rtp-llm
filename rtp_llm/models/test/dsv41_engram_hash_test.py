"""Portable Engram geometry, tokenizer and explicit-history hash contracts."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source

engram = load_codec_source("_dsv41_engram_hash_test", "v41_engram_hash.py")


def small_layout():
    return engram.V41EngramLayout.from_config(
        dict(
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[56, 180],
            engram_max_ngram_size=4,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_compressed_vocab_size=4,
            engram_vocab_size=3,
        ),
        pad_token_id=2,
    )


def scalar_hashes(hasher, windows):
    result = []
    for raw in windows.tolist():
        blocked, mapped = False, []
        for token in raw:
            blocked = blocked or not 0 <= token < hasher.token_map.numel()
            mapped.append(hasher.pad_id if blocked else int(hasher.token_map[token]))
        layers = []
        for multipliers, primes in zip(
            hasher.multipliers.tolist(), hasher.layout.primes
        ):
            row, rolling, offset = [], mapped[0] * multipliers[0], 0
            for lag in range(1, 4):
                rolling ^= mapped[lag] * multipliers[lag]
                for prime in primes[
                    (lag - 1) * hasher.layout.n_heads : lag * hasher.layout.n_heads
                ]:
                    row.append(rolling % prime + offset)
                    offset += prime
            layers.append(row)
        result.append(layers)
    return torch.tensor(result, dtype=torch.int64).reshape(
        windows.shape[0], len(hasher.layout.layer_ids), -1
    )


class V41EngramHashTest(unittest.TestCase):
    def setUp(self):
        self.hasher = engram.V41EngramHasher(small_layout(), [2, 0, 1, 2, 3, 3])

    def test_checkpoint_buckets_and_seeded_multipliers(self):
        path = (
            Path(__file__).parents[2] / "config/test/testdata/deepseek_v41_flash.json"
        )
        config = json.loads(path.read_text())
        layout = engram.V41EngramLayout.from_config(
            config["text_config"], config["pad_token_id"]
        )
        hasher = engram.V41EngramHasher(layout, list(range(99092)))
        self.assertEqual(tuple(map(sum, layout.primes)), (384006168, 384016682))
        self.assertEqual(len({p for row in layout.primes for p in row}), 48)
        self.assertEqual(
            hasher.multipliers.tolist(),
            [
                [76632096046245, 4839876093313, 35959672319349, 73987337458391],
                [67716810739261, 51510806800915, 30921347202721, 82619226485591],
            ],
        )

    def test_tokenizer_normalization_preserves_spaces_and_raw_utf8_parts(self):
        texts = [
            " The",
            "the",
            "THE",
            "\u00c9",
            "e",
            "\uff25",
            " ",
            "\t\n",
            "",
            "\ufffd",
            "\ufffd",
            "\u0301",
        ]

        class Backend:
            def decode(self, ids, skip_special_tokens):
                self_test.assertFalse(skip_special_tokens)
                return texts[ids[0]]

            def id_to_token(self, token):
                return f"raw-{token}"

        class Tokenizer:
            backend_tokenizer = Backend()

            def __len__(self):
                return len(texts)

        self_test = self
        mapping, size = engram.build_compressed_token_map(Tokenizer())
        self.assertEqual(mapping, [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6])
        self.assertEqual(size, 7)

    def test_hashes_match_scalar_oracle_and_table_ranges(self):
        generator = torch.Generator().manual_seed(4141)
        windows = torch.randint(-1, 6, (129, 4), generator=generator, dtype=torch.int32)
        actual = self.hasher(windows, backend="eager")
        torch.testing.assert_close(
            actual.ids, scalar_hashes(self.hasher, windows), rtol=0, atol=0
        )
        torch.testing.assert_close(actual.valid_rows, windows[:, 0] >= 0)
        self.assertTrue(bool((actual.ids >= self.hasher.offsets).all()))
        self.assertTrue(
            bool((actual.ids < self.hasher.offsets + self.hasher.primes).all())
        )

    def test_lookback_stops_at_first_dead_token(self):
        windows = torch.tensor(
            [[4, -1, 0, 1], [4, -1, -1, -1], [-1, 2, 3, 4], [-1, -1, -1, -1]]
        )
        result = self.hasher(windows, backend="eager")
        torch.testing.assert_close(result.ids[0], result.ids[1], rtol=0, atol=0)
        torch.testing.assert_close(result.ids[2], result.ids[3], rtol=0, atol=0)
        self.assertEqual(result.valid_rows.tolist(), [True, True, False, False])
        invalid = self.hasher(
            torch.tensor([[6, 2, 3, 4], [4, 6, 0, 1]]), backend="eager"
        )
        torch.testing.assert_close(invalid.ids[0], result.ids[3], rtol=0, atol=0)
        torch.testing.assert_close(invalid.ids[1], result.ids[0], rtol=0, atol=0)

    def test_chunks_cp_permutation_and_interleaved_forwards_have_no_state(self):
        windows = torch.tensor(
            [[3, -1, -1, -1], [4, 3, -1, -1], [5, 4, 3, -1], [0, 5, 4, 3], [1, 0, 5, 4]]
        )
        full = self.hasher(windows, backend="eager").ids
        first = self.hasher(windows[:2], backend="eager").ids
        self.hasher(torch.tensor([[2, 2, 2, 2]]), backend="eager")
        last = self.hasher(windows[2:], backend="eager").ids
        torch.testing.assert_close(torch.cat([first, last]), full, rtol=0, atol=0)
        order = torch.tensor([4, 0, 3, 1, 2])
        torch.testing.assert_close(
            self.hasher(windows[order], backend="eager").ids,
            full[order],
            rtol=0,
            atol=0,
        )

    def test_empty_windows_and_invalid_input_contracts(self):
        result = self.hasher(torch.empty(0, 4, dtype=torch.int32), backend="eager")
        self.assertEqual(result.ids.shape, (0, 2, 6))
        self.assertEqual(result.valid_rows.shape, (0,))
        for windows in (
            torch.zeros(4, 4),
            torch.zeros(2, 3, dtype=torch.int32),
            torch.zeros(4, 2, dtype=torch.int32).T,
        ):
            with self.assertRaises(ValueError):
                self.hasher(windows, backend="eager")
        for mapping in ([], [0, 1, 2], [0, 1, 4], [0, 1, 2, 3.0]):
            with self.assertRaises(ValueError):
                engram.V41EngramHasher(small_layout(), mapping)


@unittest.skipUnless(
    os.environ.get("TRITON_INTERPRET") == "1", "Set TRITON_INTERPRET=1"
)
class V41EngramHashInterpreterTest(unittest.TestCase):
    def test_integer_hash_kernel_exact_for_tail_dead_rows_and_int64_products(self):
        name = "rtp_llm.models_py.modules.dsv4._v41_engram_hash_triton"
        with patch.dict(sys.modules):
            load_codec_source(name, "_v41_engram_hash_triton.py")
            hasher = engram.V41EngramHasher(small_layout(), [2, 0, 1, 2, 3, 3])
            for dtype in (torch.int32, torch.int64):
                windows = torch.randint(
                    -1,
                    7,
                    (67, 4),
                    generator=torch.Generator().manual_seed(41),
                    dtype=dtype,
                )
                actual = hasher(windows).ids
                torch.testing.assert_close(
                    actual, scalar_hashes(hasher, windows), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
