"""Real full-table Engram to block32 projection/gate binding, including Graph.

The CPU table decoder is independent. Projection/gate equivalence isolates
the model binding and GPU lookup; it is not full-model numerical acceptance.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

import torch
from host_cuda_test_support import cpu_lookup_reference
from transformers import AutoTokenizer

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.model_loader.host_shared_cuda import SharedEngramLookup
from rtp_llm.models_py.modules.dsv41.engram import (
    Engram,
    EngramHash,
    build_compressed_token_map,
)
from rtp_llm.models_py.modules.dsv41.math import engram_inject
from rtp_llm.utils.database import CkptDatabase


class FullHostEngramBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getuid() == 0 or not torch.cuda.is_available():
            raise RuntimeError("Engram binding requires non-root CUDA13 Blackwell")
        cls.checkpoint = Path(os.environ["DSV41_MODEL_PATH"])
        cls.config = V41Config.from_path(cls.checkpoint)
        cls.lookup = SharedEngramLookup.from_checkpoint(
            cls.checkpoint,
            os.environ["DSV41_ENGRAM_STORE_ROOT"],
            os.environ["DSV41_HF_REVISION"],
            device=0,
        )
        try:
            cls.lookup.warmup()
            cls.accounting = cls.lookup.accounting()
            cls.database = CkptDatabase(str(cls.checkpoint))
            cls.tokenizer = AutoTokenizer.from_pretrained(
                str(cls.checkpoint), local_files_only=True
            )
            cls.hasher = EngramHash(
                cls.config, build_compressed_token_map(cls.tokenizer)
            ).cuda()
            cls.modules, cls.loaded = {}, {}
            for layer in (1, 14):
                local = {}
                for suffix, dtype in (
                    ("wkv.weight", torch.float8_e4m3fn),
                    ("wkv.scale", torch.float8_e8m0fnu),
                    ("q_weight", torch.bfloat16),
                    ("k_weight", torch.bfloat16),
                ):
                    local["engram." + suffix] = cls.database.load_tensor(
                        f"layers.{layer}.engram.{suffix}", dtype
                    )[0].cuda()
                cls.loaded[layer] = local
                cls.modules[layer] = Engram.from_weights(layer, local, cls.lookup)
            cls.records = []
        except BaseException:
            cls.lookup.close()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            if output:
                Path(output, "engram_model_binding.json").write_text(
                    json.dumps(
                        {
                            "scope": "full host-table/model projection binding; not full-model or topology acceptance",
                            "revision": cls.lookup.shared.manifest["revision"],
                            "registration": cls.accounting,
                            "cases": cls.records,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        finally:
            cls.lookup.close()

    def rows(self, count, offset):
        ids = (torch.arange(count, device="cuda", dtype=torch.int64) + offset).view(
            1, count
        )
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[:, 1::4] = False
        history = torch.tensor([[2, 3, 4]], device="cuda", dtype=torch.int64)
        history_valid = torch.tensor([[False, True, True]], device="cuda")
        hashes = self.hasher(ids, history, history_valid, valid).squeeze(0)
        hidden = (
            torch.arange(count * 4 * 5120, device="cuda", dtype=torch.float32)
            .remainder(31)
            .sub_(15)
            .mul_(0.03125)
            .reshape(count, 4, 5120)
            .bfloat16()
        )
        return hashes, valid[0], hidden

    def expected(self, module, hidden, hashes, valid):
        decoded = cpu_lookup_reference(
            self.lookup.shared,
            module.layer_id,
            hashes,
            valid[:, None].expand_as(hashes),
        ).cuda()
        projected = module.projection(decoded.flatten(-2))
        return engram_inject(hidden, projected, module.q_weight, module.k_weight, valid)

    def equal(self, actual, expected, **metadata):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        digest = lambda value: hashlib.sha256(
            value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        self.records.append(
            {
                "test": self.id(),
                "actual_sha256": digest(actual),
                "expected_sha256": digest(expected),
                **metadata,
            }
        )

    def test_loaded_projection_and_gates_keep_real_checkpoint_storage(self):
        self.assertEqual(self.lookup.total_bytes, 202758032400)
        self.assertEqual(self.lookup.total_bytes, self.config.engram_host_bytes)
        for layer, module in self.modules.items():
            local = self.loaded[layer]
            self.assertIs(module.shared_lookup, self.lookup)
            self.assertEqual(
                module.projection.weight.data_ptr(),
                local["engram.wkv.weight"].data_ptr(),
            )
            self.assertEqual(
                module.q_weight.data_ptr(), local["engram.q_weight"].data_ptr()
            )
            self.assertEqual(
                module.k_weight.data_ptr(), local["engram.k_weight"].data_ptr()
            )
            self.assertFalse(any("embed" in key for key in module.state_dict()))
        bad = dict(self.loaded[1])
        bad["engram.embed.weight"] = torch.empty(0)
        with self.assertRaisesRegex(ValueError, "host-shared"):
            Engram.from_weights(1, bad, self.lookup)
        bad = dict(self.loaded[1])
        bad["engram.q_weight"] = bad["engram.q_weight"].float()
        with self.assertRaisesRegex(ValueError, "BF16"):
            Engram.from_weights(1, bad, self.lookup)

    @torch.inference_mode()
    def test_full_table_lookup_projection_and_image_mask_match_cpu_decoded_rows(self):
        for count in (1, 5, 6, 33):
            hashes, valid, hidden = self.rows(count, 42)
            for index, (layer, module) in enumerate(self.modules.items()):
                ids = hashes[:, index].contiguous()
                actual = module(hidden, ids, valid)
                expected = self.expected(module, hidden, ids, valid)
                self.equal(actual, expected, layer=layer, rows=count, graph=False)
                self.equal(
                    actual[~valid],
                    hidden[~valid],
                    layer=layer,
                    unchanged_image_rows=True,
                )

    @torch.inference_mode()
    def test_graph_reads_changed_canonical_hashes_and_masks_with_live_host_lease(self):
        stream = torch.cuda.Stream()
        for index, (layer, module) in enumerate(self.modules.items()):
            for count in (5, 6):
                hashes, valid, hidden = self.rows(count, 13)
                ids = hashes[:, index].contiguous()
                lookup_output = torch.empty(
                    count, 24, 256, dtype=torch.bfloat16, device="cuda"
                )
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        module(hidden, ids, valid, lookup_output=lookup_output)
                stream.synchronize()
                graph = self.lookup.graph()
                try:
                    with graph.capture(stream=stream):
                        actual = module(hidden, ids, valid, lookup_output=lookup_output)
                    for offset in (17, 97):
                        new_hashes, new_valid, new_hidden = self.rows(count, offset)
                        new_valid.logical_not_()
                        ids.copy_(new_hashes[:, index])
                        valid.copy_(new_valid)
                        hidden.copy_(new_hidden * 2)
                        graph.replay()
                        expected = self.expected(module, hidden, ids, valid)
                        self.equal(
                            actual,
                            expected,
                            layer=layer,
                            rows=count,
                            graph=True,
                            offset=offset,
                        )
                finally:
                    graph.close()


if __name__ == "__main__":
    unittest.main()
