"""Local head binding checks; distributed CP8 gather is an engine integration gate."""

import unittest
from types import SimpleNamespace

import test_transformer
import torch
import torch.nn.functional as F
from test_transformer import FixedProjection, RecordHasher, RecordSublayer

from rtp_llm.models_py.modules.dsv41.transformer import V41TargetModel
from rtp_llm.utils.model_weight import W


class TargetHeadBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_transformer.TargetComposerTest.setUpClass()
        cls.fixture = test_transformer.TargetComposerTest()

    def model(self, base, head, **metadata):
        return V41TargetModel(
            base.config,
            base.embedding,
            base.norm,
            head,
            base.blocks,
            {int(layer): value for layer, value in base.engrams.items()},
            base.token_hasher,
            **metadata,
        )

    def test_all_eight_checkpoint_shards_reconstruct_full_fp32_vocabulary(self):
        base, _, _ = self.fixture.make_model()
        # Integral, nonuniform weights make ordering and missing/duplicate shards
        # observable without introducing a new floating-point error threshold.
        columns = torch.arange(base.hidden_size, device=base.head.device)
        rows = torch.arange(base.config.text["vocab_size"], device=base.head.device)
        base.head.copy_(((rows[:, None] % 127) - 63) * (columns[None, :] % 3))
        hidden = (
            torch.arange(6 * base.hidden_size, device=base.head.device)
            .reshape(6, -1)
            .remainder(5)
            .bfloat16()
        )
        expected = F.linear(hidden.float(), base.head)
        shards = base.head.chunk(8, dim=0)
        outputs = []
        for rank, shard in enumerate(shards):
            with self.subTest(rank=rank):
                model = self.model(base, shard, head_tp_size=8, head_tp_rank=rank)
                self.assertEqual(model.head.data_ptr(), shard.data_ptr())
                self.assertEqual(model.embedding.data_ptr(), base.embedding.data_ptr())
                start, end = model.head_vocab_start, model.head_vocab_end
                self.assertEqual(start, rank * 16160)
                actual = model.local_logits(hidden)
                torch.testing.assert_close(
                    actual, expected[:, start:end], rtol=0, atol=0
                )
                self.assertEqual(actual.dtype, torch.float32)
                outputs.append(actual)
                with self.assertRaisesRegex(RuntimeError, "vocabulary gather"):
                    model.logits(hidden)
                with self.assertRaisesRegex(RuntimeError, "vocabulary gather"):
                    model.topk_logits(hidden, 3)
        torch.testing.assert_close(torch.cat(outputs, dim=-1), expected, rtol=0, atol=0)
        self.assertEqual(base.local_logits(hidden[:0]).shape, (0, 129280))

    def test_partition_metadata_cannot_be_inferred_from_a_small_tensor(self):
        base, _, _ = self.fixture.make_model()
        shard = base.head.chunk(8, dim=0)[0]
        with self.assertRaisesRegex(ValueError, "head shape"):
            self.model(base, shard)
        for size, rank in (
            (True, 0),
            (2, 0),
            (4, 0),
            (16, 0),
            (8, -1),
            (8, 8),
            (1, 1),
            (8, False),
        ):
            with self.subTest(size=size, rank=rank):
                with self.assertRaisesRegex(ValueError, "head partition"):
                    self.model(base, shard, head_tp_size=size, head_tp_rank=rank)
        with self.assertRaisesRegex(ValueError, "head shape"):
            self.model(base, base.head, head_tp_size=8, head_tp_rank=0)

    def test_fp32_head_rejects_cuda_autocast(self):
        base, _, _ = self.fixture.make_model()
        hidden = base.embedding[:1]
        shard = self.model(base, base.head.chunk(8)[0], head_tp_size=8, head_tp_rank=0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for model in (base, shard):
                with self.subTest(tp=model.head_tp_size), self.assertRaisesRegex(
                    RuntimeError, "autocast off"
                ):
                    model.local_logits(hidden)
            with self.assertRaisesRegex(RuntimeError, "autocast off"):
                base.logits(hidden)

    def test_installed_cp_head_binds_without_expanding_vocabulary_storage(self):
        base, _, lookup = self.fixture.make_model()
        layers = []
        for layer in range(40):
            local = {
                "v41." + name: value
                for name, value in self.fixture.weights(layer).items()
            }
            if layer in (1, 14):
                for name in ("q_weight", "k_weight"):
                    local["v41.engram." + name] = base.engrams[str(layer)].q_weight
                local["v41.engram.wkv.weight"] = torch.zeros(
                    160, 6144, device=base.head.device
                ).to(torch.float8_e4m3fn)
                local["v41.engram.wkv.scale"] = torch.ones(
                    5, 192, device=base.head.device
                ).to(torch.float8_e8m0fnu)
            layers.append(local)
        shard = base.head.chunk(8, dim=0)[7].clone()
        installed = SimpleNamespace(
            weights=layers,
            global_weights={
                W.embedding: base.embedding,
                W.final_ln_gamma: base.norm,
                W.lm_head: shard,
            },
        )
        bound = V41TargetModel.from_model_weights(
            base.config,
            installed,
            attention_factory=lambda layer, _: RecordSublayer(layer, "attention", []),
            moe_factory=lambda layer, _: RecordSublayer(layer, "moe", []),
            shared_lookup=lookup,
            token_hasher=RecordHasher(),
            projection_factory=lambda weight, scale: FixedProjection(32),
            head_tp_size=8,
            head_tp_rank=7,
        )
        self.assertEqual(bound.head.data_ptr(), shard.data_ptr())
        self.assertEqual(bound.head.untyped_storage().nbytes(), shard.numel() * 4)
        self.assertEqual(bound.head_vocab_start, 113120)
        self.assertEqual(bound.head_vocab_end, 129280)
        output = bound(self.fixture.rows(), object(), execution_mode="full")
        self.assertEqual(bound.local_logits(output.hidden_states).shape, (4, 16160))
        self.assertIs(bound.engrams["14"].shared_lookup, lookup)


if __name__ == "__main__":
    unittest.main()
