import tempfile
import unittest
import weakref
from types import SimpleNamespace

import torch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.utils.model_weight import W


def _make_model(model_type, ckpt_path, embedding, lm_head, mtp_layer_offset=0):
    config = SimpleNamespace(
        model_type=model_type,
        ckpt_path=ckpt_path,
        mtp_layer_offset=mtp_layer_offset,
    )
    weight = SimpleNamespace(
        global_weights={W.embedding: embedding, W.lm_head: lm_head}
    )
    py_model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=embedding))
    return SimpleNamespace(
        model_config=config, weight=weight, py_model=py_model, merge_lora=False
    )


class GlmMtpGlobalWeightShareTest(unittest.TestCase):
    def setUp(self):
        self.ckpt_dir = tempfile.TemporaryDirectory()
        self.target_embedding = torch.zeros((8, 4), dtype=torch.bfloat16)
        self.target_lm_head = torch.ones((8, 4), dtype=torch.bfloat16)
        self.propose_embedding = torch.full((8, 4), 2, dtype=torch.bfloat16)
        self.propose_lm_head = torch.full((8, 4), 3, dtype=torch.bfloat16)
        self.target = _make_model(
            "glm_5",
            self.ckpt_dir.name,
            self.target_embedding,
            self.target_lm_head,
        )

    def tearDown(self):
        self.ckpt_dir.cleanup()

    def _make_propose(self, **overrides):
        values = {
            "model_type": "glm_5_mtp",
            "ckpt_path": self.ckpt_dir.name,
            "embedding": self.propose_embedding,
            "lm_head": self.propose_lm_head,
            "mtp_layer_offset": 78,
        }
        values.update(overrides)
        return _make_model(**values)

    def test_shares_both_globals_and_python_embedding(self):
        propose = self._make_propose()

        self.assertTrue(
            ModelFactory._share_glm5_mtp_global_weights(self.target, propose)
        )
        self.assertIs(propose.weight.global_weights[W.embedding], self.target_embedding)
        self.assertIs(propose.weight.global_weights[W.lm_head], self.target_lm_head)
        self.assertIs(propose.py_model.embed_tokens.weight, self.target_embedding)

    def test_incompatible_weight_is_transactional(self):
        propose = self._make_propose(lm_head=torch.empty((9, 4), dtype=torch.bfloat16))
        old_embedding = propose.weight.global_weights[W.embedding]
        old_lm_head = propose.weight.global_weights[W.lm_head]

        self.assertFalse(
            ModelFactory._share_glm5_mtp_global_weights(self.target, propose)
        )
        self.assertIs(propose.weight.global_weights[W.embedding], old_embedding)
        self.assertIs(propose.weight.global_weights[W.lm_head], old_lm_head)
        self.assertIs(propose.py_model.embed_tokens.weight, old_embedding)

    def test_standalone_mtp_layout_is_not_shared(self):
        propose = self._make_propose(mtp_layer_offset=0)

        self.assertFalse(
            ModelFactory._share_glm5_mtp_global_weights(self.target, propose)
        )
        self.assertIs(
            propose.weight.global_weights[W.embedding], self.propose_embedding
        )

    def test_different_checkpoint_is_not_shared(self):
        with tempfile.TemporaryDirectory() as other_ckpt:
            propose = self._make_propose(ckpt_path=other_ckpt)

            self.assertFalse(
                ModelFactory._share_glm5_mtp_global_weights(self.target, propose)
            )
            self.assertIs(
                propose.weight.global_weights[W.embedding], self.propose_embedding
            )

    def test_unexpected_python_embedding_reference_is_not_shared(self):
        propose = self._make_propose()
        unexpected = torch.empty_like(self.propose_embedding)
        propose.py_model.embed_tokens.weight = unexpected

        self.assertFalse(
            ModelFactory._share_glm5_mtp_global_weights(self.target, propose)
        )
        self.assertIs(
            propose.weight.global_weights[W.embedding], self.propose_embedding
        )
        self.assertIs(propose.py_model.embed_tokens.weight, unexpected)

    def test_merged_lora_target_is_not_shared(self):
        propose = self._make_propose()
        self.target.merge_lora = True

        self.assertFalse(
            ModelFactory._share_glm5_mtp_global_weights(self.target, propose)
        )
        self.assertIs(
            propose.weight.global_weights[W.embedding], self.propose_embedding
        )

    def test_old_propose_storages_are_released(self):
        embedding = torch.empty_like(self.propose_embedding)
        lm_head = torch.empty_like(self.propose_lm_head)
        embedding_ref = weakref.ref(embedding)
        lm_head_ref = weakref.ref(lm_head)
        propose = self._make_propose(embedding=embedding, lm_head=lm_head)
        del embedding, lm_head

        self.assertTrue(
            ModelFactory._share_glm5_mtp_global_weights(self.target, propose)
        )
        self.assertIsNone(embedding_ref())
        self.assertIsNone(lm_head_ref())


if __name__ == "__main__":
    unittest.main()
