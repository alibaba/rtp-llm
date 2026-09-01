import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel


class MtpMultimodalTest(unittest.TestCase):
    def test_draft_fuses_image_embeddings_after_cp_slicing(self):
        model = DeepSeekV4MtpModel.__new__(DeepSeekV4MtpModel)
        torch.nn.Module.__init__(model)
        embed = torch.nn.Embedding.from_pretrained(
            torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )
        model.v4 = SimpleNamespace(embed=embed, layers=[], fp8_kv_cache=True, hc_mult=2)
        model._v4_args = SimpleNamespace(vocab_size=8, hc_mult=2, dim=4)
        model.kv_cache = object()
        model.parallelism_config = None
        model.enorm = torch.nn.Identity()
        model.hnorm = torch.nn.Identity()
        model.e_proj = torch.nn.Identity()
        model.h_proj = torch.nn.Identity()
        model._mtp_fusion_chunk_logged = True

        # Rank-local rows after the global next-token shift and CP slicing.
        # Image IDs are deliberately outside the vocabulary; row 3 is CP padding.
        inputs = SimpleNamespace(
            input_ids=torch.tensor([-100, 3, 1000, 0, 4]),
            input_hiddens=torch.arange(40, dtype=torch.float32).reshape(5, 8),
            multimodal_features=[torch.full((1, 4), 50.0), torch.full((1, 4), 60.0)],
            mm_features_locs=torch.tensor([0, 2], dtype=torch.int32),
            text_tokens_mask=torch.tensor([0, 1, 0, 1, 1]),
            attention_inputs=SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                context_parallel_info=SimpleNamespace(
                    prefill_shuffle_indices=torch.tensor([1, 2, 7, -1, 8])
                ),
            ),
        )
        positions = torch.tensor([0, 1, 6, 0, 7])
        embedding = torch.stack(
            [
                torch.zeros(4),
                embed.weight[3],
                torch.full((4,), 60.0),
                torch.zeros(4),
                embed.weight[4],
            ]
        )
        expected = inputs.input_hiddens.reshape(5, 2, 4) + embedding.unsqueeze(1)

        def prefill(*args, prepare_hidden_fn, **kwargs):
            return prepare_hidden_fn(inputs.input_ids, positions)

        for chunk_tokens in (100, 2):
            with self.subTest(chunk_tokens=chunk_tokens):
                current = SimpleNamespace(**vars(inputs))
                current.input_hiddens = inputs.input_hiddens.clone()
                with (
                    patch.object(
                        model, "_mtp_fusion_chunk_tokens", return_value=chunk_tokens
                    ),
                    patch("torch.cuda.is_current_stream_capturing", return_value=False),
                    patch(
                        "rtp_llm.models_py.model_desc.deepseek_v4_model.forward_prefill",
                        side_effect=prefill,
                    ),
                ):
                    actual = model.forward(current)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(
                    model.v4._image_token_mask.tolist(),
                    [True, False, True, False, False],
                )
                self.assertIsNone(model._cur_inputs)

    def test_other_draft_classes_still_reject_visual_features(self):
        class HiddenOnlyDraft(DeepSeekV4Model):
            pass

        model = HiddenOnlyDraft.__new__(HiddenOnlyDraft)
        torch.nn.Module.__init__(model)
        model.v4 = SimpleNamespace(layers=[])
        model.kv_cache = object()
        inputs = SimpleNamespace(
            multimodal_features=[torch.ones(1, 4)],
            attention_inputs=SimpleNamespace(is_prefill=True, is_target_verify=False),
        )
        with self.assertRaisesRegex(RuntimeError, "do not consume image features"):
            model.forward(inputs)


if __name__ == "__main__":
    unittest.main()
