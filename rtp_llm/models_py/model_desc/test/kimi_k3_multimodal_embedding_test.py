"""CPU coverage of the real K3 embedding, SP and native MTP boundaries.

Only CUDA lookup and distributed communication are replaced. The real model
methods, Embedding.forward, feature injector and MTP normalization/projection
run on CPU. The target consumes image features; native MTP restores the media
token in already-shifted request inputs without looking up feature hashes.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.modules.base.common import embedding as embedding_module
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.ops import HybridAttentionType


class _IdentityLayer(nn.Module):
    layer_type = HybridAttentionType.NONE

    def forward(self, hidden, *args):
        return hidden


class _CpuEmbeddingBoundary:
    """Emulate sharded lookup/gather without indexing any unmasked hash id."""

    def __init__(self, table, rank):
        self.table = table
        self.rank = rank
        self.masks = []
        self.lookup_ids = []
        self.gather_count = 0

    def lookup(self, output, ids, weight, positions, types, mask):
        if mask is not None and mask.numel() != ids.numel():
            raise AssertionError(
                "invalid text-mask length reached the vocabulary lookup"
            )
        self.masks.append(mask)
        self.lookup_ids.append(ids.detach().clone())
        valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
        invalid = ((ids < 0) | (ids >= weight.shape[0])) & valid
        if invalid.any():
            raise AssertionError("image hash ids reached an unmasked vocabulary lookup")
        output.zero_()
        output[valid] = weight[ids[valid].long()]

    def gather(self, local, *args, **kwargs):
        self.gather_count += 1
        mask = self.masks[-1]
        ids = self.lookup_ids[-1]
        valid = torch.ones_like(ids, dtype=torch.bool)
        if mask is not None:
            valid = mask.bool()
        full = torch.zeros((len(valid), self.table.shape[1]), dtype=self.table.dtype)
        full[valid] = self.table[ids[valid].long()]
        shards = full.chunk(2, dim=1)
        torch.testing.assert_close(local, shards[self.rank], rtol=0, atol=0)
        return torch.cat(shards, dim=0)


class KimiK3MultimodalEmbeddingTest(unittest.TestCase):
    def _inputs(self, images):
        if images:
            ids = [0, 1, 2, -1700000000, 2000000000, 3, -1900000000, 0]
            mask = [1, 1, 1, 0, 0, 1, 0, 1]
            features = [
                torch.tensor([[100.0, -30.0, 7.0, 12.0], [-8.0, 23.0, 5.0, -100.0]]),
                torch.tensor([[9.0, 0.0, 71.0, -11.0]]),
            ]
            locs = [3, 6]
        else:
            ids = [0, 1, 2, 3, 0, 1, 2, 3]
            mask = [1] * 8
            features, locs = [], []
        return SimpleNamespace(
            input_ids=torch.tensor(ids, dtype=torch.int32),
            combo_position_ids=torch.arange(8, dtype=torch.int32),
            input_hiddens=torch.full((8, 4), 0.5),
            embedding_inputs=SimpleNamespace(
                combo_tokens_type_ids=None,
                text_tokens_mask=torch.tensor(mask, dtype=torch.int32),
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=features,
                mm_features_locs=torch.tensor(locs, dtype=torch.int32),
            ),
            attention_inputs={
                "default": SimpleNamespace(
                    is_prefill=False,
                    is_target_verify=False,
                    valid_token_mask=None,
                )
            },
        )

    def _shifted_packed_inputs(self):
        inputs = self._inputs(images=True)
        # Two requests of length 3 and 5, after the native C++ per-request
        # shift. The second image now starts at the second request's first
        # token. Shifting its location again would overwrite request 1's tail.
        inputs.input_ids = torch.tensor(
            [-1700000000, 1, 3, 2000000000, -1900000000, 2, 0, 1],
            dtype=torch.int32,
        )
        inputs.combo_position_ids = torch.tensor(
            [1, 2, 2, 1, 2, 3, 4, 4], dtype=torch.int32
        )
        inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [0, 1, 1, 0, 0, 1, 1, 1], dtype=torch.int32
        )
        inputs.multimodal_inputs.multimodal_features = [
            torch.tensor([[9.0, 0.0, 71.0, -11.0]]),
            torch.tensor([[100.0, -30.0, 7.0, 12.0], [-8.0, 23.0, 5.0, -100.0]]),
        ]
        inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [0, 3], dtype=torch.int32
        )
        return inputs

    def _model(self, cls, rank):
        model = object.__new__(cls)
        nn.Module.__init__(model)
        model.tp_size, model.tp_rank = 2, rank
        model.num_blocks = 0
        model.use_paged_conv_prefill = False
        model.kv_cache = None
        model.layers = nn.ModuleList([_IdentityLayer()])
        # Native MTP disables ViT construction and uses the media token for
        # image spans, while the target consumes the request's real features.
        model.config = SimpleNamespace(
            mm_model_config=SimpleNamespace(is_multimodal=False),
            mm_related_params=SimpleNamespace(special_token_ids={"image_token_index": 2}),
        )
        # Distinct directions survive LayerNorm: a feature injection cannot
        # accidentally equal E(media) merely because both rows differ by a bias.
        table = torch.tensor(
            [
                [0.0, 1.0, 4.0, -2.0],
                [3.0, -1.0, 2.0, 5.0],
                [-4.0, 2.0, 7.0, 1.0],
                [6.0, 0.0, -3.0, 2.0],
            ]
        )
        boundary = _CpuEmbeddingBoundary(table, rank)
        parallel = SimpleNamespace(get_attn_tp_size=lambda: 2)
        model.embed_tokens = Embedding(
            model.config, parallel, table[:, rank * 2 : rank * 2 + 2]
        )
        model.layers[0].attention = SimpleNamespace(
            _mtp_bf16_collectives=SimpleNamespace(
                all_gather_modeling_boundary=boundary.gather
            )
        )
        if cls is KimiK3MtpModel:
            model.enorm = nn.LayerNorm(4)
            model.hnorm = nn.Identity()
            model.eh_proj = nn.Linear(8, 4, bias=False)
            with torch.no_grad():
                model.eh_proj.weight.copy_(
                    torch.cat((torch.eye(4), 2 * torch.eye(4)), dim=1)
                )
        return model, boundary

    def _expected_embedding(self, images):
        if images:
            return torch.tensor(
                [
                    [0.0, 1.0, 4.0, -2.0],
                    [3.0, -1.0, 2.0, 5.0],
                    [-4.0, 2.0, 7.0, 1.0],
                    [100.0, -30.0, 7.0, 12.0],
                    [-8.0, 23.0, 5.0, -100.0],
                    [6.0, 0.0, -3.0, 2.0],
                    [9.0, 0.0, 71.0, -11.0],
                    [0.0, 1.0, 4.0, -2.0],
                ]
            )
        return torch.tensor(
            [
                [0.0, 1.0, 4.0, -2.0],
                [3.0, -1.0, 2.0, 5.0],
                [-4.0, 2.0, 7.0, 1.0],
                [6.0, 0.0, -3.0, 2.0],
                [0.0, 1.0, 4.0, -2.0],
                [3.0, -1.0, 2.0, 5.0],
                [-4.0, 2.0, 7.0, 1.0],
                [6.0, 0.0, -3.0, 2.0],
            ]
        )

    def _run_target(self, images):
        for rank in range(2):
            with self.subTest(tp_rank=rank):
                inputs = self._inputs(images)
                original_ids = inputs.input_ids.clone()
                if not images:
                    inputs.embedding_inputs.text_tokens_mask = None
                model, boundary = self._model(KimiK3Model, rank)
                with (
                    patch.object(
                        embedding_module.rtp_llm_ops, "embedding", new=boundary.lookup
                    ),
                    patch.object(embedding_module, "all_gather", new=boundary.gather),
                ):
                    actual = model._forward_layers(
                        None, inputs, fmha_impl=object(), gather_output=False
                    )
                expected = self._expected_embedding(images)[rank * 4 : rank * 4 + 4]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(boundary.gather_count, 1)
                torch.testing.assert_close(
                    boundary.lookup_ids[0], original_ids, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    inputs.input_ids, original_ids, rtol=0, atol=0
                )
                if images:
                    self.assertIs(
                        boundary.masks[0], inputs.embedding_inputs.text_tokens_mask
                    )
                else:
                    self.assertIsNone(boundary.masks[0])

    def _run_mtp(self, images, *, packed=False):
        for rank in range(2):
            with self.subTest(tp_rank=rank):
                inputs = (
                    self._shifted_packed_inputs() if packed else self._inputs(images)
                )
                original_ids = inputs.input_ids.clone()
                if not images:
                    inputs.embedding_inputs.text_tokens_mask = None
                model, boundary = self._model(KimiK3MtpModel, rank)
                if not images:
                    model.config.mm_related_params.special_token_ids.clear()
                enorm_inputs = []
                hook = model.enorm.register_forward_pre_hook(
                    lambda module, args: enorm_inputs.append(args[0].detach().clone())
                )
                try:
                    with (
                        patch.object(
                            embedding_module.rtp_llm_ops,
                            "embedding",
                            new=boundary.lookup,
                        ),
                        patch.object(
                            embedding_module,
                            "all_gather",
                            side_effect=AssertionError(
                                "MTP must keep its modeling-boundary gather"
                            ),
                        ),
                    ):
                        actual = model._project_local_mtp_input(inputs)
                finally:
                    hook.remove()
                if packed:
                    lookup_ids = [2, 1, 3, 2, 2, 2, 0, 1]
                    full = torch.tensor(
                        [
                            [-4.0, 2.0, 7.0, 1.0],
                            [3.0, -1.0, 2.0, 5.0],
                            [6.0, 0.0, -3.0, 2.0],
                            [-4.0, 2.0, 7.0, 1.0],
                            [-4.0, 2.0, 7.0, 1.0],
                            [-4.0, 2.0, 7.0, 1.0],
                            [0.0, 1.0, 4.0, -2.0],
                            [3.0, -1.0, 2.0, 5.0],
                        ]
                    )
                elif images:
                    lookup_ids = [0, 1, 2, 2, 2, 3, 2, 0]
                    full = torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [3.0, -1.0, 2.0, 5.0],
                            [-4.0, 2.0, 7.0, 1.0],
                            [-4.0, 2.0, 7.0, 1.0],
                            [-4.0, 2.0, 7.0, 1.0],
                            [6.0, 0.0, -3.0, 2.0],
                            [-4.0, 2.0, 7.0, 1.0],
                            [0.0, 1.0, 4.0, -2.0],
                        ]
                    )
                else:
                    lookup_ids = [0, 1, 2, 3, 0, 1, 2, 3]
                    full = self._expected_embedding(images=False)
                    full[0].zero_()  # Position zero has no previous token.
                expected_local = full[rank * 4 : rank * 4 + 4]
                torch.testing.assert_close(
                    enorm_inputs[0], expected_local, rtol=0, atol=0
                )
                expected = F.layer_norm(expected_local, (4,)) + 1.0
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(boundary.gather_count, 1)
                self.assertIsNone(boundary.masks[0])
                torch.testing.assert_close(
                    boundary.lookup_ids[0],
                    torch.tensor(lookup_ids, dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    inputs.input_ids, original_ids, rtol=0, atol=0
                )

    def test_target_injects_images_before_sequence_parallel_slicing(self):
        self._run_target(images=True)

    def test_native_mtp_restores_media_token_before_enorm_with_text_only_model_flag(
        self,
    ):
        self._run_mtp(images=True)

    def test_native_mtp_uses_already_shifted_packed_request_mask(self):
        self._run_mtp(images=True, packed=True)

    def test_images_require_complete_mask_before_vocabulary_lookup(self):
        for cls in (KimiK3Model, KimiK3MtpModel):
            for mask in (
                None,
                torch.ones(7, dtype=torch.int32),
                torch.ones((2, 4), dtype=torch.int32),
            ):
                with self.subTest(
                    model=cls.__name__,
                    mask_shape=None if mask is None else tuple(mask.shape),
                ):
                    inputs = self._inputs(images=True)
                    inputs.embedding_inputs.text_tokens_mask = mask
                    model, boundary = self._model(cls, 0)
                    with (
                        patch.object(
                            embedding_module.rtp_llm_ops,
                            "embedding",
                            new=boundary.lookup,
                        ),
                        patch.object(
                            embedding_module, "all_gather", new=boundary.gather
                        ),
                    ):
                        with self.assertRaises(ValueError):
                            if cls is KimiK3Model:
                                model._forward_layers(
                                    None,
                                    inputs,
                                    fmha_impl=object(),
                                    gather_output=False,
                                )
                            else:
                                model._project_local_mtp_input(inputs)
                    self.assertEqual(boundary.masks, [])
                    self.assertEqual(boundary.gather_count, 0)

    def test_native_mtp_requires_media_token_before_vocabulary_lookup(self):
        inputs = self._inputs(images=True)
        original_ids = inputs.input_ids.clone()
        model, boundary = self._model(KimiK3MtpModel, 0)
        model.config.mm_related_params.special_token_ids.clear()
        with patch.object(
            embedding_module.rtp_llm_ops, "embedding", new=boundary.lookup
        ):
            with self.assertRaises(ValueError):
                model._project_local_mtp_input(inputs)
        self.assertEqual(boundary.lookup_ids, [])
        self.assertEqual(boundary.gather_count, 0)
        torch.testing.assert_close(inputs.input_ids, original_ids, rtol=0, atol=0)

    def test_target_text_keeps_unmasked_embedding_fast_path(self):
        self._run_target(images=False)

    def test_native_mtp_text_keeps_modeling_boundary_gather(self):
        self._run_mtp(images=False)


if __name__ == "__main__":
    unittest.main()
