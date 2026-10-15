"""K3 image embedding across TP/SP and native MTP request boundaries.

CPU cases run the real model methods, Embedding layout, feature injection and
MTP projection while substituting only lookup and communication. CUDA cases
exercise the real embedding kernel and text-only CUDA Graph capture. Their
same-GPU peer callback checks TP layout without claiming NCCL coverage.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.models_py.modules.base.common import embedding as embedding_module
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.ops import HybridAttentionType
from rtp_llm.ops.compute_ops import PyAttentionInputs


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


class KimiK3MultimodalEmbeddingCpuTest(unittest.TestCase):
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
            mm_related_params=SimpleNamespace(
                special_token_ids={"image_token_index": 2}
            ),
            k3_runtime_config=SimpleNamespace(media_placeholder_token_id=2),
        )
        model.multimodal_embedding_injector = MultimodalEmbeddingInjector()
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
            model.media_token_id = (
                model.config.k3_runtime_config.media_placeholder_token_id
            )
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
                    model.media_token_id = None
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
        model.media_token_id = None
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

    def test_target_custom_gather_keeps_full_width_embedding_and_image_order(self):
        for images in (False, True):
            for rank in range(2):
                with self.subTest(images=images, tp_rank=rank):
                    inputs = self._inputs(images)
                    if not images:
                        inputs.embedding_inputs.text_tokens_mask = None
                    original_ids = inputs.input_ids.clone()
                    model, boundary = self._model(KimiK3Model, rank)
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
                                "explicit TP gather must bypass the default collective"
                            ),
                        ),
                    ):
                        actual = model._embed_inputs(inputs, tp_gather=boundary.gather)
                    torch.testing.assert_close(
                        actual, self._expected_embedding(images), rtol=0, atol=0
                    )
                    self.assertEqual(boundary.gather_count, 1)
                    torch.testing.assert_close(
                        inputs.input_ids, original_ids, rtol=0, atol=0
                    )


class TextOnlyMultimodalInputs:
    multimodal_features = ()

    @property
    def mm_features_locs(self):
        raise AssertionError("text embedding must not read or copy MM locations")


class KimiK3MultimodalEmbeddingCudaTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "this target requires a CUDA GPU")
        self.weight = torch.arange(16 * 8, device="cuda", dtype=torch.bfloat16).reshape(
            16, 8
        )

    def make_model(self, cls=KimiK3Model, tp=1, rank=0):
        model = object.__new__(cls)
        nn.Module.__init__(model)
        model.tp_size, model.tp_rank = tp, rank
        model.embed_tokens = Embedding(
            SimpleNamespace(),
            SimpleNamespace(get_attn_tp_size=lambda: tp),
            self.weight,
        )
        model.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        model.kv_cache = None
        model.layers = nn.ModuleList()
        model.num_blocks = 0
        model.use_paged_conv_prefill = False
        if cls is KimiK3MtpModel:
            model.media_token_id = 11
        return model

    def inputs(self, ids, mask, features, locations):
        return SimpleNamespace(
            input_ids=torch.tensor(ids, device="cuda", dtype=torch.int32),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=(
                    None
                    if mask is None
                    else torch.tensor(mask, device="cuda", dtype=torch.int32)
                )
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=features,
                mm_features_locs=torch.tensor(
                    locations, device="cuda", dtype=torch.int32
                ),
            ),
            attention_inputs=PyAttentionInputs(),
        )

    def peer_gather(self, local, *args, **kwargs):
        # Embedding's TP output consists of hidden-dimension shards, gathered
        # as rank-major rows before its transpose/reshape to full hidden rows.
        return torch.cat((local, local + 100), dim=0)

    def expected(self, ids, features, locations, *, tp=1):
        safe_ids = ids.to(torch.long).clamp(0, self.weight.shape[0] - 1)
        local = self.weight.index_select(0, safe_ids)
        full = torch.cat((local, local + 100), dim=-1) if tp > 1 else local
        for feature, location in zip(features, locations):
            full[location : location + feature.shape[0]] = feature
        return full

    def test_target_masks_hashes_and_injects_after_tp_gather_before_sp_slice(self):
        feature = (
            torch.arange(2 * 16, device="cuda", dtype=torch.bfloat16).reshape(2, 16)
            + 500
        )
        inputs = self.inputs(
            [1, -(2**31), 2**31 - 1, 4, 5, 6, 7, 8],
            [1, 0, 0, 1, 1, 1, 1, 1],
            [feature],
            [1],
        )
        original_ids = inputs.input_ids.clone()
        full = self.expected(inputs.input_ids, [feature], [1], tp=2)
        for rank in (0, 1):
            model = self.make_model(tp=2, rank=rank)
            with patch(
                "rtp_llm.models_py.modules.base.common.embedding.all_gather",
                side_effect=self.peer_gather,
            ) as gather:
                actual = model._forward_layers(None, inputs, {}, gather_output=False)
            torch.testing.assert_close(
                actual, full.narrow(0, rank * 4, 4), rtol=0, atol=0
            )
            self.assertEqual(gather.call_count, 1)
        torch.testing.assert_close(inputs.input_ids, original_ids, rtol=0, atol=0)

    def test_mtp_restores_media_ids_from_shifted_mask_with_explicit_gather(self):
        # The C++ MTP producer already drops the first row of a feature at a
        # request boundary, and moves any other packed location left by one.
        # These are its resulting two four-token request windows.
        features = [
            torch.full((2, 16), 500, device="cuda", dtype=torch.bfloat16),
            torch.full((2, 16), 700, device="cuda", dtype=torch.bfloat16),
        ]
        inputs = self.inputs(
            [-(2**31), 2**31 - 1, 4, 5, 6, 2**31 - 1, -(2**31), 7],
            [0, 0, 1, 1, 1, 0, 0, 1],
            features,
            [0, 5],
        )
        inputs.combo_position_ids = torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3], device="cuda", dtype=torch.int32
        )
        inputs.input_hiddens = torch.zeros((8, 16), device="cuda", dtype=torch.bfloat16)
        original_ids = inputs.input_ids.clone()
        restored_ids = torch.tensor(
            [11, 11, 4, 5, 6, 11, 11, 7], device="cuda", dtype=torch.int32
        )
        full = self.expected(restored_ids, [], [], tp=2)
        full = full * (inputs.combo_position_ids[:, None] != 0)
        for rank in (0, 1):
            model = self.make_model(KimiK3MtpModel, tp=2, rank=rank)
            layer = nn.Module()
            layer.attention = SimpleNamespace(
                _mtp_bf16_collectives=SimpleNamespace(
                    all_gather_modeling_boundary=self.peer_gather
                )
            )
            model.layers = nn.ModuleList([layer])
            model.enorm = nn.Identity()
            model.hnorm = nn.Identity()
            model.eh_proj = nn.Linear(
                32, 16, bias=False, device="cuda", dtype=torch.bfloat16
            )
            with torch.no_grad():
                model.eh_proj.weight.zero_()
                model.eh_proj.weight[:, :16].copy_(
                    torch.eye(16, device="cuda", dtype=torch.bfloat16)
                )
            with (
                patch(
                    "rtp_llm.models_py.modules.base.common.embedding.all_gather",
                    side_effect=AssertionError(
                        "MTP must keep its explicit gather fast path"
                    ),
                ),
                patch.object(
                    model.multimodal_embedding_injector,
                    "forward",
                    side_effect=AssertionError(
                        "MTP must not inject external vision features"
                    ),
                ),
            ):
                actual = model._project_local_mtp_input(inputs)
            torch.testing.assert_close(
                actual, full.narrow(0, rank * 4, 4), rtol=0, atol=0
            )
        torch.testing.assert_close(inputs.input_ids, original_ids, rtol=0, atol=0)

    def test_text_embedding_keeps_graph_capture_without_multimodal_work(self):
        for cls in (KimiK3Model, KimiK3MtpModel):
            for mask in (None, []):
                with self.subTest(model=cls.__name__, mask=mask):
                    model = self.make_model(cls)
                    model.media_token_id = None
                    inputs = self.inputs([1, 2, 3, 4], mask, [], [])
                    inputs.multimodal_inputs = TextOnlyMultimodalInputs()
                    with (
                        patch.object(
                            model.multimodal_embedding_injector,
                            "forward",
                            side_effect=AssertionError(
                                "text must bypass the multimodal injector"
                            ),
                        ),
                        patch(
                            "torch.where",
                            side_effect=AssertionError(
                                "text must not restore media IDs"
                            ),
                        ),
                    ):
                        warmup = torch.cuda.Stream()
                        warmup.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(warmup):
                            for _ in range(3):
                                model._embed_inputs(inputs)
                        torch.cuda.current_stream().wait_stream(warmup)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = model._embed_inputs(inputs)
                        graph.replay()
                    torch.testing.assert_close(
                        actual,
                        self.weight.index_select(0, inputs.input_ids.to(torch.long)),
                        rtol=0,
                        atol=0,
                    )

    def test_multimodal_without_mask_fails_before_embedding_kernel(self):
        for cls in (KimiK3Model, KimiK3MtpModel):
            for mask in (None, [], [0], [[0, 1]]):
                with self.subTest(model=cls.__name__, mask=mask):
                    model = self.make_model(cls)
                    inputs = self.inputs(
                        [2**31 - 1, 1],
                        mask,
                        [torch.ones((1, 8), device="cuda", dtype=torch.bfloat16)],
                        [0],
                    )
                    with patch.object(model.embed_tokens, "forward") as embedding:
                        with self.assertRaisesRegex(ValueError, "text.*mask"):
                            model._embed_inputs(inputs)
                        embedding.assert_not_called()

    def test_mtp_missing_media_id_fails_only_for_multimodal_input(self):
        model = self.make_model(KimiK3MtpModel)
        model.media_token_id = None
        inputs = self.inputs(
            [2**31 - 1, 1],
            [0, 1],
            [torch.ones((1, 8), device="cuda", dtype=torch.bfloat16)],
            [0],
        )
        with patch.object(model.embed_tokens, "forward") as embedding:
            with self.assertRaisesRegex(ValueError, "media_placeholder_token_id"):
                model._embed_inputs(inputs)
            embedding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
