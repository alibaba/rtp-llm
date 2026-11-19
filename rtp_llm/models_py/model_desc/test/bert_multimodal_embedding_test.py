from types import SimpleNamespace
from unittest import SkipTest, TestCase, main, mock

import torch
from torch.nn import functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.ops import ActivationType, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta
from rtp_llm.utils.model_weight import W

_ZERO_LAYER_HIDDEN_SIZE = 4
_VOCAB_SIZE = 8
_ONE_LAYER_HIDDEN_SIZE = 128


class _UnusedFmhaImpl:
    """Fail explicitly if a zero-layer test unexpectedly reaches attention."""

    def forward(self, *_args, **_kwargs):
        raise AssertionError("zero-layer BertModel must not invoke FMHA")


_FMHA_NOT_USED = _UnusedFmhaImpl()


def _build_word_embedding(
    *,
    dtype: torch.dtype = torch.float16,
    hidden_size: int = _ZERO_LAYER_HIDDEN_SIZE,
) -> torch.Tensor:
    # Keep every row distinguishable after BF16 conversion. A flat low-precision
    # arange loses unit increments at large hidden sizes and can let a wrong-row
    # lookup satisfy the oracle.
    rows = torch.arange(_VOCAB_SIZE, dtype=torch.float32, device="cuda").unsqueeze(1)
    columns = (
        torch.arange(hidden_size, dtype=torch.float32, device="cuda") % 16
    ).unsqueeze(0)
    return (rows * 0.5 + columns / 64).to(dtype)


def _build_inputs(
    *,
    with_text_tokens_mask: bool = True,
    dtype: torch.dtype = torch.float16,
    hidden_size: int = _ZERO_LAYER_HIDDEN_SIZE,
) -> PyModelInputs:
    inputs = PyModelInputs()
    inputs.input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
        [0, 1, 2, 3], dtype=torch.int32, device="cuda"
    )
    inputs.bert_embedding_inputs.position_encoding = torch.zeros(
        (4, hidden_size), dtype=dtype, device="cuda"
    )
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.zeros(
        4, dtype=torch.int32, device="cuda"
    )
    inputs.bert_embedding_inputs.token_type_embedding = torch.zeros(
        (1, hidden_size), dtype=dtype, device="cuda"
    )
    if with_text_tokens_mask:
        inputs.embedding_inputs.text_tokens_mask = torch.ones(
            4, dtype=torch.int32, device="cuda"
        )
    return inputs


class BertMultimodalEmbeddingTest(TestCase):
    """Covers multimodal injection placement and values without decoder layers."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def _build_model(
        self,
        *,
        dtype: torch.dtype = torch.float16,
        hidden_size: int = _ZERO_LAYER_HIDDEN_SIZE,
    ) -> BertModel:
        config = ModelConfig()
        config.num_layers = 0
        config.hidden_size = hidden_size
        config.vocab_size = _VOCAB_SIZE

        weights = ModelWeights(0, "cuda", dtype)
        weights.set_global_weight(
            W.embedding,
            _build_word_embedding(dtype=dtype, hidden_size=hidden_size),
        )
        weights.set_global_weight(
            W.pre_decoder_ln_gamma,
            torch.ones(hidden_size, dtype=dtype, device="cuda"),
        )
        weights.set_global_weight(
            W.pre_decoder_ln_beta,
            torch.zeros(hidden_size, dtype=dtype, device="cuda"),
        )
        model = BertModel(config, ParallelismConfig(), weights, 1)
        self.assertEqual(len(model.layers), 0)
        return model

    def _build_one_layer_model(self) -> BertModel:
        hidden_size = _ONE_LAYER_HIDDEN_SIZE
        config = ModelConfig()
        config.num_layers = 1
        config.hidden_size = hidden_size
        config.vocab_size = _VOCAB_SIZE
        config.inter_size = hidden_size * 2
        config.activation_type = ActivationType.Gelu
        config.attn_config.head_num = 1
        config.attn_config.kv_head_num = 1
        config.attn_config.size_per_head = hidden_size
        config.attn_config.tokens_per_block = 64
        config.attn_config.kernel_tokens_per_block = 64
        config.use_kvcache = False

        generator = torch.Generator(device="cuda").manual_seed(7)
        weights = ModelWeights(1, "cuda", torch.float16)
        weights.set_global_weight(
            W.embedding,
            torch.randn(
                (config.vocab_size, hidden_size),
                generator=generator,
                dtype=torch.float16,
                device="cuda",
            ),
        )
        weights.set_global_weight(
            W.pre_decoder_ln_gamma,
            torch.ones(hidden_size, dtype=torch.float16, device="cuda"),
        )
        weights.set_global_weight(
            W.pre_decoder_ln_beta,
            torch.zeros(hidden_size, dtype=torch.float16, device="cuda"),
        )
        layer_weights = {
            W.attn_qkv_w: torch.randn(
                (hidden_size, hidden_size * 3),
                generator=generator,
                dtype=torch.float16,
                device="cuda",
            )
            * 0.02,
            W.attn_o_w: torch.randn(
                (hidden_size, hidden_size),
                generator=generator,
                dtype=torch.float16,
                device="cuda",
            )
            * 0.02,
            W.ffn_w3: torch.randn(
                (hidden_size, config.inter_size),
                generator=generator,
                dtype=torch.float16,
                device="cuda",
            )
            * 0.02,
            W.ffn_w2: torch.randn(
                (config.inter_size, hidden_size),
                generator=generator,
                dtype=torch.float16,
                device="cuda",
            )
            * 0.02,
            W.post_ln_gamma: torch.ones(
                hidden_size, dtype=torch.float16, device="cuda"
            ),
            W.post_ln_beta: torch.zeros(
                hidden_size, dtype=torch.float16, device="cuda"
            ),
            W.post_ffn_ln_gamma: torch.ones(
                hidden_size, dtype=torch.float16, device="cuda"
            ),
            W.post_ffn_ln_beta: torch.zeros(
                hidden_size, dtype=torch.float16, device="cuda"
            ),
        }
        for name, tensor in layer_weights.items():
            weights.set_layer_weight(0, name, tensor)
        return BertModel(config, ParallelismConfig(), weights, 1)

    def _build_one_layer_inputs(self) -> PyModelInputs:
        hidden_size = _ONE_LAYER_HIDDEN_SIZE
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        inputs.bert_embedding_inputs.combo_position_ids = torch.arange(
            4, dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.position_encoding = torch.zeros(
            (4, hidden_size), dtype=torch.float16, device="cuda"
        )
        inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.zeros(
            4, dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.token_type_embedding = torch.zeros(
            (1, hidden_size), dtype=torch.float16, device="cuda"
        )
        attention_inputs = PyAttentionInputs()
        attention_inputs.is_prefill = True
        attention_inputs.input_lengths = torch.tensor(
            [4], dtype=torch.int32
        ).pin_memory()
        attention_inputs.sequence_lengths = torch.empty(
            0, dtype=torch.int32
        ).pin_memory()
        attention_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32).pin_memory()
        attention_inputs.cu_seqlens_device = torch.tensor(
            [0, 4], dtype=torch.int32, device="cuda"
        )
        attention_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=torch.float16, device="cuda")
        )
        inputs.attention_inputs = attention_inputs
        return inputs

    def test_forward_splices_post_layernorm_features_and_preserves_text(self):
        model = self._build_model()
        baseline_inputs = _build_inputs(with_text_tokens_mask=False)
        baseline = model.forward(
            baseline_inputs, fmha_impl=_FMHA_NOT_USED
        ).hidden_states

        multimodal_inputs = _build_inputs()
        first_feature = torch.tensor(
            [[9.0, 8.0, 7.0, 6.0]], dtype=torch.float16, device="cuda"
        )
        last_feature = torch.tensor(
            [[6.0, 7.0, 8.0, 9.0], [5.0, 4.0, 3.0, 2.0]],
            dtype=torch.float16,
            device="cuda",
        )
        multimodal_inputs.multimodal_inputs.multimodal_features = [
            first_feature,
            last_feature,
        ]
        multimodal_inputs.input_ids[0] = 123456
        multimodal_inputs.input_ids[2:] = torch.tensor(
            [-3, 741852], dtype=torch.int32, device="cuda"
        )
        multimodal_inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.tensor(
            [-1, 0, -3, 741852], dtype=torch.int32, device="cuda"
        )
        multimodal_inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [0, 1, 0, 0], dtype=torch.int32, device="cuda"
        )
        multimodal_inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [0, 2], dtype=torch.int32, device="cuda"
        )
        output = model.forward(
            multimodal_inputs, fmha_impl=_FMHA_NOT_USED
        ).hidden_states

        torch.testing.assert_close(output[0:1], first_feature)
        torch.testing.assert_close(output[2:4], last_feature)
        torch.testing.assert_close(output[1:2], baseline[1:2])

    def test_multimodal_producer_contract_matches_full_reference_oracle(self):
        """Check text and producer-owned image rows against independent formulas."""
        model = self._build_model()
        inputs = _build_inputs()

        word_embedding = _build_word_embedding()
        position_encoding = torch.tensor(
            [
                [0.5, -0.5, 1.0, -1.0],
                [1.5, 0.25, -0.75, 0.5],
                [-0.25, 1.25, 0.75, -1.5],
                [2.0, -1.0, 0.5, 1.5],
            ],
            dtype=torch.float16,
            device="cuda",
        )
        token_type_embedding = torch.tensor(
            [
                [0.125, -0.25, 0.5, -0.75],
                [1.0, 0.75, -0.5, -0.25],
            ],
            dtype=torch.float16,
            device="cuda",
        )
        text_ln_weight = torch.tensor(
            [1.25, 0.75, 1.5, 0.5], dtype=torch.float16, device="cuda"
        )
        text_ln_bias = torch.tensor(
            [0.2, -0.3, 0.4, -0.1], dtype=torch.float16, device="cuda"
        )
        text_ln_eps = 1e-5
        input_embedding_scalar = 0.5

        model.pre_decoder_layernorm.weight.copy_(text_ln_weight)
        model.pre_decoder_layernorm.beta.copy_(text_ln_bias)
        model.pre_decoder_layernorm.variance_epsilon = text_ln_eps
        inputs.input_ids = torch.tensor(
            [1, 123456, 3, 4], dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
            [3, 2, 1, 0], dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.position_encoding = position_encoding
        inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.tensor(
            [1, 0, 1, 0], dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding
        inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar
        inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [1, 0, 1, 1], dtype=torch.int32, device="cuda"
        )

        vision_input = torch.tensor(
            [[0.25, -0.5, 1.5]], dtype=torch.float32, device="cuda"
        )
        projector_weight = torch.tensor(
            [
                [1.0, -0.5, 0.25],
                [0.0, 0.5, 1.0],
                [-1.0, 0.25, 0.5],
                [0.75, 0.0, -0.25],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        projector_bias = torch.tensor(
            [0.1, -0.2, 0.3, -0.4], dtype=torch.float32, device="cuda"
        )
        projector_ln_weight = torch.tensor(
            [0.5, 1.5, 2.0, 0.75], dtype=torch.float32, device="cuda"
        )
        projector_ln_bias = torch.tensor(
            [0.2, -0.1, 0.4, -0.3], dtype=torch.float32, device="cuda"
        )
        projector_ln_eps = 1e-5
        projected_feature = F.layer_norm(
            F.linear(vision_input, projector_weight, projector_bias),
            (_ZERO_LAYER_HIDDEN_SIZE,),
            projector_ln_weight,
            projector_ln_bias,
            projector_ln_eps,
        ).to(torch.float16)

        inputs.multimodal_inputs.multimodal_features = [projected_feature]
        inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [1], dtype=torch.int32, device="cuda"
        )

        actual = model.forward(inputs, fmha_impl=_FMHA_NOT_USED).hidden_states

        text_rows = torch.tensor([0, 2, 3], dtype=torch.long, device="cuda")
        text_input_ids = inputs.input_ids.index_select(0, text_rows).long()
        text_position_ids = (
            inputs.bert_embedding_inputs.combo_position_ids.index_select(
                0, text_rows
            ).long()
        )
        text_token_type_ids = (
            inputs.bert_embedding_inputs.combo_tokens_type_ids.index_select(
                0, text_rows
            ).long()
        )
        text_pre_ln = (
            word_embedding.index_select(0, text_input_ids) * input_embedding_scalar
            + position_encoding.index_select(0, text_position_ids)
            + token_type_embedding.index_select(0, text_token_type_ids)
        )
        expected_text = F.layer_norm(
            text_pre_ln.float(),
            (_ZERO_LAYER_HIDDEN_SIZE,),
            text_ln_weight.float(),
            text_ln_bias.float(),
            text_ln_eps,
        ).to(torch.float16)
        expected = torch.empty_like(actual)
        expected.index_copy_(0, text_rows, expected_text)
        expected[1:2].copy_(projected_feature)

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

        image_position_and_type = position_encoding[2:3] + token_type_embedding[0:1]
        image_with_text_components = projected_feature + image_position_and_type
        self.assertFalse(torch.allclose(actual[1:2], image_with_text_components))

        doubly_normalized = F.layer_norm(
            projected_feature.float(),
            (_ZERO_LAYER_HIDDEN_SIZE,),
            text_ln_weight.float(),
            text_ln_bias.float(),
            text_ln_eps,
        ).to(torch.float16)
        self.assertFalse(torch.allclose(actual[1:2], doubly_normalized))

    def test_post_layernorm_features_flow_through_real_decoder_and_fmha(self):
        if torch.version.hip is not None:
            self.skipTest("real Bert FMHA coverage is CUDA-only")
        model = self._build_one_layer_model()
        baseline_inputs = self._build_one_layer_inputs()
        multimodal_inputs = self._build_one_layer_inputs()
        multimodal_inputs.embedding_inputs.text_tokens_mask = torch.ones(
            4, dtype=torch.int32, device="cuda"
        )
        feature = torch.linspace(
            -1.0, 1.0, _ONE_LAYER_HIDDEN_SIZE, dtype=torch.float16, device="cuda"
        ).reshape(1, _ONE_LAYER_HIDDEN_SIZE)
        multimodal_inputs.input_ids[1] = 123456
        multimodal_inputs.embedding_inputs.text_tokens_mask[1] = 0
        multimodal_inputs.multimodal_inputs.multimodal_features = [feature]
        multimodal_inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [1], dtype=torch.int32, device="cuda"
        )

        decoder_inputs = []
        hook = model.layers[0].register_forward_pre_hook(
            lambda _module, args: decoder_inputs.append(args[0].detach().clone())
        )
        try:
            baseline = model.forward(baseline_inputs).hidden_states
            actual = model.forward(multimodal_inputs).hidden_states
        finally:
            hook.remove()

        self.assertEqual(len(decoder_inputs), 2)
        baseline_decoder_input, multimodal_decoder_input = decoder_inputs
        torch.testing.assert_close(multimodal_decoder_input[1:2], feature)
        text_rows = torch.tensor([0, 2, 3], device="cuda")
        torch.testing.assert_close(
            multimodal_decoder_input[text_rows], baseline_decoder_input[text_rows]
        )
        self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(torch.allclose(actual, baseline))

    def test_handler_negotiates_first_or_last_rows_and_can_require_full_output(self):
        from rtp_llm.utils.tensor_utils import first_token_indices, last_token_indices

        if torch.version.hip is not None:
            self.skipTest("real Bert FMHA coverage is CUDA-only")
        model = self._build_one_layer_model()
        first = SimpleNamespace(pooled_row_indices=first_token_indices)
        self.assertIsNone(
            model.get_pooled_embedding_forward(first)
        )  # Plain FP16 linear.
        with mock.patch.object(BertModel, "supports_pooled_embedding", True):
            self.assertIsNone(model.get_pooled_embedding_forward(object()))
            inputs = self._build_one_layer_inputs()
            full = model.forward(inputs).hidden_states
            for indices, expected in (
                (first_token_indices, 0),
                (last_token_indices, -1),
            ):
                forward = model.get_pooled_embedding_forward(
                    SimpleNamespace(pooled_row_indices=indices)
                )
                torch.testing.assert_close(
                    forward(inputs).hidden_states,
                    full[[expected]],
                    atol=1e-3,
                    rtol=1e-3,
                )

    def test_attention_output_projection_preserves_bias_gate_and_tp_reduction(self):
        attn = self._build_one_layer_model().layers[0].self_attn
        hidden = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3)
        bias = torch.tensor([1.0, -1.0, 2.0], device="cuda")
        gate = torch.full_like(hidden, 2.0)
        rows = torch.tensor([2, 0, 2], device="cuda")
        expected = hidden * gate.sigmoid()
        fmha = SimpleNamespace(forward=lambda qkv, kv_cache, layer_idx: qkv)

        class Projection(torch.nn.Module):
            supports_deferred_bias = True

            def __init__(self):
                super().__init__()
                self.bias = bias

            def forward(self, x):
                return x + self.bias

            def forward_without_bias(self, x):
                return x

        attn.qkv_proj = torch.nn.Identity()
        attn.qk_fuse_norm = None
        attn.o_proj = Projection()
        torch.testing.assert_close(attn(hidden, fmha, None, gate), expected + bias)
        compact, deferred_bias = attn.forward_without_output_bias(
            hidden, fmha, None, gate, output_rows=rows
        )
        torch.testing.assert_close(compact, expected[rows])
        self.assertIs(deferred_bias, bias)

        attn.tp_size = 2
        with mock.patch(
            "rtp_llm.models_py.modules.hybrid.causal_attention.all_reduce",
            side_effect=lambda x, group: x * 2,
        ) as reduce:
            torch.testing.assert_close(
                attn(hidden, fmha, None, gate), (expected + bias) * 2
            )
            reduced, deferred_bias = attn.forward_without_output_bias(
                hidden, fmha, None, gate
            )
            torch.testing.assert_close(reduced, (expected + bias) * 2)
            self.assertIsNone(deferred_bias)
            self.assertEqual(reduce.call_count, 2)

    def test_final_layer_computes_only_requested_rows_after_full_attention(self):
        if torch.version.hip is not None:
            self.skipTest("real Bert FMHA coverage is CUDA-only")
        model = self._build_one_layer_model()
        rows = torch.tensor([3, 0, 3], dtype=torch.long, device="cuda")
        full = model.forward(self._build_one_layer_inputs()).hidden_states
        compact = model.forward(
            self._build_one_layer_inputs(), output_rows=rows
        ).hidden_states
        torch.testing.assert_close(
            compact, full.index_select(0, rows), atol=1e-3, rtol=1e-3
        )

    def test_forward_rejects_nonempty_mask_without_multimodal_features(self):
        model = self._build_model()
        masked_inputs = _build_inputs()
        masked_inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [0, 1, 0, 1], dtype=torch.int32, device="cuda"
        )
        with self.assertRaisesRegex(
            ValueError,
            "features, locations, and text_tokens_mask must be provided together",
        ):
            model.forward(masked_inputs, fmha_impl=_FMHA_NOT_USED)

    def test_forward_rejects_multimodal_features_without_mask(self):
        model = self._build_model()
        inputs = _build_inputs(with_text_tokens_mask=False)
        self.assertIsNone(inputs.embedding_inputs.text_tokens_mask)
        inputs.multimodal_inputs.multimodal_features = [
            torch.ones((1, 4), dtype=torch.float16, device="cuda")
        ]
        inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            model.forward(inputs, fmha_impl=_FMHA_NOT_USED)

    def test_forward_rejects_multimodal_features_with_empty_mask(self):
        model = self._build_model()
        inputs = _build_inputs()
        inputs.multimodal_inputs.multimodal_features = [
            torch.ones((1, 4), dtype=torch.float16, device="cuda")
        ]
        inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        inputs.embedding_inputs.text_tokens_mask = torch.empty(
            0, dtype=torch.int32, device="cuda"
        )

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            model.forward(inputs, fmha_impl=_FMHA_NOT_USED)

    def test_forward_rejects_locations_without_features_and_mask(self):
        model = self._build_model()
        inputs = _build_inputs(with_text_tokens_mask=False)
        inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            model.forward(inputs, fmha_impl=_FMHA_NOT_USED)


if __name__ == "__main__":
    main()
