from unittest import SkipTest, TestCase, main

import torch
from torch.nn import functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.ops import ActivationType, ParallelismConfig, rtp_llm_ops
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


class BertEmbeddingOpTest(TestCase):
    """Covers the embedding_bert op contract with a plain word-embedding table."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.weight = _build_word_embedding()

    def _embed(
        self,
        weight: torch.Tensor,
        inputs: PyModelInputs,
        mask,
        *,
        input_ids: torch.Tensor | None = None,
    ):
        bert_inputs = inputs.bert_embedding_inputs
        tokens = inputs.input_ids if input_ids is None else input_ids
        output = torch.empty(
            (tokens.size(0), weight.size(1)), dtype=weight.dtype, device=weight.device
        )
        rtp_llm_ops.embedding_bert(
            output,
            tokens,
            weight,
            bert_inputs.combo_position_ids,
            bert_inputs.position_encoding,
            bert_inputs.combo_tokens_type_ids,
            bert_inputs.token_type_embedding,
            bert_inputs.input_embedding_scalar,
            mask,
        )
        return output

    def test_embedding_bert_skips_masked_oov_hashes(self):
        inputs = _build_inputs()
        inputs.input_ids = torch.tensor(
            [123456, 2, -3, 741852], dtype=torch.int32, device="cuda"
        )
        inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [0, 1, 0, 0], dtype=torch.int32, device="cuda"
        )
        bert_inputs = inputs.bert_embedding_inputs
        bert_inputs.position_encoding = torch.arange(
            16, dtype=torch.float16, device="cuda"
        ).reshape(4, 4)
        bert_inputs.token_type_embedding = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0]], dtype=torch.float16, device="cuda"
        )
        actual = self._embed(
            self.weight, inputs, inputs.embedding_inputs.text_tokens_mask
        )
        masked_rows = torch.tensor([0, 2, 3], device="cuda")
        expected_masked_rows = (
            bert_inputs.position_encoding[masked_rows]
            + bert_inputs.token_type_embedding[0]
        )
        torch.testing.assert_close(actual[masked_rows], expected_masked_rows)
        # Construct the text-row oracle from detached values and explicit
        # scalar arithmetic rather than the module's embedding expression.
        token_weight = self.weight.detach().clone()[2]
        scalar = float(inputs.bert_embedding_inputs.input_embedding_scalar)
        expected_text_row = (
            torch.mul(token_weight, scalar)
            + bert_inputs.position_encoding[1]
            + bert_inputs.token_type_embedding[0]
        )
        torch.testing.assert_close(actual[1], expected_text_row)

    def test_embedding_bert_rejects_invalid_output(self):
        inputs = _build_inputs()
        bert_inputs = inputs.bert_embedding_inputs
        output = torch.empty(
            (4, _ZERO_LAYER_HIDDEN_SIZE), dtype=torch.float32, device="cuda"
        )
        with self.assertRaisesRegex(RuntimeError, "output dtype must match"):
            rtp_llm_ops.embedding_bert(
                output,
                inputs.input_ids,
                self.weight,
                bert_inputs.combo_position_ids,
                bert_inputs.position_encoding,
                bert_inputs.combo_tokens_type_ids,
                bert_inputs.token_type_embedding,
                bert_inputs.input_embedding_scalar,
                inputs.embedding_inputs.text_tokens_mask,
            )

    def test_embedding_bert_mask_semantics_cover_bf16_and_multiple_warps(self):
        hidden_size = 768
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.weight = _build_word_embedding(
                    dtype=dtype, hidden_size=hidden_size
                )
                inputs = _build_inputs(dtype=dtype, hidden_size=hidden_size)
                inputs.input_ids = torch.tensor(
                    [0, 123456, -3, _VOCAB_SIZE - 1],
                    dtype=torch.int32,
                    device="cuda",
                )
                mask = torch.tensor([1, 0, 0, 1], dtype=torch.int32, device="cuda")

                actual = self._embed(self.weight, inputs, mask)

                torch.testing.assert_close(actual[[0, 3]], self.weight[[0, -1]])
                torch.testing.assert_close(
                    actual[1:3],
                    torch.zeros((2, hidden_size), dtype=dtype, device="cuda"),
                )

    def test_embedding_bert_rejects_int64_input_ids(self):
        inputs = _build_inputs()
        with self.assertRaisesRegex(RuntimeError, "input_ids must be int32"):
            self._embed(
                self.weight,
                inputs,
                inputs.embedding_inputs.text_tokens_mask,
                input_ids=inputs.input_ids.to(torch.int64),
            )

    def test_embedding_bert_rejects_empty_position_or_type_table(self):
        def clear_position_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.position_encoding = torch.empty(
                (0, _ZERO_LAYER_HIDDEN_SIZE),
                dtype=torch.float16,
                device="cuda",
            )

        def clear_token_type_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.token_type_embedding = torch.empty(
                (0, _ZERO_LAYER_HIDDEN_SIZE),
                dtype=torch.float16,
                device="cuda",
            )

        cases = [
            (
                "position_encoding",
                clear_position_table,
                "position embedding table must not be empty",
            ),
            (
                "token_type_embedding",
                clear_token_type_table,
                "token type embedding table must not be empty",
            ),
        ]
        for name, mutate, expected_message in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_rejects_short_position_or_type_ids(self):
        def shorten_position_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_position_ids = torch.zeros(
                3, dtype=torch.int32, device="cuda"
            )

        def shorten_token_type_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.zeros(
                3, dtype=torch.int32, device="cuda"
            )

        cases = [
            (
                "combo_position_ids",
                shorten_position_ids,
                "combo_position_ids must have at least one id per token, got 3 vs 4",
            ),
            (
                "combo_tokens_type_ids",
                shorten_token_type_ids,
                "combo_tokens_type_ids must have at least one id per token, got 3 vs 4",
            ),
        ]
        for name, mutate, expected_message in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_rejects_cpu_ids_or_tables(self):
        def move_position_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_position_ids = (
                inputs.bert_embedding_inputs.combo_position_ids.cpu()
            )

        def move_token_type_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_tokens_type_ids = (
                inputs.bert_embedding_inputs.combo_tokens_type_ids.cpu()
            )

        def move_position_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.position_encoding = (
                inputs.bert_embedding_inputs.position_encoding.cpu()
            )

        def move_token_type_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.token_type_embedding = (
                inputs.bert_embedding_inputs.token_type_embedding.cpu()
            )

        cases = [
            ("combo_position_ids", move_position_ids),
            ("combo_tokens_type_ids", move_token_type_ids),
            ("position_encoding", move_position_table),
            ("token_type_embedding", move_token_type_table),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(
                    RuntimeError, rf"{name} must be a CUDA tensor"
                ):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_rejects_table_hidden_size_or_dtype_mismatch(self):
        def change_position_hidden_size(inputs: PyModelInputs) -> None:
            current = inputs.bert_embedding_inputs.position_encoding
            inputs.bert_embedding_inputs.position_encoding = torch.zeros(
                (current.size(0), _ZERO_LAYER_HIDDEN_SIZE + 1),
                dtype=torch.float16,
                device="cuda",
            )

        def change_token_type_hidden_size(inputs: PyModelInputs) -> None:
            current = inputs.bert_embedding_inputs.token_type_embedding
            inputs.bert_embedding_inputs.token_type_embedding = torch.zeros(
                (current.size(0), _ZERO_LAYER_HIDDEN_SIZE + 1),
                dtype=torch.float16,
                device="cuda",
            )

        def change_position_dtype(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.position_encoding = (
                inputs.bert_embedding_inputs.position_encoding.float()
            )

        def change_token_type_dtype(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.token_type_embedding = (
                inputs.bert_embedding_inputs.token_type_embedding.float()
            )

        cases = [
            (
                "position_hidden_size",
                change_position_hidden_size,
                r"position_encoding\.size\(1\).*5 vs 4",
            ),
            (
                "token_type_hidden_size",
                change_token_type_hidden_size,
                r"token_type_embedding\.size\(1\).*5 vs 4",
            ),
            (
                "position_dtype",
                change_position_dtype,
                "position_encoding dtype must match",
            ),
            (
                "token_type_dtype",
                change_token_type_dtype,
                "token_type_embedding dtype must match",
            ),
        ]
        for name, mutate, expected_message in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_all_zero_mask_with_single_token(self):
        inputs = _build_inputs()
        inputs.input_ids = torch.tensor([123456], dtype=torch.int32, device="cuda")
        inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        mask = torch.zeros(1, dtype=torch.int32, device="cuda")

        actual = self._embed(self.weight, inputs, mask)

        torch.testing.assert_close(
            actual,
            inputs.bert_embedding_inputs.position_encoding[0:1]
            + inputs.bert_embedding_inputs.token_type_embedding[0],
        )

    def test_embedding_bert_rejects_invalid_masks(self):
        inputs = _build_inputs()
        cases = [
            (
                "dtype",
                lambda: torch.ones(4, dtype=torch.bool, device="cuda"),
                "text_tokens_mask must be int32",
            ),
            (
                "device",
                lambda: torch.ones(4, dtype=torch.int32),
                "text_tokens_mask must be a CUDA tensor",
            ),
            (
                "contiguous",
                lambda: torch.ones(8, dtype=torch.int32, device="cuda")[::2],
                "text_tokens_mask must be contiguous",
            ),
            (
                "dimension",
                lambda: torch.ones((2, 2), dtype=torch.int32, device="cuda"),
                "text_tokens_mask must be a 1D tensor",
            ),
            (
                "length",
                lambda: torch.ones(3, dtype=torch.int32, device="cuda"),
                "text_tokens_mask must have one id per token, got 3 vs 4",
            ),
        ]

        for name, mask_factory, expected_message in cases:
            with self.subTest(name=name):
                invalid_mask = mask_factory()
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(self.weight, inputs, invalid_mask)

    def test_embedding_bert_accepts_empty_mask_as_absent(self):
        inputs = _build_inputs()
        empty_mask = torch.empty(0, dtype=torch.int32, device="cuda")

        actual = self._embed(self.weight, inputs, empty_mask)
        expected = self._embed(self.weight, inputs, None)

        torch.testing.assert_close(actual, expected)

    def test_embedding_bert_applies_scalar_only_to_unmasked_word_embeddings(self):
        inputs = _build_inputs()
        bert_inputs = inputs.bert_embedding_inputs
        bert_inputs.input_embedding_scalar = 2.0
        bert_inputs.position_encoding = torch.arange(
            16, dtype=torch.float16, device="cuda"
        ).reshape(4, 4)
        bert_inputs.token_type_embedding = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0]], dtype=torch.float16, device="cuda"
        )
        mask = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")

        actual = self._embed(self.weight, inputs, mask)
        word_embeddings = self.weight
        expected = (
            word_embeddings[inputs.input_ids.long()] * 2.0
            + bert_inputs.position_encoding
            + bert_inputs.token_type_embedding[0]
        )
        masked_rows = mask == 0
        expected[masked_rows] = (
            bert_inputs.position_encoding[masked_rows]
            + bert_inputs.token_type_embedding[0]
        )

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    main()
