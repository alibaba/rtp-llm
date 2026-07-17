"""G1 aux-hidden-state capture UT (dspark/dflash multi-layer feature export).

Runs a tiny random-weight Qwen3Model and checks:
  1. capture off  -> PyModelOutputs.aux_hidden_states stays unset;
  2. capture on   -> [token, L, hidden] stack whose slots equal the
     independently recomputed per-layer outputs (full residual stream after
     layer i, for the configured 0-based layer ids).

Layer-id semantics note: the ids here are 0-based "output of layer i".  The
draft ckpt's aux_hidden_state_layer_ids are 1-based (id j = output of 1-based
layer j); conversion ids = [j - 1 for j in aux_ids] happens where the draft
ckpt is read (see ModelConfig.capture_aux_hidden_layer_ids).
"""

import unittest

import torch

HIDDEN = 128
LAYERS = 4
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
INTER = 256
VOCAB = 128
TOKENS = 6


def _build_tiny_model():
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.model_loader.model_weight_info import ModelWeights
    from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
    from rtp_llm.ops import ParallelismConfig
    from rtp_llm.utils.model_weight import W

    config = ModelConfig()
    config.num_layers = LAYERS
    config.hidden_size = HIDDEN
    config.vocab_size = VOCAB
    config.max_seq_len = 64
    config.attn_config.head_num = HEADS
    config.attn_config.kv_head_num = KV_HEADS
    config.attn_config.size_per_head = HEAD_DIM
    config.attn_config.rope_config.dim = HEAD_DIM
    config.attn_config.rope_config.style = 1
    config.attn_config.rope_config.base = 10000
    config.attn_config.tokens_per_block = 16
    config.attn_config.kernel_tokens_per_block = 16

    dt = torch.bfloat16
    dev = "cuda"
    weights = ModelWeights(LAYERS, dev, dt)

    def rand(*shape):
        return torch.randn(*shape, dtype=dt, device=dev) * 0.05

    weights.set_global_weight(W.embedding, rand(VOCAB, HIDDEN))
    weights.set_global_weight(W.final_ln_gamma, torch.ones(HIDDEN, dtype=dt, device=dev))
    for i in range(LAYERS):
        lw = weights.weights[i]
        lw[W.pre_ln_gamma] = torch.ones(HIDDEN, dtype=dt, device=dev)
        lw[W.post_ln_gamma] = torch.ones(HIDDEN, dtype=dt, device=dev)
        lw[W.attn_qkv_w] = rand(HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM)
        lw[W.attn_o_w] = rand(HEADS * HEAD_DIM, HIDDEN)
        lw[W.ffn_w1] = rand(HIDDEN, INTER)
        lw[W.ffn_w3] = rand(HIDDEN, INTER)
        lw[W.ffn_w2] = rand(INTER, HIDDEN)

    model = Qwen3Model(
        config, ParallelismConfig(), weights, max_generate_batch_size=4
    )
    model.attn_configs = config.getAttentionConfigs(1)
    model.attn_configs.dtype = dt
    model.attn_configs.need_rope_kv_cache = True
    return config, model


def _make_prefill_inputs(input_ids: torch.Tensor):
    """Pure ragged prefill (no prefix, no paged cache)."""
    from rtp_llm.ops.compute_ops import (
        PyAttentionInputs,
        PyModelInputs,
        get_typemeta,
    )

    tokens = input_ids.shape[0]
    ai = PyAttentionInputs()
    ai.is_prefill = True
    ai.input_lengths = torch.tensor(
        [tokens], dtype=torch.int32, device="cpu"
    ).pin_memory()
    ai.sequence_lengths = torch.tensor(
        [tokens], dtype=torch.int32, device="cpu"
    ).pin_memory()
    ai.prefix_lengths = torch.zeros(1, dtype=torch.int32, device="cpu").pin_memory()
    block_ids = torch.zeros(1, 1, dtype=torch.int32)
    ai.kv_cache_block_id_host = block_ids
    ai.kv_cache_block_id_device = block_ids.cuda()
    ai.kv_cache_kernel_block_id_host = block_ids
    ai.kv_cache_kernel_block_id_device = block_ids.cuda()
    ai.cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    ai.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))

    inputs = PyModelInputs()
    inputs.input_ids = input_ids
    inputs.input_hiddens = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    inputs.attention_inputs = ai
    return inputs


class Qwen3AuxCaptureTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)

    def _make_impl(self, model, inputs):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferPrefillImpl,
        )

        return PyFlashinferPrefillImpl(model.attn_configs, inputs.attention_inputs)

    def test_capture_off_leaves_aux_unset(self):
        config, model = _build_tiny_model()
        input_ids = torch.randint(0, VOCAB, (TOKENS,), dtype=torch.int32, device="cuda")
        inputs = _make_prefill_inputs(input_ids)
        with torch.no_grad():
            outputs = model.forward(inputs, self._make_impl(model, inputs))
        aux = outputs.aux_hidden_states
        self.assertTrue(
            aux is None or (isinstance(aux, torch.Tensor) and aux.numel() == 0),
            f"aux_hidden_states must stay unset when capture is off, got {aux}",
        )

    def test_capture_matches_per_layer_outputs(self):
        capture_ids = [0, 2, 3]
        config, model = _build_tiny_model()
        config.capture_aux_hidden_layer_ids = capture_ids

        input_ids = torch.randint(0, VOCAB, (TOKENS,), dtype=torch.int32, device="cuda")
        inputs = _make_prefill_inputs(input_ids)
        impl = self._make_impl(model, inputs)
        with torch.no_grad():
            outputs = model.forward(inputs, impl)

        aux = outputs.aux_hidden_states
        self.assertIsNotNone(aux)
        self.assertEqual(tuple(aux.shape), (TOKENS, len(capture_ids), HIDDEN))

        # Recompute per-layer outputs independently through the same modules.
        from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer

        with torch.no_grad():
            hidden = model.embed_tokens(input_ids)
            expected = {}
            for i, layer in enumerate(model.layers):
                select_block_map_for_layer(inputs.attention_inputs, i)
                hidden = layer(hidden, impl, kv_cache=None)
                if i in capture_ids:
                    expected[i] = hidden

        for slot, layer_id in enumerate(capture_ids):
            torch.testing.assert_close(
                aux[:, slot, :],
                expected[layer_id],
                atol=2e-2,
                rtol=2e-2,
                msg=f"aux slot {slot} (layer {layer_id} output) mismatch",
            )


if __name__ == "__main__":
    unittest.main()
