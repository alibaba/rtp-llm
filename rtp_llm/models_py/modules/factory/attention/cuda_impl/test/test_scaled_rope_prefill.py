"""Real FlashInfer attention, KV writes and NewLoader; no Q-only FMHA fake."""

import tempfile
import types
import unittest
from itertools import product

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.models_py.kernels.cuda.test.scaled_rope_reference import reference_rope
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    fill_paged_kv_cache,
)
from rtp_llm.models_py.new_models.llama import LlamaForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import FlashInferMlaAttnParams, PyModelInputs


class ScaledRopePrefillTest(BaseAttentionTest):
    def test_hf_dynamic_scaling_reaches_runtime_config(self):
        from rtp_llm.models.llama import Llama

        config = types.SimpleNamespace(
            attn_config=self.config(False), special_tokens=types.SimpleNamespace()
        )
        Llama.from_huggingface(
            config,
            {
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 512,
                "num_hidden_layers": 1,
                "vocab_size": 64,
                "intermediate_size": 256,
                "max_position_embeddings": 4096,
                "rope_scaling": {"type": "dynamic", "factor": 8.0},
            },
        )
        self.assertEqual(config.attn_config.rope_config.style, RopeStyle.DynamicNTK)
        self.assertEqual(config.attn_config.rope_config.scale, 8.0)
        self.assertEqual(config.attn_config.rope_config.max_pos, 4096)

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("FlashInfer scaled RoPE tests require a CUDA GPU")

    def config(self, llama3):
        config = self._create_config(
            head_num=4, head_num_kv=2, size_per_head=128, seq_size_per_block=16
        ).attn_configs
        config.need_rope_kv_cache = True
        config.is_causal = True
        config.max_seq_len = 256
        rope = config.rope_config
        rope.style = RopeStyle.Llama3 if llama3 else RopeStyle.DynamicNTK
        rope.dim, rope.base, rope.scale, rope.max_pos = 128, 10000.0, 8.0, 32
        rope.factor1, rope.factor2 = 1.0, 4.0
        return config

    @staticmethod
    def rope(x, prefix, total, llama3):
        positions = torch.arange(prefix, prefix + x.shape[0], device=x.device)
        lengths = torch.full_like(positions, total)
        return reference_rope(
            x,
            positions,
            lengths,
            rotary_dim=128,
            base=10000.0,
            scale=8.0,
            original_max_position=32,
            llama3=llama3,
        )

    @staticmethod
    def attention(q, k, v, prefix):
        # Explicit offset causal mask: SDPA's is_causal uses top-left alignment.
        mask = torch.arange(k.shape[0], device=q.device)[None, :] <= (
            torch.arange(q.shape[0], device=q.device)[:, None] + prefix
        )
        return (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).float(),
                k.repeat_interleave(2, dim=1).transpose(0, 1).float(),
                v.repeat_interleave(2, dim=1).transpose(0, 1).float(),
                attn_mask=mask,
            )
            .transpose(0, 1)
            .to(q.dtype)
        )

    def test_ragged_and_prefix_paged_outputs_and_kv_match_reference(self):
        for llama3 in (True, False):
            for prefixes, device_input in product(([0, 0], [7, 35]), (False, True)):
                with self.subTest(
                    llama3=llama3, prefixes=prefixes, device_input=device_input
                ):
                    counts = [17, 41]
                    config = self.config(llama3)
                    inputs = self._create_chunked_prefill_attention_inputs(
                        2, prefixes, counts, 16
                    )
                    if device_input:
                        inputs.input_lengths = inputs.input_lengths.to(self.device)
                        inputs.prefix_lengths = inputs.prefix_lengths.to(self.device)
                        inputs.prefix_lengths_device = inputs.prefix_lengths
                    lengths = [p + n for p, n in zip(prefixes, counts)]
                    raw = [
                        torch.randn(n, 8, 128, device=self.device, dtype=torch.float16)
                        for n in lengths
                    ]
                    keys = [
                        self.rope(x[:, 4:6], 0, n, llama3) for x, n in zip(raw, lengths)
                    ]
                    values = [x[:, 6:] for x in raw]
                    cache = fill_paged_kv_cache(
                        keys,
                        values,
                        prefixes,
                        inputs.kv_cache_kernel_block_id,
                        16,
                        2,
                        128,
                        torch.float16,
                        self.device,
                        total_pages=sum((n + 15) // 16 for n in lengths),
                    )
                    impl_type = (
                        PyFlashinferPagedPrefillImpl
                        if any(prefixes)
                        else PyFlashinferPrefillImpl
                    )
                    impl = impl_type(config, inputs)
                    if device_input and any(prefixes):
                        # Poison and refill: the old device planner left this
                        # buffer untouched, even though DynamicNTK consumes it.
                        impl.fmha_params.kvlen_d.fill_(-777)
                        impl.fmha_impl.prepare(inputs)
                        torch.testing.assert_close(
                            impl.fmha_params.kvlen_d.cpu(),
                            torch.tensor(lengths, dtype=torch.int32),
                        )
                    packed = torch.cat([x[p:] for x, p in zip(raw, prefixes)]).reshape(
                        sum(counts), -1
                    )
                    actual = impl.forward(packed.clone(), cache).reshape(
                        sum(counts), 4, 128
                    )
                    expected = torch.cat(
                        [
                            self.attention(self.rope(x[p:, :4], p, n, llama3), k, v, p)
                            for x, k, v, p, n in zip(
                                raw, keys, values, prefixes, lengths
                            )
                        ]
                    )
                    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
                    full_cache = fill_paged_kv_cache(
                        keys,
                        values,
                        lengths,
                        inputs.kv_cache_kernel_block_id,
                        16,
                        2,
                        128,
                        torch.float16,
                        self.device,
                    )
                    torch.testing.assert_close(
                        cache.kv_cache_base,
                        full_cache.kv_cache_base,
                        rtol=3e-3,
                        atol=3e-3,
                    )

    def test_device_planner_kv_lengths_decode_prefill_and_reuse(self):
        params = FlashInferMlaAttnParams()
        empty = torch.empty(0, device=self.device, dtype=torch.int32)
        blocks = torch.arange(16, device=self.device, dtype=torch.int32).reshape(2, 8)
        for decode in (False, True):
            for batch in (2, 1, 2):
                lengths = torch.tensor(
                    [5, 9][:batch], device=self.device, dtype=torch.int32
                )
                prefixes = torch.tensor(
                    [17, 35][:batch], device=self.device, dtype=torch.int32
                )
                args = (
                    empty if decode else prefixes,
                    prefixes if decode else empty,
                    lengths,
                    blocks[:batch],
                    16,
                    False,
                )
                params.fill_params_mha_device(*args)
                params.kvlen_d.fill_(-777)
                params.fill_params_mha_device(*args)
                expected = prefixes + 1 if decode else prefixes + lengths
                torch.testing.assert_close(params.kvlen_d, expected)
                self.assertEqual(params.kvlen_d.numel(), batch)

    def test_device_planner_and_dynamic_ntk_graph_replay(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
            MhaRotaryEmbeddingOp,
        )

        params = FlashInferMlaAttnParams()
        empty = torch.empty(0, device=self.device, dtype=torch.int32)
        lengths = torch.tensor([3, 5], device=self.device, dtype=torch.int32)
        prefixes = torch.tensor([0, 29], device=self.device, dtype=torch.int32)
        blocks = torch.arange(16, device=self.device, dtype=torch.int32).reshape(2, 8)
        raw = torch.randn(8, 8, 128, device=self.device, dtype=torch.float16)
        qkv = raw.clone()
        rope = MhaRotaryEmbeddingOp(self.config(False))
        rope.set_params(params)

        def run():
            params.fill_params_mha_device(prefixes, empty, lengths, blocks, 16, True)
            rope.forward(qkv.reshape(8, -1))

        params.fill_params_mha_device(prefixes, empty, lengths, blocks, 16, False)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                qkv.copy_(raw)
                run()
        torch.cuda.current_stream().wait_stream(stream)
        ptr = params.kvlen_d.data_ptr()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for offsets in ([0, 29], [39, 67]):
            prefixes.copy_(torch.tensor(offsets, device=self.device, dtype=torch.int32))
            params.kvlen_d.fill_(-777)
            qkv.copy_(raw)
            graph.replay()
            self.assertEqual(params.kvlen_d.data_ptr(), ptr)
            torch.testing.assert_close(params.kvlen_d, prefixes + lengths)
            for start, count, prefix in ((0, 3, offsets[0]), (3, 5, offsets[1])):
                for left, right in ((0, 4), (4, 6)):
                    expected = self.rope(
                        raw[start : start + count, left:right],
                        prefix,
                        prefix + count,
                        False,
                    )
                    torch.testing.assert_close(
                        qkv[start : start + count, left:right],
                        expected,
                        rtol=4e-3,
                        atol=2e-2,
                    )
            torch.testing.assert_close(qkv[:, 6:], raw[:, 6:], rtol=0, atol=0)

    def test_full_newloader_llama_forward_uses_real_scaled_attention(self):
        for llama3 in (True, False):
            with self.subTest(
                llama3=llama3
            ), tempfile.TemporaryDirectory() as directory:
                config = self.config(llama3)
                parallel = types.SimpleNamespace(
                    tp_size=1,
                    tp_rank=0,
                    ep_size=1,
                    ep_rank=0,
                    prefill_cp_config=types.SimpleNamespace(
                        is_enabled=lambda: False, is_prefill_enabled=lambda: False
                    ),
                    get_attn_tp_size=lambda: 1,
                    get_attn_tp_rank=lambda: 0,
                    get_ffn_tp_size=lambda: 1,
                    get_ffn_tp_rank=lambda: 0,
                )
                model_config = types.SimpleNamespace(
                    model_type="llama",
                    num_layers=1,
                    vocab_size=64,
                    hidden_size=512,
                    inter_size=256,
                    attn_config=config,
                    layernorm_eps=1e-6,
                    enable_fp32_lm_head=False,
                    tie_word_embeddings=True,
                    attention_bias=False,
                    mlp_bias=False,
                    hidden_act="silu",
                )
                shapes = {
                    "model.embed_tokens.weight": (64, 512),
                    "model.layers.0.input_layernorm.weight": (512,),
                    "model.layers.0.self_attn.q_proj.weight": (512, 512),
                    "model.layers.0.self_attn.k_proj.weight": (256, 512),
                    "model.layers.0.self_attn.v_proj.weight": (256, 512),
                    "model.layers.0.self_attn.o_proj.weight": (512, 512),
                    "model.layers.0.post_attention_layernorm.weight": (512,),
                    "model.layers.0.mlp.gate_proj.weight": (256, 512),
                    "model.layers.0.mlp.up_proj.weight": (256, 512),
                    "model.layers.0.mlp.down_proj.weight": (512, 256),
                    "model.norm.weight": (512,),
                }
                weights = {
                    name: (
                        torch.ones(shape)
                        if len(shape) == 1
                        else torch.randn(shape) * 0.08
                    ).half()
                    for name, shape in shapes.items()
                }
                save_file(weights, directory + "/model.safetensors")
                load_config = NewLoaderConfig(
                    device="cuda",
                    compute_dtype=torch.float16,
                    quant_config=QuantizationConfig("none"),
                    parallelism_config=parallel,
                )
                model = NewModelLoader(
                    model_config, load_config, model_path=directory
                ).load()
                self.assertIsInstance(model, LlamaForCausalLM)
                inputs = self._create_prefill_attention_inputs(
                    1, [49], 16, with_kv_cache_block_ids=False
                )
                impl = PyFlashinferPrefillImpl(config, inputs)
                ids = torch.arange(49, device=self.device, dtype=torch.int32)
                actual = model(
                    PyModelInputs(input_ids=ids), fmha_impl=impl
                ).hidden_states
                weights = {
                    name: value.to(self.device) for name, value in weights.items()
                }

                def norm(x, name):
                    value = x.float() * torch.rsqrt(
                        x.float().square().mean(-1, keepdim=True) + 1e-6
                    )
                    return (value * weights[name].float()).half()

                hidden = F.embedding(ids.long(), weights["model.embed_tokens.weight"])
                residual = hidden
                x = norm(hidden, "model.layers.0.input_layernorm.weight")
                q, k, v = [
                    F.linear(
                        x, weights[f"model.layers.0.self_attn.{p}_proj.weight"]
                    ).view(49, heads, 128)
                    for p, heads in (("q", 4), ("k", 2), ("v", 2))
                ]
                attended = self.attention(
                    self.rope(q, 0, 49, llama3), self.rope(k, 0, 49, llama3), v, 0
                )
                hidden = residual + F.linear(
                    attended.reshape(49, 512),
                    weights["model.layers.0.self_attn.o_proj.weight"],
                )
                residual = hidden
                x = norm(hidden, "model.layers.0.post_attention_layernorm.weight")
                gate = F.linear(x, weights["model.layers.0.mlp.gate_proj.weight"])
                up = F.linear(x, weights["model.layers.0.mlp.up_proj.weight"])
                hidden = residual + F.linear(
                    (F.silu(gate.float()) * up.float()).half(),
                    weights["model.layers.0.mlp.down_proj.weight"],
                )
                expected = norm(hidden, "model.norm.weight")
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
