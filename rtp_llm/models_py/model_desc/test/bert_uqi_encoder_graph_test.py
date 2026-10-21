"""CUDA/HIP replay oracle across changing UQI layouts, including pure QI batches."""
import unittest
from types import SimpleNamespace

import torch

if torch.version.hip is not None:
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.bert_uqi_two_pass import (
        BertUqiRocmEncoderGraphImpl as GraphImpl,
    )
else:
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi_two_pass import (
        BertUqiEncoderGraphImpl as GraphImpl,
    )


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class UqiEncoderGraphTest(unittest.TestCase):
    def test_replay_changing_segments(self):
        self._check_replay(torch.float16)

    def test_replay_bfloat16(self):
        self._check_replay(torch.bfloat16)

    def test_replay_aggregate_bucket_exceeds_sequence_limit(self):
        self._check_replay(
            torch.float16, capacity=6144, max_sequence_length=512,
            request_capacity=16,
            layouts=[
                ([512] * 12 + [0] * 4, [128] * 12 + [-1] * 4, [64] * 12 + [0] * 4),
                ([170] * 16, [-1] * 16, [0] * 16),
                ([7, 5] + [0] * 14, [2, 0] + [-1] * 14, [3, 5] + [0] * 14),
            ],
        )

    @torch.inference_mode()
    def test_multilayer_transformer_replay_matches_dense_eager(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.models_py.model_desc.bert import BertModel
        from rtp_llm.ops import ActivationType, ParallelismConfig
        from rtp_llm.utils.model_weight import W

        capacity, hidden, limit = 80, 128, 32
        config = ModelConfig()
        config.num_layers = 2
        config.hidden_size = hidden
        config.vocab_size = 16
        config.inter_size = 2 * hidden
        config.max_seq_len = limit
        config.activation_type = ActivationType.Gelu
        config.attn_config.is_causal = False
        config.attn_config.head_num = 1
        config.attn_config.kv_head_num = 1
        config.attn_config.size_per_head = hidden
        config.attn_config.tokens_per_block = 64
        config.attn_config.kernel_tokens_per_block = 64
        config.use_kvcache = False
        generator = torch.Generator(device="cuda").manual_seed(17)

        def random(shape):
            return torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * 0.02

        weights = ModelWeights(2, "cuda", torch.float16)
        weights.set_global_weight(W.embedding, random((16, hidden)))
        weights.set_global_weight(W.pre_decoder_ln_gamma, torch.ones(hidden, device="cuda", dtype=torch.float16))
        weights.set_global_weight(W.pre_decoder_ln_beta, torch.zeros(hidden, device="cuda", dtype=torch.float16))
        for layer in range(2):
            for name, shape in ((W.attn_qkv_w, (hidden, 3 * hidden)),
                                (W.attn_o_w, (hidden, hidden)),
                                (W.ffn_w3, (hidden, 2 * hidden)),
                                (W.ffn_w2, (2 * hidden, hidden))):
                weights.set_layer_weight(layer, name, random(shape))
            for name in (W.post_ln_gamma, W.post_ffn_ln_gamma):
                weights.set_layer_weight(layer, name, torch.ones(hidden, device="cuda", dtype=torch.float16))
            for name in (W.post_ln_beta, W.post_ffn_ln_beta):
                weights.set_layer_weight(layer, name, torch.zeros(hidden, device="cuda", dtype=torch.float16))
        model = BertModel(config, ParallelismConfig(), weights, 1).eval()
        model.use_user_profile_mask = True

        def attention_inputs(lengths, starts, counts):
            return SimpleNamespace(
                input_lengths=torch.tensor(lengths, dtype=torch.int32),
                prefix_lengths=torch.zeros(3, dtype=torch.int32),
                uqi_b_starts=torch.tensor(starts, dtype=torch.int32),
                uqi_b_lens=torch.tensor(counts, dtype=torch.int32),
            )

        initial = attention_inputs([capacity, 0, 0], [-1] * 3, [0] * 3)
        impl = model.prepare_fmha_impl(SimpleNamespace(
            input_ids=torch.zeros(capacity, device="cuda", dtype=torch.int32),
            attention_inputs=initial,
        ), is_cuda_graph=True)
        source = random((capacity, hidden))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.forward(
                    SimpleNamespace(input_hiddens=source), impl
                )
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = model.forward(
                SimpleNamespace(input_hiddens=source), impl
            ).hidden_states

        for lengths, starts, counts in [
            ([29, 25, 0], [4, 10, -1], [7, 5, 0]),
            ([7, 5, 3], [-1, -1, -1], [0, 0, 0]),
            ([32, 32, 16], [0, -1, 4], [32, 0, 6]),
        ] * 2:
            impl.prepare_cuda_graph(attention_inputs(lengths, starts, counts))
            source.copy_(random(source.shape))
            # Dense eager oracle stays in original token order and computes
            # each request separately; it does not reuse Graph layout helpers.
            def dense_forward(qkv, kv_cache, layer_idx=0):
                q, k, v = qkv.reshape(qkv.shape[0], 3 * hidden).split(hidden, dim=-1)
                rows, offset = [], 0
                for length, start, count in zip(lengths, starts, counts):
                    if length == 0:
                        continue
                    end = offset + length
                    pos = torch.arange(length, device="cuda")
                    is_b = (pos >= start) & (pos < start + count)
                    allowed = is_b[:, None] | ~is_b[None, :]
                    logits = q[offset:end].float() @ k[offset:end].float().T / hidden**0.5
                    probs = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
                    rows.append((probs @ v[offset:end].float()).to(qkv.dtype))
                    offset = end
                return torch.cat(rows)

            graph.replay()
            torch.cuda.synchronize()
            saved = actual[:sum(lengths)].clone()
            reference = model.forward(
                SimpleNamespace(input_hiddens=source[:sum(lengths)]),
                SimpleNamespace(forward=dense_forward),
            ).hidden_states
            self.assertTrue(torch.isfinite(saved).all().item())
            torch.testing.assert_close(saved, reference, atol=1e-2, rtol=1e-2)

    def _check_replay(self, dtype, capacity=80, max_sequence_length=None,
                      request_capacity=3, layouts=None):
        heads, kv_heads, dim = 4, 2, 64
        max_sequence_length = max_sequence_length or capacity
        def inputs(lengths, starts, counts):
            return SimpleNamespace(
                input_lengths=torch.tensor(lengths, dtype=torch.int32),
                prefix_lengths=torch.zeros(request_capacity, dtype=torch.int32),
                uqi_b_starts=torch.tensor(starts, dtype=torch.int32),
                uqi_b_lens=torch.tensor(counts, dtype=torch.int32),
            )
        config = SimpleNamespace(is_causal=False, head_num=heads, kv_head_num=kv_heads,
                                 size_per_head=dim, dtype=dtype)
        initial = inputs([capacity] + [0] * (request_capacity - 1),
                         [-1] * request_capacity, [0] * request_capacity)
        impl = GraphImpl(config, initial, capacity, max_sequence_length)
        source = torch.randn(capacity, (heads + 2 * kv_heads) * dim, device="cuda", dtype=dtype)
        def run():
            if torch.version.hip is not None:
                ordered = source.index_select(0, impl.uqi_perm)
                return impl.forward(ordered, None).index_select(0, impl.uqi_inv_perm)
            return impl.forward(source, None)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        buffers = (
            (
                impl.uqi_perm,
                impl.uqi_inv_perm,
                impl.split_cu_seqlens,
                impl.full_cu_seqlens,
                impl.b_mask,
                impl.valid_mask,
            )
            if torch.version.hip is not None
            else (impl.fmha_impl.custom_mask_buffer, impl.fmha_impl.mask_indptr_buffer)
        )
        pointers = tuple(t.data_ptr() for t in buffers)
        default_layouts = [
            ([7, 5, 0], [2, 1, -1], [3, 2, 0]),
            ([80, 0, 0], [-1] * 3, [0] * 3),
            ([67, 9, 0], [0, 2, -1], [35, 4, 0]),
            ([3, 5, 8], [0, -1, 5], [3, 0, 2]),
            ([2, 3, 1], [-1, 0, 0], [0, 3, 1]),
        ]
        for lengths, starts, counts in (layouts or default_layouts) * 2:
            impl.prepare_cuda_graph(inputs(lengths, starts, counts))
            source.normal_()
            if torch.version.hip is not None:
                # Padding must not be read by attention and must be overwritten.
                source[sum(lengths):].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            if torch.version.hip is not None:
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertEqual(torch.count_nonzero(output[sum(lengths):]).item(), 0)
            expected = []
            offset = 0
            for n, start, count in zip(lengths, starts, counts):
                if not n:
                    continue
                q, k, v = source[offset:offset+n].split(
                    [heads * dim, kv_heads * dim, kv_heads * dim], dim=-1
                )
                q = q.reshape(n, heads, dim)
                k = k.reshape(n, kv_heads, dim).repeat_interleave(heads // kv_heads, dim=1)
                v = v.reshape(n, kv_heads, dim).repeat_interleave(heads // kv_heads, dim=1)
                logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / dim**0.5
                b = (torch.arange(n, device="cuda") >= start) & (torch.arange(n, device="cuda") < start + count)
                allowed = b[:, None] | ~b[None, :]
                probs = logits.masked_fill(~allowed[None], -torch.inf).softmax(-1)
                expected.append(torch.einsum("hqk,khd->qhd", probs, v.float()).reshape(n, -1))
                offset += n
            torch.testing.assert_close(output[:offset].float(), torch.cat(expected), atol=2e-2 if dtype == torch.bfloat16 else 4e-3, rtol=2e-2 if dtype == torch.bfloat16 else 4e-3)
            if torch.version.hip is not None:
                from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
                    build_bert_uqi_two_pass_schedule_from_bounds,
                )
                from rtp_llm.models_py.modules.factory.attention.rocm_impl.bert_uqi_two_pass_core import (
                    prepare_two_pass, run_two_pass,
                )
                cu = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
                schedule = build_bert_uqi_two_pass_schedule_from_bounds(
                    torch.tensor(starts), torch.tensor(counts), cu, source.device)
                live = source[:offset]
                if schedule.perm is not None:
                    live = live.index_select(0, schedule.perm)
                q, k, v = live.split([heads*dim, kv_heads*dim, kv_heads*dim], -1)
                reference = run_two_pass(
                    prepare_two_pass(schedule, source.device),
                    q.reshape(offset, heads, dim), k.reshape(offset, kv_heads, dim),
                    v.reshape(offset, kv_heads, dim)).reshape(offset, -1)
                if schedule.inv_perm is not None:
                    reference = reference.index_select(0, schedule.inv_perm)
                torch.testing.assert_close(output[:offset], reference,
                    atol=2e-2 if dtype == torch.bfloat16 else 4e-3,
                    rtol=2e-2 if dtype == torch.bfloat16 else 4e-3)
            self.assertEqual(pointers, tuple(t.data_ptr() for t in buffers))


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class EmbeddingPrefillGraphTest(unittest.TestCase):
    def test_encoder_graph_replans_layout_and_isolates_padding(self):
        self._check_encoder_graph(causal=False)

    def test_causal_encoder_graph_replans_layout_and_isolates_padding(self):
        self._check_encoder_graph(causal=True)

    def _check_encoder_graph(self, causal):
        if torch.version.hip is None:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                PyFlashinferPrefillImpl as DevicePrefillImpl,
            )
        else:
            from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
                AiterPrefillImplNonAsm as DevicePrefillImpl,
            )
        from rtp_llm.ops import KvCacheDataType
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        capacity, dim = 8, 64
        configs = SimpleNamespace(
            is_causal=causal,
            kernel_tokens_per_block=8,
            kv_cache_dtype=KvCacheDataType.BASE,
            head_num=1,
            kv_head_num=1,
            size_per_head=dim,
            dtype=torch.bfloat16,
        )
        inputs = PyAttentionInputs()
        inputs.input_lengths = torch.tensor([8, 0], dtype=torch.int32).pin_memory()
        inputs.prefix_lengths = torch.zeros(2, dtype=torch.int32).pin_memory()
        impl = DevicePrefillImpl(
            configs,
            inputs,
            is_cuda_graph=True,
            token_capacity=capacity,
            max_seq_len=capacity,
        )
        qkv = torch.randn(capacity, 3 * dim, dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            impl.forward(qkv, None)
            impl.forward(qkv, None)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = impl.forward(qkv, None)
        pointer = impl.fmha_impl.cu_seqlens_device.data_ptr()
        for lengths in ([3, 2], [8, 0], [2, 0], [1, 5], [4, 4]):
            with self.subTest(lengths=lengths):
                torch.cuda.synchronize()
                inputs.input_lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
                impl.prepare_cuda_graph(inputs)
                qkv.normal_()
                real_tokens = sum(lengths)
                # Adversarial padding would contaminate a real sequence if its
                # length were incorrectly extended to the token bucket.
                qkv[real_tokens:].fill_(20)
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(impl.fmha_impl.cu_seqlens_device.data_ptr(), pointer)
                offset = 0
                for length in lengths:
                    if length:
                        q, k, v = qkv[offset : offset + length].chunk(3, dim=-1)
                        expected = torch.nn.functional.scaled_dot_product_attention(
                            q[None, None],
                            k[None, None],
                            v[None, None],
                            is_causal=causal,
                        )[0, 0]
                        torch.testing.assert_close(
                            output[offset : offset + length],
                            expected,
                            atol=2e-2,
                            rtol=2e-2,
                        )
                    offset += length


if __name__ == "__main__":
    unittest.main()
