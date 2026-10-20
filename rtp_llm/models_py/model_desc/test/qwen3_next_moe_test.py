"""Decoder construction and forwarding into the real generic routed-only layer."""

from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Moe
from rtp_llm.models_py.model_desc import generic_moe, qwen3_next
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class _Attention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _Norm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, residual


class _RoutedBackend(nn.Module):
    includes_shared_expert = False
    topk_ids_dtype = torch.int32
    router = SimpleNamespace(tp_collective_size=1, supports_skip_tp_allreduce=False)

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        self.calls += 1
        return hidden_states + topk_weights.sum(dim=-1, keepdim=True)


def _select_topk(logits, ids, weights):
    values, selected = logits.softmax(dim=-1).topk(ids.shape[-1])
    ids.copy_(selected)
    weights.copy_(values / values.sum(dim=-1, keepdim=True))


class Qwen3NextRoutedOnlyTest(TestCase):
    def test_routed_only_decoder_constructs_and_forwards_generic_moe(self):
        for model_cls in (Qwen3Next, Qwen35Moe):
            for shared_width in (None, 0):
                with self.subTest(model=model_cls.__name__, shared_width=shared_width):
                    config = ModelConfig()
                    config.num_layers = 1
                    config.hidden_size = 8
                    config.max_seq_len = 16
                    config.activation_type = "SiGLU"
                    config.hybrid_attention_config.hybrid_attention_types = [
                        HybridAttentionType.LINEAR
                    ]
                    hf = {
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 4,
                    }
                    if shared_width is not None:
                        hf["shared_expert_intermediate_size"] = shared_width
                    model_cls._parse_moe_config(hf, config)
                    weights = {
                        W.pre_ln_gamma: torch.ones(8),
                        W.post_ln_gamma: torch.ones(8),
                        W.moe_gate: torch.ones(4, 8),
                        W.moe_w1: torch.ones(4, 8, 8),
                        W.moe_w2: torch.ones(4, 8, 4),
                    }
                    backend = _RoutedBackend()
                    with (
                        patch.object(
                            qwen3_next,
                            "Qwen3NextGatedDeltaNet",
                            return_value=_Attention(),
                        ),
                        patch.object(
                            qwen3_next,
                            "RMSResNorm",
                            side_effect=lambda *a, **kw: _Norm(),
                        ),
                        patch.object(
                            generic_moe.LinearFactory,
                            "create_linear_from_weights",
                            return_value=nn.Linear(8, 4, bias=False),
                        ),
                        patch.object(
                            generic_moe, "SelectTopk", return_value=_select_topk
                        ),
                        patch.object(
                            generic_moe.FusedMoeFactory,
                            "create_fused_moe",
                            return_value=backend,
                        ),
                        patch.object(generic_moe, "DenseMLP") as shared_mlp,
                    ):
                        layer = qwen3_next.Qwen3NextDecoderLayer(
                            config, ParallelismConfig(), weights, 0, MoeConfig()
                        )
                        self.assertIsInstance(layer.mlp, generic_moe.GenericMoeLayer)
                        self.assertIsNone(layer.mlp.shared_expert)
                        hidden = torch.randn(3, 8)
                        residual = torch.randn_like(hidden)
                        output, result_residual = layer(hidden, residual, None)
                    shared_mlp.assert_not_called()
                    self.assertEqual(backend.calls, 1)
                    torch.testing.assert_close(output, hidden + 1)
                    self.assertIs(result_residual, residual)


@skipUnless(torch.cuda.is_available(), "CUDA metadata consumer required")
class Qwen3NextHostConvMetadataTest(TestCase):
    def inputs(self, lengths):
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        attention = PyAttentionInputs()
        attention.is_prefill = True
        attention.is_target_verify = False
        attention.is_cuda_graph = False
        host = torch.zeros(len(lengths) + 1, dtype=torch.int32, device="cpu")
        host[1:] = torch.tensor(lengths, dtype=torch.int32, device="cpu").cumsum(0)
        attention.input_lengths = torch.tensor(lengths, dtype=torch.int32, device="cpu")
        attention.cu_seqlens = host
        attention.cu_seqlens_device = host.to("cuda")
        attention.prefix_lengths = torch.zeros(
            len(lengths), dtype=torch.int32, device="cpu"
        )
        return attention

    def model(self, cache=None):
        return SimpleNamespace(
            kv_cache=cache,
            layers=[SimpleNamespace(layer_type=HybridAttentionType.LINEAR)],
            parallelism_config=SimpleNamespace(
                prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
            ),
        )

    def build(self, attention, graph=False, cache=None):
        return qwen3_next.Qwen3NextModel._build_attn_meta(
            self.model(cache),
            SimpleNamespace(attention_inputs=attention),
            torch.device("cuda"),
            is_cuda_graph=graph,
        ).prefill_conv1d_meta

    def test_host_metadata_full_conv_consumer_on_nondefault_stream(self):
        from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_fn

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.device("cuda"):
            for lengths in ([7, 17], [31, 65], [0, 17], [127, 131]):
                attention = self.inputs(lengths)
                with patch.object(
                    qwen3_next,
                    "prepare_causal_conv1d_metadata",
                    wraps=qwen3_next.prepare_causal_conv1d_metadata,
                ) as prepare:
                    metadata = self.build(attention)
                self.assertIs(
                    prepare.call_args.kwargs["query_start_loc"], attention.cu_seqlens
                )
                prefixes = [0, 16]
                blocks = [
                    (prefix + length + 15) // 16
                    for prefix, length in zip(prefixes, lengths)
                ]
                table = torch.full(
                    (2, max(blocks)), -1, device="cuda", dtype=torch.int32
                )
                offset = 0
                for row, count in enumerate(blocks):
                    table[row, :count] = torch.arange(
                        offset, offset + count, device="cuda", dtype=torch.int32
                    )
                    offset += count
                initial = torch.randn(
                    offset, 3, 256, device="cuda", dtype=torch.float32
                )
                x = torch.randn(sum(lengths), 256, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(256, 4, device="cuda", dtype=torch.float32)
                prefix = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
                results = []
                for meta in (None, metadata):
                    states = initial.clone()
                    output = causal_conv1d_fn(
                        x.clone().T,
                        weight,
                        None,
                        states.transpose(1, 2),
                        attention.cu_seqlens_device,
                        table,
                        prefix,
                        16,
                        metadata=meta,
                    )
                    results.append((output, states))
                torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
                torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
        stream.synchronize()

    def test_empty_mirror_shape_mismatch_and_graph_use_live_device_metadata(self):
        for mode in ("empty", "shape", "device_lengths", "graph_arg", "graph_field"):
            attention = self.inputs([7, 17])
            if mode == "empty":
                attention.cu_seqlens = torch.empty(0, dtype=torch.int32, device="cpu")
            elif mode == "shape":
                attention.cu_seqlens = torch.tensor(
                    [0, 24], dtype=torch.int32, device="cpu"
                )
            elif mode == "device_lengths":
                attention.input_lengths = attention.input_lengths.cuda()
                attention.cu_seqlens = torch.tensor(
                    [0, 1, 24], dtype=torch.int32, device="cpu"
                )
            else:
                attention.cu_seqlens = torch.tensor(
                    [0, 1, 24], dtype=torch.int32, device="cpu"
                )
                attention.is_cuda_graph = mode == "graph_field"
            with patch.object(
                qwen3_next,
                "prepare_causal_conv1d_metadata",
                wraps=qwen3_next.prepare_causal_conv1d_metadata,
            ) as prepare:
                metadata = self.build(attention, graph=mode == "graph_arg")
            self.assertIs(
                prepare.call_args.kwargs["query_start_loc"], attention.cu_seqlens_device
            )
            self.assertEqual(metadata.batch_ptr.tolist(), [0, 1, 1, 1, -1])
            self.assertEqual(metadata.token_chunk_offset_ptr.tolist(), [0, 0, 1, 2, -1])

    def test_tagged_linear_group_owns_the_host_partition(self):
        mha = self.inputs([1, 23])
        gdn = self.inputs([7, 17])
        cache = SimpleNamespace(
            get_layer_cache_groups=lambda _: [SimpleNamespace(tag="gdn")]
        )
        with patch.object(
            qwen3_next,
            "prepare_causal_conv1d_metadata",
            wraps=qwen3_next.prepare_causal_conv1d_metadata,
        ) as prepare:
            metadata = self.build({"mha": mha, "gdn": gdn}, cache=cache)
        self.assertIs(prepare.call_args.kwargs["query_start_loc"], gdn.cu_seqlens)
        self.assertEqual(metadata.batch_ptr.tolist(), [0, 1, 1, 1, -1])


if __name__ == "__main__":
    main()
