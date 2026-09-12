import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.distributed.sequence_parallel import sequence_parallel_layout
from rtp_llm.models_py.model_desc.kimi_k3_mtp import (
    KimiK3MtpLayer,
    KimiK3MtpModel,
    mtp_positions,
)


class _Attention(nn.Module):
    def tp_input_projection_weights(self):
        return [torch.eye(4)]

    def output_projection_weight(self):
        return torch.eye(4)

    def forward(self, x, *args, **kwargs):
        return x * 0.3


class _Moe(nn.Linear):
    def forward(self, x, **_kwargs):
        return super().forward(x)


class KimiK3MtpContractTest(unittest.TestCase):
    def test_weight_manifest_uses_config_source_layer_for_every_tp_rank(self):
        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.model_loader.weight_module import CompositeWeight
        from rtp_llm.models.kimi_k3.fp8_weight import KimiK3LoadFp8Weight
        from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3Mtp
        from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3MtpWeight
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig

        def names(info):
            pending = list(info.weights)
            for layer in info.layer_weights:
                pending.extend(layer)
            result = []
            while pending:
                weight = pending.pop()
                if isinstance(weight, CompositeWeight):
                    pending.extend(weight.sub_weights.values())
                    self.assertNotIsInstance(weight, KimiK3LoadFp8Weight)
                else:
                    result.extend(checkpoint.name for checkpoint in weight.weights)
            return result

        for source in (4, 47, 93):
            text = dict(
                num_hidden_layers=source,
                num_nextn_predict_layers=1,
                hidden_size=64,
                vocab_size=128,
                max_position_embeddings=128,
                intermediate_size=32,
                num_attention_heads=8,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_nope_head_dim=8,
                qk_rope_head_dim=8,
                v_head_dim=8,
                num_experts=16,
                num_experts_per_token=2,
                moe_intermediate_size=32,
                routed_expert_hidden_size=32,
                num_shared_experts=1,
                hidden_act="situ",
                first_k_dense_replace=1,
                moe_layer_freq=1,
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                attn_res_block_size=4,
                mla_use_nope=True,
                mla_use_output_gate=True,
                linear_attn_config=dict(
                    num_heads=8,
                    head_dim=8,
                    full_attn_layers=[source + 1],
                    kda_layers=[],
                ),
            )
            config = KimiK3Mtp._from_config_json(
                dict(model_type="kimi_k3", text_config=text)
            )
            expected = list(KimiK3MtpWeight.expected_checkpoint_tensor_names(config))
            self.assertTrue(
                all(
                    name.startswith(f"language_model.model.layers.{source}.")
                    for name in expected
                )
            )
            self.assertEqual(config.num_layers, 1)
            self.assertEqual(config.moe_layer_index, [0])
            self.assertEqual(config.k3_runtime_config.mtp_source_layer, source)
            config.data_type = "bf16"
            config.k3_attention_quant_config = None
            for tp in (1, 2, 4, 8):
                for rank in range(tp):
                    with self.subTest(source=source, tp=tp, rank=rank):
                        parallel = ParallelismConfig()
                        parallel.tp_size = parallel.ep_size = tp
                        parallel.world_size = parallel.local_world_size = tp
                        parallel.tp_rank = parallel.ep_rank = rank
                        manifest = KimiK3MtpWeight(
                            config, parallel, HWKernelConfig(), KVCacheConfig()
                        )._get_weight_info()
                        checkpoint_names = names(manifest)
                        self.assertTrue(checkpoint_names)
                        self.assertTrue(
                            all(
                                name.startswith(
                                    f"language_model.model.layers.{source}."
                                )
                                for name in checkpoint_names
                            )
                        )
                        self.assertFalse(
                            any("{i}" in name for name in checkpoint_names)
                        )
                        self.assertTrue(
                            any(
                                name.endswith("shared_head.norm.weight")
                                for name in checkpoint_names
                            )
                        )

    def test_weight_loader_rejects_injected_quantization(self):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig
        from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3MtpWeight
        from rtp_llm.ops import KvCacheDataType, QuantAlgo

        quant_algo = QuantAlgo()
        quant_algo.setQuantAlgo("fp8", 8, 128)
        for field, value in (
            ("data_type", "fp16"),
            ("quant_algo", quant_algo),
            ("quant_config", Fp8BlockWiseQuantConfig()),
            ("k3_attention_quant_config", Fp8BlockWiseQuantConfig()),
            ("mla_fp8_compute", True),
            ("kv_cache_dtype", KvCacheDataType.FP8),
            ("kv_cache_dtype", KvCacheDataType.INT8),
        ):
            with self.subTest(field=field, value=value):
                config = KimiK3ModelConfig()
                config.data_type = "bf16"
                config.quant_config = config.k3_attention_quant_config = None
                config.attn_config.kv_cache_dtype = KvCacheDataType.BASE
                if field in ("mla_fp8_compute", "kv_cache_dtype"):
                    setattr(config.attn_config, field, value)
                else:
                    setattr(config, field, value)
                weight = KimiK3MtpWeight.__new__(KimiK3MtpWeight)
                weight.model_config = config
                with self.assertRaisesRegex(ValueError, "native BF16 attention"):
                    weight._get_weight_info()

    def test_dense_attention_dispatch_quantizes_only_fp8_compute(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            flashmla_dense_prefill as dense,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            mla_fp8_kernels,
        )

        q, k, v = (torch.randn(2, 1, 4, dtype=torch.bfloat16) for _ in range(3))
        ptr = torch.tensor([0, 2], dtype=torch.int32)
        for enabled in (False, True):
            op = dense.MlaFlashMLAPrefillOp.__new__(dense.MlaFlashMLAPrefillOp)
            op.num_heads, op.v_head_dim, op.scale, op.q_scale = 1, 4, 0.5, 1.0
            op.fp8_compute = enabled
            op.flash_mla_cuda = MagicMock()
            op.tokenspeed_prefill = MagicMock(
                return_value=(torch.zeros_like(q), torch.zeros(2, 1))
            )
            with patch.object(
                dense, "_workspace", return_value=torch.empty(0)
            ), patch.object(
                mla_fp8_kernels, "quantize_fp8", side_effect=lambda x, *a, **kw: x
            ) as quantize:
                op._run_dense_attention(
                    q,
                    k,
                    v,
                    qo_indptr=ptr,
                    kv_indptr=ptr,
                    max_q_len=2,
                    max_kv_len=2,
                    causal=True,
                )
                self.assertEqual(quantize.call_count, 3 if enabled else 0)
                self.assertEqual(op.tokenspeed_prefill.call_count, int(enabled))
                self.assertEqual(
                    op.flash_mla_cuda.dense_prefill_fwd.call_count, int(not enabled)
                )
                if not enabled:
                    args = op.flash_mla_cuda.dense_prefill_fwd.call_args.args
                    for actual, expected in zip(args[1:4], (q, k, v)):
                        self.assertIs(actual, expected)
                        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_factory_binds_target_before_model_creation_and_retains_draft_norm(self):
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.ops import ParallelismConfig, SpeculativeType
        from rtp_llm.utils.model_weight import W

        target_weights = {W.embedding: torch.randn(8, 4), W.lm_head: torch.randn(4, 8)}
        draft_norm = torch.randn(4)
        draft_weights = {
            key: torch.zeros_like(value) for key, value in target_weights.items()
        }
        draft_weights[W.final_ln_gamma] = draft_norm
        target = SimpleNamespace(
            weight=SimpleNamespace(get_global_weight=target_weights.__getitem__)
        )
        draft = SimpleNamespace(
            weight=SimpleNamespace(
                get_global_weight=draft_weights.__getitem__,
                set_global_weight=draft_weights.__setitem__,
            )
        )

        def create_python():
            self.assertIs(draft_weights[W.embedding], target_weights[W.embedding])
            self.assertIs(draft_weights[W.lm_head], target_weights[W.lm_head])
            self.assertIs(draft_weights[W.final_ln_gamma], draft_norm)

        draft._create_python_model = MagicMock(side_effect=create_python)
        model_cls = SimpleNamespace(from_config=MagicMock(return_value=draft))
        target_config = SimpleNamespace(
            model_type="kimi_k3",
            max_seq_len=32,
            gen_num_per_cycle=3,
            mm_model_config=SimpleNamespace(is_multimodal=False),
        )
        propose_config = SimpleNamespace(model_type="kimi_k3_mtp")
        engine = MagicMock()
        engine.parallelism_config = ParallelismConfig()
        engine.sp_config.type = SpeculativeType.MTP
        engine.sp_config.gen_num_per_cycle = 3
        with patch.object(ModelFactory, "get_model_cls", return_value=model_cls):
            result = ModelFactory.get_sp_model(
                target_config, propose_config, engine, target
            )
        self.assertIs(result.model, draft)
        self.assertEqual(result.gen_num_per_circle, 3)
        self.assertTrue(model_cls.from_config.call_args.kwargs["skip_python_model"])
        draft._create_python_model.assert_called_once()

    def test_eagle_factory_keeps_replicated_cache_without_mutating_target(self):
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.ops import ParallelismConfig, SpeculativeType

        for tp_size in (8, 16):
            for rank in range(tp_size):
                with self.subTest(tp_size=tp_size, rank=rank):
                    parallel = ParallelismConfig()
                    parallel.tp_size = parallel.ep_size = tp_size
                    parallel.tp_rank = parallel.ep_rank = rank
                    parallel.prefill_cp_config.kv_cache_sharded = True
                    parallel.prefill_cp_config.prefill_cp_size = tp_size
                    engine = MagicMock()
                    engine.parallelism_config = parallel
                    engine.sp_config.type = SpeculativeType.EAGLE3
                    engine.sp_config.gen_num_per_cycle = 3
                    target = SimpleNamespace(
                        model_type="kimi_k3", max_seq_len=32, gen_num_per_cycle=3
                    )
                    draft = SimpleNamespace(model_type="kimi_k3_mla_swa_eagle3")
                    loaded = []

                    def load_model(**kwargs):
                        loaded.append(kwargs["parallelism_config"])
                        return SimpleNamespace()

                    with patch.object(
                        ModelFactory,
                        "get_model_cls",
                        return_value=SimpleNamespace(from_config=load_model),
                    ):
                        ModelFactory.get_sp_model(target, draft, engine)
                    self.assertFalse(loaded[0].kv_page_rr_enabled())
                    self.assertEqual(loaded[0].tp_size, tp_size)
                    self.assertEqual(loaded[0].tp_rank, rank)
                    self.assertTrue(parallel.kv_page_rr_enabled())
                    self.assertEqual(
                        parallel.prefill_cp_config.prefill_cp_size, tp_size
                    )
                    # Native MTP retains the target's FULL Page-RR placement.
                    mtp = ModelFactory._propose_parallelism_config(parallel)
                    self.assertTrue(mtp.kv_page_rr_enabled())

    def test_factory_rejects_wrong_mode_and_cp_before_loading(self):
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.ops import SpeculativeType

        target = SimpleNamespace(model_type="kimi_k3")
        draft = SimpleNamespace(model_type="kimi_k3_mtp")
        engine = MagicMock()
        engine.sp_config.type = SpeculativeType.EAGLE3
        with self.assertRaisesRegex(ValueError, "requires SP_MODEL_TYPE"):
            ModelFactory.get_sp_model(target, draft, engine)
        engine.sp_config.type = SpeculativeType.MTP
        engine.parallelism_config.prefill_cp_config.is_enabled.return_value = False
        engine.parallelism_config.prefill_cp_config.is_prefill_enabled.return_value = (
            True
        )
        with self.assertRaisesRegex(ValueError, "not Prefill CP"):
            ModelFactory.get_sp_model(target, draft, engine)
        engine.sp_config.type = SpeculativeType.NONE
        self.assertIsNone(ModelFactory.get_sp_model(target, None, engine))

    def test_fusion_residual_and_two_recursive_steps_match_vllm_equations(self):
        torch.manual_seed(31)
        layer = KimiK3MtpLayer.__new__(KimiK3MtpLayer)
        nn.Module.__init__(layer)
        for name in ("enorm", "hnorm", "input_norm", "post_norm"):
            setattr(layer, name, nn.RMSNorm(4, eps=1e-5))
        layer.attn_tp_size = 1
        layer.eh_proj = nn.Linear(8, 4, bias=False)
        layer.attention = _Attention()
        layer.moe = _Moe(4, 4, bias=False)
        e, previous = torch.randn(3, 4), torch.randn(3, 4)
        positions = torch.tensor([0, 5, 9])
        sp_layout = sequence_parallel_layout(
            mode="decode",
            logical_requests=3,
            physical_requests=3,
            tokens_per_request=1,
            logical_tokens=3,
            physical_tokens=3,
            world_size=1,
            rank=0,
        )
        expected_previous = previous
        with patch(
            "rtp_llm.models_py.model_desc.kimi_k3_mtp.all_gather_gemm",
            side_effect=lambda x, _weights, *, logical_m: [x[:logical_m]],
        ) as all_gather_gemm_mock, patch(
            "rtp_llm.models_py.model_desc.kimi_k3_mtp.gemm_reduce_scatter",
            side_effect=lambda x, *_args, **_kwargs: x,
        ) as gemm_reduce_scatter_mock, patch(
            "rtp_llm.models_py.model_desc.kimi_k3_mtp.get_process_group",
            return_value=object(),
        ) as get_process_group_mock:
            for step in range(2):
                p = positions + step
                masked = torch.where(p[:, None] == 0, 0, e)
                fused = F.linear(
                    torch.cat(
                        (
                            F.rms_norm(masked, (4,), eps=1e-5),
                            F.rms_norm(expected_previous, (4,), eps=1e-5),
                        ),
                        -1,
                    ),
                    layer.eh_proj.weight,
                )
                residual = fused + F.rms_norm(fused, (4,), eps=1e-5) * 0.3
                expected = residual + F.linear(
                    F.rms_norm(residual, (4,), eps=1e-5), layer.moe.weight
                )
                actual = layer(
                    e,
                    previous,
                    p,
                    None,
                    None,
                    None,
                    sp_layout=sp_layout,
                )
                torch.testing.assert_close(actual, expected)
                previous, expected_previous = actual, expected
        all_gather_gemm_mock.assert_not_called()
        gemm_reduce_scatter_mock.assert_not_called()
        get_process_group_mock.assert_not_called()

    def test_nope_positions_preserve_prefix_and_request_boundaries(self):
        inputs = SimpleNamespace(
            input_ids=torch.zeros(7),
            combo_position_ids=torch.empty(0),
            attention_inputs=SimpleNamespace(
                sequence_lengths=torch.tensor([8]),
                input_lengths=torch.tensor([99, 3, 3]),
                prefix_lengths=torch.tensor([0, 17]),
            ),
        )
        torch.testing.assert_close(
            mtp_positions(inputs), torch.tensor([8, 0, 1, 2, 17, 18, 19])
        )

    def test_decode_graph_positions_allow_absent_prefix_lengths(self):
        inputs = SimpleNamespace(
            input_ids=torch.zeros(3),
            combo_position_ids=None,
            attention_inputs=SimpleNamespace(
                is_prefill=False,
                sequence_lengths=torch.tensor([0, 8, 21]),
                input_lengths=torch.ones(3, dtype=torch.int32),
                prefix_lengths=None,
            ),
        )
        torch.testing.assert_close(mtp_positions(inputs), torch.tensor([0, 8, 21]))
        # Graph replay updates the existing sequence-length buffer.
        inputs.attention_inputs.sequence_lengths.add_(1)
        torch.testing.assert_close(mtp_positions(inputs), torch.tensor([1, 9, 22]))

    def test_prefill_graph_positions_ignore_decode_scratch_lengths(self):
        inputs = SimpleNamespace(
            input_ids=torch.zeros(6),
            combo_position_ids=None,
            attention_inputs=SimpleNamespace(
                is_prefill=True,
                sequence_lengths=torch.tensor([999, 999]),
                input_lengths=torch.tensor([3, 3]),
                prefix_lengths=torch.tensor([0, 17]),
            ),
        )
        torch.testing.assert_close(
            mtp_positions(inputs), torch.tensor([0, 1, 2, 17, 18, 19])
        )
        inputs.attention_inputs.sequence_lengths = None
        inputs.attention_inputs.prefix_lengths = None
        torch.testing.assert_close(
            mtp_positions(inputs), torch.tensor([0, 1, 2, 0, 1, 2])
        )

    def test_multimodal_uses_placeholder_embedding_without_visual_injection(self):
        model = KimiK3MtpModel.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.media_token_id = 7
        model.embedding = nn.Embedding(10, 4)
        # Packed requests [0, 4), [4, 8). Features start inside a reused
        # prefix and at the second request boundary. IDs are already shifted.
        features = [torch.full((3, 4), 100.0), torch.full((3, 4), 200.0)]
        inputs = SimpleNamespace(
            input_ids=torch.tensor([-99, 2, 3, 4, -88, -77, 5, 6]),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=features,
                mm_features_locs_host=torch.tensor([-1, 4]),
            ),
            attention_inputs=SimpleNamespace(
                cu_seqlens=torch.tensor([0, 4, 8]),
                cu_seqlens_host=torch.tensor([0, 4, 8]),
            ),
        )
        expected = model.embedding(torch.tensor([7, 2, 3, 4, 7, 7, 5, 6]))
        actual = model._embed_shifted_tokens(inputs)
        torch.testing.assert_close(actual, expected)
        # Input cache-key tokens belong to the target and must remain intact.
        self.assertEqual(inputs.input_ids[0].item(), -99)

    def test_model_returns_norm_for_logits_but_preserves_recurrent_state(self):
        model = KimiK3MtpModel.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.hidden_size = 4
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 1,
            get_attn_tp_rank=lambda: 0,
        )
        model._decode_role = True
        model._recurrent = torch.empty(8, 4)
        address = model._recurrent.data_ptr()
        model.kv_cache = None
        model.final_norm = nn.RMSNorm(4, eps=1e-5)
        h = torch.arange(12, dtype=torch.float32).view(3, 4) + 2
        inputs = SimpleNamespace(
            input_ids=torch.zeros(3, dtype=torch.long),
            input_hiddens=h,
            combo_position_ids=torch.arange(3),
            attention_inputs=SimpleNamespace(
                is_prefill=False,
                is_target_verify=False,
                input_lengths=torch.ones(3, dtype=torch.int32),
            ),
        )
        layer = MagicMock(return_value=h)
        with patch.object(model, "_embed_shifted_tokens", return_value=h), patch.object(
            model, "layer", layer, create=True
        ), patch(
            "rtp_llm.models_py.model_desc.kimi_k3_mtp.select_block_map_for_layer"
        ), patch(
            "rtp_llm.models_py.model_desc.kimi_k3_mtp.PyModelOutputs",
            side_effect=lambda z, _: z,
        ), patch(
            "rtp_llm.models_py.model_desc.kimi_k3_mtp.all_gather_trim",
            side_effect=lambda z, *_args, **_kwargs: z,
        ):
            z = model.forward(inputs, SimpleNamespace(fmha_params=None))
        layout = layer.call_args.kwargs["sp_layout"]
        self.assertEqual(layout.tokens.logical_tokens, 3)
        self.assertEqual(layout.tokens.physical_tokens, 3)
        torch.testing.assert_close(z, F.rms_norm(h, (4,), eps=1e-5))
        torch.testing.assert_close(model.get_mtp_target_hidden_states(3), h)
        torch.testing.assert_close(model.get_mtp_target_hidden_states(-1), h)
        self.assertEqual(model._recurrent.data_ptr(), address)
        self.assertFalse(torch.allclose(z, model.get_mtp_target_hidden_states(3)))


if __name__ == "__main__":
    unittest.main()
