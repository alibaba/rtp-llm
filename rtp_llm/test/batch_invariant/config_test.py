"""Check startup parsing, spawn serialization, and an incompatible configuration."""

import os
import pickle
import unittest
from unittest.mock import patch

from rtp_llm.config.engine_config import EngineConfig, configure_batch_invariant_model
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.ops import CPRotateMethod, DataType, TaskType, VitSeparation
from rtp_llm.server.server_args.server_args import setup_args


def make_engine():
    env = {
        "RTP_BATCH_INVARIANT": "1",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "TP_SIZE": "1",
        "DP_SIZE": "1",
        "EP_SIZE": "1",
        "PP_SIZE": "1",
    }
    with patch.dict(os.environ, env, clear=True):
        return EngineConfig.create(setup_args([]))


class BatchInvariantConfigTest(unittest.TestCase):
    def test_switch_defaults_off_and_survives_spawn(self):
        with patch.dict(os.environ):
            os.environ.pop("RTP_BATCH_INVARIANT", None)
            self.assertFalse(setup_args([]).py_hw_kernel_config.batch_invariant)
            os.environ["RTP_BATCH_INVARIANT"] = "1"
            parsed = setup_args([])
            os.environ["RTP_BATCH_INVARIANT"] = "0"
            engine = EngineConfig.create(pickle.loads(pickle.dumps(parsed)))
        self.assertTrue(engine.hw_kernel_config.batch_invariant)
        self.assertTrue(engine.collective_config.batch_invariant)

    def test_invalid_switch(self):
        with patch.dict(os.environ, {"RTP_BATCH_INVARIANT": "true"}):
            with self.assertRaisesRegex(
                ValueError, "RTP_BATCH_INVARIANT must be 0 or 1"
            ):
                setup_args([])

    def test_m11_keeps_sparse_attention_and_native_mega_moe(self):
        engine = make_engine()
        engine.kv_cache_config.reuse_cache = False
        parallel = engine.parallelism_config
        parallel.world_size = parallel.local_world_size = parallel.dp_size = (
            parallel.ep_size
        ) = 8
        engine.moe_config.moe_strategy = "mega_moe"
        engine.moe_config.use_deepep_moe = False
        model = ModelConfig()
        model.model_type = "qwen4_exp"
        model.task_type = TaskType.LANGUAGE_MODEL
        model.data_type = "bf16"
        model.num_layers = 60
        model.hidden_size = 8192
        model.expert_num = 768
        model.moe_k = 10
        model.moe_inter_size = 2048
        model._qwen4_mxfp4_source = True
        model.vit_separation = VitSeparation.VIT_SEPARATION_REMOTE
        model.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
        attn = model.attn_config
        attn.head_num, attn.kv_head_num, attn.size_per_head = 64, 4, 256
        attn.is_sparse = attn.use_sparse_gqa_fmha = True
        attn.tokens_per_block = attn.kernel_tokens_per_block = 128
        checkpoint = {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen4_exp_text",
                "dtype": "bfloat16",
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
                "hc_count": 4,
                "ple_layer_ids": [2],
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
            },
        }
        with (
            patch("rtp_llm.config.engine_config.is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
            patch(
                "rtp_llm.config.engine_config.get_config_from_path",
                return_value=checkpoint,
            ),
        ):
            configure_batch_invariant_model(engine, model)
            self.assertEqual(engine.moe_config.moe_strategy, "mega_moe")
            self.assertTrue(attn.use_sparse_gqa_fmha)
            self.assertEqual(attn.kernel_tokens_per_block, 128)
            attn.kernel_tokens_per_block = 64
            with self.assertRaisesRegex(ValueError, "requires 128-token"):
                configure_batch_invariant_model(engine, model)
            attn.kernel_tokens_per_block = 128
            model._qwen4_mxfp4_source = False
            with self.assertRaisesRegex(ValueError, "native Qwen4-Max M11"):
                configure_batch_invariant_model(engine, model)

    def test_model_conflicts(self):
        engine = make_engine()
        engine.kv_cache_config.reuse_cache = True
        model = ModelConfig()
        model.model_type = "qwen35_moe"
        model.task_type = TaskType.LANGUAGE_MODEL
        model.num_layers = 40
        model.hidden_size = 2048
        model.expert_num = 256
        model.moe_k = 8
        model.moe_inter_size = 512
        model.data_type = "bf16"
        model.attn_config.head_num = 16
        model.attn_config.kv_head_num = 2
        model.attn_config.size_per_head = 256
        model.attn_config.tokens_per_block = 64
        model.attn_config.kernel_tokens_per_block = 64
        model.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
        with (
            patch("rtp_llm.config.engine_config.is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
            patch("rtp_llm.config.engine_config.get_config_from_path") as read_config,
        ):
            read_config.return_value = {"text_config": {"dtype": "bfloat16"}}
            with self.assertRaisesRegex(ValueError, "prefix reuse"):
                configure_batch_invariant_model(engine, model)
            engine.kv_cache_config.reuse_cache = False
            read_config.return_value["text_config"]["dtype"] = "float16"
            with self.assertRaisesRegex(ValueError, "requires a BF16 checkpoint"):
                configure_batch_invariant_model(engine, model)
            read_config.return_value["text_config"]["dtype"] = "bfloat16"
            read_config.return_value["torch_dtype"] = "float16"
            model.linear_attention_config.ssm_state_dtype = DataType.TYPE_BF16
            with self.assertRaisesRegex(ValueError, "requires FP32 SSM state"):
                configure_batch_invariant_model(engine, model)
            model.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
            model.mm_model_config.is_multimodal = True
            configure_batch_invariant_model(engine, model)
            model.quant_config = Fp8BlockWiseQuantConfig()
            configure_batch_invariant_model(engine, model)
            engine.moe_config.moe_strategy = "fp8_per_block_no_dp_masked"
            with self.assertRaisesRegex(ValueError, "local DeepGEMM MoE"):
                configure_batch_invariant_model(engine, model)
            engine.moe_config.moe_strategy = "auto"
            model.quant_config = Fp8BlockWiseQuantConfig(group_size=64)
            with self.assertRaisesRegex(ValueError, "FP8_PER_BLOCK with 128x128"):
                configure_batch_invariant_model(engine, model)
            parallel = engine.parallelism_config
            parallel.world_size = parallel.local_world_size = 4
            parallel.dp_size = parallel.ep_size = 4
            engine.moe_config.use_all_gather = False
            engine.moe_config.use_deepep_moe = True
            model.quant_config = Fp8BlockWiseQuantConfig()
            configure_batch_invariant_model(engine, model)
            self.assertEqual(engine.moe_config.moe_strategy, "fp8_per_block_pure_dp")
            self.assertTrue(engine.moe_config.use_all_gather)
            self.assertFalse(engine.moe_config.use_deepep_moe)
            model.quant_config = None
            engine.moe_config.moe_strategy = "auto"
            configure_batch_invariant_model(engine, model)
            self.assertEqual(engine.moe_config.moe_strategy, "no_quant_pure_dp")
            engine.moe_config.moe_strategy = "auto"
            parallel.world_size = parallel.local_world_size = 1
            parallel.dp_size = parallel.ep_size = 1
            model.quant_config = None
            model.num_layers = 60
            model.hidden_size = 4096
            model.expert_num = 512
            model.moe_k = 10
            model.moe_inter_size = 1024
            model.attn_config.head_num = 32
            with self.assertRaisesRegex(
                ValueError, "Qwen3.5-397B-A17B requires .*TP_SIZE=4"
            ):
                configure_batch_invariant_model(engine, model)
            parallel.world_size = parallel.local_world_size = parallel.tp_size = 4
            parallel.ffn_tp_size = 4
            configure_batch_invariant_model(engine, model)
            parallel.ep_size = 4
            configure_batch_invariant_model(engine, model)
            model.quant_config = Fp8BlockWiseQuantConfig()
            configure_batch_invariant_model(engine, model)
            self.assertEqual(engine.moe_config.moe_strategy, "auto")
            parallel.ep_size = 2
            with self.assertRaisesRegex(ValueError, "DP_SIZE=1/EP_SIZE=4"):
                configure_batch_invariant_model(engine, model)
            parallel.ep_size = 1
            parallel.prefill_cp_config.method = CPRotateMethod.PREFILL_CP
            with self.assertRaisesRegex(ValueError, "CP disabled"):
                configure_batch_invariant_model(engine, model)
        self.assertFalse(model.mm_model_config.is_multimodal)
        self.assertFalse(model.enable_fp32_lm_head)

    def test_standard_qwen3_constraints(self):
        for model_type in ("qwen_3", "qwen_3_moe"):
            with self.subTest(model_type=model_type):
                engine = make_engine()
                model = ModelConfig()
                model.model_type = model_type
                model.task_type = TaskType.LANGUAGE_MODEL
                model.data_type = "bf16"
                model.linear_attention_config.ssm_state_dtype = DataType.TYPE_BF16
                attn = model.attn_config
                attn.head_num = 32
                attn.size_per_head = 128
                attn.tokens_per_block = attn.kernel_tokens_per_block = 64
                if model_type == "qwen_3":
                    model.num_layers = 36
                    model.hidden_size = 4096
                    model.inter_size = 12288
                    attn.kv_head_num = 8
                else:
                    model.num_layers = 48
                    model.hidden_size = 2048
                    model.expert_num = 128
                    model.moe_k = 8
                    model.moe_inter_size = 768
                    attn.kv_head_num = 4
                with (
                    patch("rtp_llm.config.engine_config.is_cuda", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=(10, 3)),
                    patch(
                        "rtp_llm.config.engine_config.get_config_from_path"
                    ) as read_config,
                ):
                    read_config.return_value = {
                        "torch_dtype": "bfloat16",
                        "text_config": {"dtype": "float16"},
                    }
                    configure_batch_invariant_model(engine, model)
                    self.assertFalse(model.enable_fp32_lm_head)
                    engine.moe_config.use_deepep_moe = True
                    if model_type == "qwen_3_moe":
                        with self.assertRaisesRegex(ValueError, "local Triton MoE"):
                            configure_batch_invariant_model(engine, model)
                    else:
                        configure_batch_invariant_model(engine, model)
                    engine.moe_config.use_deepep_moe = False
                    read_config.return_value["dtype"] = "float16"
                    with self.assertRaisesRegex(
                        ValueError, "requires a BF16 checkpoint"
                    ):
                        configure_batch_invariant_model(engine, model)
                    read_config.return_value["dtype"] = "bf16"
                    configure_batch_invariant_model(engine, model)
                    engine.parallelism_config.tp_size = 2
                    with self.assertRaisesRegex(ValueError, "TP_SIZE=1"):
                        configure_batch_invariant_model(engine, model)
                    engine.parallelism_config.tp_size = 1
                    engine.parallelism_config.world_size = 4
                    engine.parallelism_config.local_world_size = 4
                    engine.parallelism_config.dp_size = 4
                    engine.parallelism_config.ep_size = 4
                    if model_type == "qwen_3_moe":
                        configure_batch_invariant_model(engine, model)
                        self.assertEqual(
                            engine.moe_config.moe_strategy, "no_quant_pure_dp"
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, "DP_SIZE=EP_SIZE=1"):
                            configure_batch_invariant_model(engine, model)
                    attn.kv_head_num = 2
                    with self.assertRaisesRegex(
                        ValueError, "requires one of these architectures"
                    ):
                        configure_batch_invariant_model(engine, model)


def make_qwen38(layers=48):
    """Qwen3.8-Flash-Next model config, geometry taken from the shipped checkpoint."""
    model = ModelConfig()
    model.model_type = "qwen4_exp"
    model.task_type = TaskType.LANGUAGE_MODEL
    model.data_type = "bf16"
    model.num_layers = layers
    model.hidden_size = 2560
    model.expert_num = 512
    model.moe_k = 10
    model.moe_inter_size = 640
    model.vit_separation = VitSeparation.VIT_SEPARATION_LOCAL
    model.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
    attn = model.attn_config
    attn.head_num, attn.kv_head_num, attn.size_per_head = 24, 2, 256
    attn.tokens_per_block = attn.kernel_tokens_per_block = 128
    attn.is_sparse = True
    attn.use_sparse_gqa_fmha = True
    return model


def qwen38_checkpoint(layers=48):
    period = ["linear_attention"] * 3 + ["full_attention"]
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "dtype": "bfloat16",
            "model_type": "qwen4_exp_text",
            "layer_types": period * (layers // 4),
            "hc_count": 4,
            "ple_layer_ids": [2],
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
        },
    }


def qwen38_engine(tp, dp, ep, *, cuda_graph=False, reuse_cache=False):
    engine = make_engine()
    parallel = engine.parallelism_config
    parallel.world_size = parallel.local_world_size = max(tp * dp, ep)
    parallel.tp_size = parallel.ffn_tp_size = tp
    parallel.dp_size = dp
    parallel.ep_size = ep
    parallel.pp_size = 1
    engine.moe_config.use_deepep_moe = False
    engine.moe_config.use_all_gather = True
    engine.moe_config.moe_strategy = "auto"
    engine.hw_kernel_config.enable_cuda_graph = cuda_graph
    engine.kv_cache_config.reuse_cache = reuse_cache
    return engine


class Qwen38FlashNextBatchInvariantTest(unittest.TestCase):
    """The Qwen3.8-Flash-Next BF16 profile: topologies, SM100, graph, prefix reuse."""

    def _patches(self, checkpoint, capability=(10, 3)):
        return (
            patch("rtp_llm.config.engine_config.is_cuda", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=capability),
            patch(
                "rtp_llm.config.engine_config.get_config_from_path",
                return_value=checkpoint,
            ),
        )

    def test_accepts_48l_and_8l_proxy_on_every_supported_topology(self):
        for layers in (48, 8):
            for tp, dp, ep in ((8, 1, 8), (4, 1, 4), (1, 8, 8)):
                model = make_qwen38(layers)
                engine = qwen38_engine(tp, dp, ep)
                patches = self._patches(qwen38_checkpoint(layers))
                with patches[0], patches[1], patches[2]:
                    configure_batch_invariant_model(engine, model)
                self.assertFalse(engine.hw_kernel_config.enable_cuda_graph)
                # TP == EP takes the Triton path over the capturable
                # PureTpRouter; DP/EP takes the fused BF16 MegaMoE kernel.
                self.assertEqual(
                    engine.moe_config.moe_strategy,
                    "bf16_mega_moe" if dp > 1 else "no_quant_cpp",
                )

    def test_accepts_sm100(self):
        model = make_qwen38()
        engine = qwen38_engine(8, 1, 8)
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint, capability=(10, 0))
        with patches[0], patches[1], patches[2]:
            configure_batch_invariant_model(engine, model)

    def test_rejects_unsupported_capability_and_topology(self):
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint, capability=(9, 0))
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(ValueError, "SM100 or SM103"):
                configure_batch_invariant_model(qwen38_engine(8, 1, 8), make_qwen38())
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(ValueError, "Qwen3.8-Flash-Next 48L requires"):
                configure_batch_invariant_model(qwen38_engine(2, 1, 2), make_qwen38())

    def test_cuda_graph_allowed_on_both_moe_backends(self):
        """Both Qwen3.8 MoE backends are capturable, for different reasons.

        TP == EP uses PureTpRouter, whose all-gather dispatch and fixed-order
        reduction are both capturable; DP/EP uses BF16 MegaMoE, which has no
        router to capture at all. The pure-DP rejection that used to cover
        DP/EP here now only applies to the backends that still use that router,
        which ``BatchInvariantConfigTest`` covers.
        """
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            engine = qwen38_engine(8, 1, 8, cuda_graph=True)
            configure_batch_invariant_model(engine, make_qwen38())
            self.assertTrue(engine.hw_kernel_config.enable_cuda_graph)
            self.assertEqual(engine.moe_config.moe_strategy, "no_quant_cpp")
            engine = qwen38_engine(1, 8, 8, cuda_graph=True)
            configure_batch_invariant_model(engine, make_qwen38())
            self.assertTrue(engine.hw_kernel_config.enable_cuda_graph)
            self.assertEqual(engine.moe_config.moe_strategy, "bf16_mega_moe")

    def test_local_prefix_reuse_allowed_but_external_caches_are_not(self):
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            engine = qwen38_engine(8, 1, 8, reuse_cache=True)
            configure_batch_invariant_model(engine, make_qwen38())
            self.assertTrue(engine.kv_cache_config.reuse_cache)
            engine = qwen38_engine(8, 1, 8, reuse_cache=True)
            engine.kv_cache_config.enable_disk_cache = True
            with self.assertRaisesRegex(ValueError, "external KV caches"):
                configure_batch_invariant_model(engine, make_qwen38())

    def test_rejects_mxfp4_source_and_quantization(self):
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            model = make_qwen38()
            model._qwen4_mxfp4_source = True
            with self.assertRaisesRegex(ValueError, "BF16 Qwen3.8-Flash-Next"):
                configure_batch_invariant_model(qwen38_engine(8, 1, 8), model)

    def test_accepts_either_vit_separation_but_not_a_vit_role(self):
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            for separation in (
                VitSeparation.VIT_SEPARATION_LOCAL,
                VitSeparation.VIT_SEPARATION_REMOTE,
            ):
                model = make_qwen38()
                model.vit_separation = separation
                configure_batch_invariant_model(qwen38_engine(8, 1, 8), model)
            model = make_qwen38()
            model.vit_separation = VitSeparation.VIT_SEPARATION_ROLE
            with self.assertRaisesRegex(ValueError, "no separate ViT"):
                configure_batch_invariant_model(qwen38_engine(8, 1, 8), model)

    def test_dp_ep_selects_bf16_mega_moe_and_keeps_cuda_graph(self):
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            for tp, dp, ep in ((1, 8, 8), (16, 16, 16), (32, 32, 32)):
                engine = qwen38_engine(1, dp, ep, cuda_graph=True, reuse_cache=True)
                configure_batch_invariant_model(engine, make_qwen38())
                self.assertEqual(engine.moe_config.moe_strategy, "bf16_mega_moe")
                self.assertTrue(engine.hw_kernel_config.enable_cuda_graph)
                # MegaMoE dispatches over NVLink, not the process group.
                self.assertFalse(engine.moe_config.use_all_gather)
                self.assertFalse(engine.moe_config.use_deepep_moe)
            # TP == EP keeps the Triton path and its all-gather dispatch.
            engine = qwen38_engine(8, 1, 8, cuda_graph=True)
            configure_batch_invariant_model(engine, make_qwen38())
            self.assertEqual(engine.moe_config.moe_strategy, "no_quant_cpp")
            self.assertTrue(engine.moe_config.use_all_gather)

    def test_multi_node_requires_dp_ep_inside_one_nvlink_domain(self):
        checkpoint = qwen38_checkpoint()
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            # DP16/EP16 across two nodes of a 72-rank NVLink domain is fine.
            engine = qwen38_engine(1, 16, 16, cuda_graph=True)
            engine.parallelism_config.local_world_size = 8
            configure_batch_invariant_model(engine, make_qwen38())
            self.assertEqual(engine.moe_config.moe_strategy, "bf16_mega_moe")
            # TP8/DP1/EP8 has no multi-node form: its MoE dispatch is an
            # all-gather over the process group.
            engine = qwen38_engine(8, 1, 8)
            engine.parallelism_config.local_world_size = 4
            with self.assertRaisesRegex(ValueError, "NVLink domain"):
                configure_batch_invariant_model(engine, make_qwen38())

    def test_rejects_wrong_layer_schedule(self):
        checkpoint = qwen38_checkpoint()
        checkpoint["text_config"]["layer_types"] = ["full_attention"] * 48
        patches = self._patches(checkpoint)
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(ValueError, "BF16 Qwen3.8-Flash-Next"):
                configure_batch_invariant_model(qwen38_engine(8, 1, 8), make_qwen38())


if __name__ == "__main__":
    unittest.main()
