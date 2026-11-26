import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.config.server_config_setup import setup_default_args
from rtp_llm.model_factory import configure_batch_invariant_model
from rtp_llm.models_py.determinism import batch_invariant_config as bi
from rtp_llm.ops import (
    DataType,
    KvCacheDataType,
    RoleType,
    SpeculativeType,
    TaskType,
    VitSeparation,
)


class BatchInvariantConfigTest(unittest.TestCase):
    def setUp(self):
        self.hw = SimpleNamespace(
            batch_invariant=True,
            enable_cuda_graph=False,
            enable_dynamic_decode_backend=True,
            enable_decode_cascade_attention=True,
        )
        self.parallel = SimpleNamespace(
            world_size=1,
            local_world_size=1,
            tp_size=1,
            ffn_tp_size=1,
            dp_size=1,
            ep_size=1,
            pp_size=1,
            role_type=RoleType.PDFUSION,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False, is_prefill_enabled=lambda: False
            ),
        )
        self.cache = SimpleNamespace(
            reuse_cache=False,
            enable_remote_cache=False,
            enable_disk_cache=False,
            enable_memory_cache=False,
            multi_task_prompt=None,
            multi_task_prompt_str=None,
            multi_task_prompt_tokens={},
        )
        self.engine = SimpleNamespace(
            hw_kernel_config=self.hw,
            parallelism_config=self.parallel,
            kv_cache_config=self.cache,
            sp_config=SimpleNamespace(type=SpeculativeType.NONE),
            runtime_config=SimpleNamespace(use_batch_decode_scheduler=False),
        )
        self.model = SimpleNamespace(
            model_type="qwen_3",
            num_layers=36,
            hidden_size=4096,
            inter_size=12288,
            attn_config=SimpleNamespace(
                head_num=32,
                kv_head_num=8,
                size_per_head=128,
                tokens_per_block=64,
                kernel_tokens_per_block=64,
                kv_cache_dtype=KvCacheDataType.BASE,
            ),
            config_dtype="bfloat16",
            data_type=DataType.TYPE_BF16,
            task_type=TaskType.LANGUAGE_MODEL,
            quant_config=None,
            lora_infos=[],
            enable_fp32_lm_head=True,
        )
        for name, result in (
            ("is_cuda", True),
            ("torch.cuda.get_device_capability", (10, 3)),
        ):
            patcher = patch(f"{bi.__name__}.{name}", return_value=result)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_disabled_leaves_regular_options(self):
        self.hw.batch_invariant = False
        configure_batch_invariant_model(self.engine, None)
        self.assertTrue(self.hw.enable_dynamic_decode_backend)
        self.assertTrue(self.hw.enable_decode_cascade_attention)

    def test_only_explicit_true_enables_validation(self):
        for engine in (
            SimpleNamespace(),
            SimpleNamespace(hw_kernel_config=MagicMock()),
        ):
            with self.subTest(engine=engine):
                configure_batch_invariant_model(engine, None)

    def _set_tp_size(self, size):
        for key in ("tp_size", "world_size", "local_world_size", "ffn_tp_size"):
            setattr(self.parallel, key, size)

    def _set_qwen3_moe(self):
        self.model.model_type = "qwen_3_moe"
        self.model.num_layers = 48
        self.model.hidden_size = 2048
        self.model.attn_config.kv_head_num = 4
        self.model.inter_size = 0
        self.model.moe_inter_size = 768
        self.model.expert_num = 128
        self.model.moe_k = 8
        self.model.eplb_config = SimpleNamespace(
            enable_eplb=lambda: False, redundant_expert=0
        )
        self.model.phy2log_path = ""
        self.engine.moe_config = SimpleNamespace(
            moe_strategy="auto",
            use_all_gather=True,
            use_deepep_moe=False,
            use_mori_ep=False,
            fake_balance_expert=False,
        )

    def _set_deepseek_flash(self, v41):
        self._set_qwen3_moe()
        tp = 4 if v41 else 2
        self._set_tp_size(tp)
        self.parallel.ep_size = tp
        self.model.model_type = "deepseek_v41" if v41 else "deepseek_v4"
        self.model.num_layers = 40 if v41 else 43
        self.model.hidden_size = 5120 if v41 else 4096
        self.model.expert_num = 384 if v41 else 256
        self.model.moe_inter_size = 2304 if v41 else 2048
        self.model.moe_k = 6
        self.model.hc_mult = 4
        self.model.quant_config = Fp8BlockWiseQuantConfig(
            group_size=32 if v41 else 128, is_quanted=True
        )
        self.model.mm_model_config = SimpleNamespace(is_multimodal=v41)
        self.model.vit_separation = VitSeparation.VIT_SEPARATION_LOCAL
        attn = self.model.attn_config
        attn.head_num, attn.kv_head_num, attn.size_per_head = 64, 1, 512
        attn.rope_head_dim, attn.indexer_head_dim = 64, 128
        attn.layer_compress_ratios = [0, 1, 2] if v41 else [0, 4, 128]
        attn.tokens_per_block = 128 if v41 else 256
        attn.kernel_tokens_per_block = 128
        attn.kv_cache_dtype = KvCacheDataType.FP8
        self.engine.moe_config.use_deepep_moe = True

    def test_deepseek_uses_native_quant_and_tp_aligned_experts(self):
        for v41 in (False, True):
            with self.subTest(v41=v41):
                self._set_deepseek_flash(v41)
                configure_batch_invariant_model(self.engine, self.model)
                self.assertEqual(self.engine.moe_config.moe_strategy, "auto")
                self.assertFalse(self.engine.moe_config.use_deepep_moe)
                self.assertFalse(self.model.mm_model_config.is_multimodal)
                self.assertFalse(self.model.enable_fp32_lm_head)

    def test_deepseek_rejects_other_layouts_before_applying_policy(self):
        self._set_deepseek_flash(v41=True)
        self.hw.enable_cuda_graph = True
        self.model.attn_config.kv_cache_dtype = KvCacheDataType.BASE
        self.parallel.ep_size = 1
        self.model.quant_config = Fp8BlockWiseQuantConfig(group_size=128)
        with self.assertRaises(ValueError) as raised:
            configure_batch_invariant_model(self.engine, self.model)
        for reason in ("eager", "FP8 KV", "TP/DP/EP", "native DeepSeek Flash"):
            self.assertIn(reason, str(raised.exception))
        self.assertTrue(self.model.mm_model_config.is_multimodal)
        self.assertTrue(self.engine.moe_config.use_deepep_moe)

    def test_qwen3_moe_uses_no_quant_cpp_before_weight_loading(self):
        self._set_qwen3_moe()
        for size in (1, 2, 4):
            self._set_tp_size(size)
            for strategy in ("auto", "no_quant_cpp"):
                with self.subTest(size=size, strategy=strategy):
                    self.engine.moe_config.moe_strategy = strategy
                    configure_batch_invariant_model(self.engine, self.model)
                    self.assertEqual(
                        self.engine.moe_config.moe_strategy, "no_quant_cpp"
                    )

    def test_disabled_qwen3_moe_preserves_automatic_selection(self):
        self._set_qwen3_moe()
        self.hw.batch_invariant = False
        configure_batch_invariant_model(self.engine, self.model)
        self.assertEqual(self.engine.moe_config.moe_strategy, "auto")

    def test_qwen3_moe_requires_30b_expert_shape(self):
        self._set_qwen3_moe()
        for key, value in (
            ("expert_num", 64),
            ("moe_k", 4),
            ("moe_inter_size", 512),
            ("inter_size", 768),
        ):
            with self.subTest(key=key), patch.object(self.model, key, value):
                with self.assertRaisesRegex(ValueError, "Qwen3-30B-A3B"):
                    configure_batch_invariant_model(self.engine, self.model)

    def test_qwen3_moe_rejects_incompatible_routes_and_expert_remapping(self):
        self._set_qwen3_moe()
        moe = self.engine.moe_config
        for config, key, value, reason in (
            (moe, "moe_strategy", "mega_moe", "local BF16 MoE"),
            (moe, "moe_strategy", "batch_invariant", "local BF16 MoE"),
            (moe, "use_all_gather", False, "USE_ALL_GATHER"),
            (moe, "use_deepep_moe", True, "DeepEP"),
            (moe, "use_mori_ep", True, "MoriEP"),
            (moe, "fake_balance_expert", True, "fake balancing"),
            (self.model.eplb_config, "enable_eplb", lambda: True, "EPLB"),
            (self.model.eplb_config, "redundant_expert", 1, "expert ordering"),
            (self.model, "phy2log_path", "expert_map.json", "expert ordering"),
            (self.parallel, "ep_size", 2, "DP=EP=PP=1"),
        ):
            with self.subTest(key=key), patch.object(config, key, value):
                with self.assertRaisesRegex(ValueError, reason):
                    configure_batch_invariant_model(self.engine, self.model)
            self.assertEqual(moe.moe_strategy, "auto")
            self.assertTrue(self.hw.enable_dynamic_decode_backend)

    def test_pure_tp_eager_uses_bf16_head_and_static_attention(self):
        for size in (1, 2, 4):
            with self.subTest(size=size):
                self._set_tp_size(size)
                self.model.enable_fp32_lm_head = True
                self.hw.enable_dynamic_decode_backend = True
                self.hw.enable_decode_cascade_attention = True
                configure_batch_invariant_model(self.engine, self.model)
                self.assertFalse(self.model.enable_fp32_lm_head)
                self.assertFalse(self.hw.enable_dynamic_decode_backend)
                self.assertFalse(self.hw.enable_decode_cascade_attention)

    def test_unsupported_tp_sizes_are_rejected(self):
        for size in (3, 8):
            with self.subTest(size=size):
                self._set_tp_size(size)
                with self.assertRaisesRegex(ValueError, "TP1, TP2 or TP4"):
                    configure_batch_invariant_model(self.engine, self.model)

    def test_tp_requires_single_node_matching_ffn_and_no_other_parallelism(self):
        self._set_tp_size(4)
        for key, value in (
            ("world_size", 8),
            ("local_world_size", 2),
            ("ffn_tp_size", 2),
            ("dp_size", 2),
            ("ep_size", 4),
            ("pp_size", 2),
            ("role_type", RoleType.DECODE),
        ):
            with self.subTest(key=key), patch.object(self.parallel, key, value):
                with self.assertRaisesRegex(ValueError, "single-node PDFUSION"):
                    configure_batch_invariant_model(self.engine, self.model)
        for key in ("is_enabled", "is_prefill_enabled"):
            with self.subTest(key=key), patch.object(
                self.parallel.prefill_cp_config, key, return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "context parallelism disabled"):
                    configure_batch_invariant_model(self.engine, self.model)

    def test_other_qwen3_shape_is_rejected(self):
        self.model.num_layers = 28
        with self.assertRaisesRegex(ValueError, "requires Qwen3-8B Dense"):
            configure_batch_invariant_model(self.engine, self.model)

    def test_bf16_checkpoint_aliases(self):
        for dtype in ("BF16", "BFloat16", "bp16"):
            with self.subTest(dtype=dtype):
                self.model.config_dtype = dtype
                configure_batch_invariant_model(self.engine, self.model)

    def test_missing_or_non_bf16_checkpoint_dtype_is_rejected(self):
        self.hw.enable_cuda_graph = True
        for dtype in (None, 16, "unknown", "float16"):
            with self.subTest(dtype=dtype):
                self.model.config_dtype = dtype
                with self.assertRaises(ValueError) as raised:
                    configure_batch_invariant_model(self.engine, self.model)
                self.assertIn("requires unquantized BF16", str(raised.exception))
                self.assertIn("requires eager execution", str(raised.exception))

    def test_headwise_checkpoint_is_rejected(self):
        self.model.headwise_config = {"sliding_window": 128}
        with self.assertRaisesRegex(ValueError, "headwise_config"):
            configure_batch_invariant_model(self.engine, self.model)

    def test_conflicts_are_reported_before_applying_overrides(self):
        self.hw.enable_cuda_graph = True
        self.parallel.tp_size = 2
        self.model.quant_config = object()
        self.cache.reuse_cache = True
        with patch.object(bi.torch.cuda, "get_device_capability", return_value=(10, 0)):
            with self.assertRaises(ValueError) as raised:
                configure_batch_invariant_model(self.engine, self.model)
        for reason in (
            "SM103",
            "single-node PDFUSION",
            "unquantized BF16",
            "eager",
            "prefix reuse",
        ):
            self.assertIn(reason, str(raised.exception))
        self.assertTrue(self.hw.enable_dynamic_decode_backend)
        self.assertTrue(self.model.enable_fp32_lm_head)

    def _set_hybrid(self, tp):
        self._set_qwen3_moe()
        model = self.model
        model.model_type = "qwen35_moe"
        model.ckpt_path = "/test/checkpoint"
        model.num_layers, model.hidden_size = (40, 2048) if tp == 1 else (60, 4096)
        model.expert_num, model.moe_k = (256, 8) if tp == 1 else (512, 10)
        model.inter_size = model.moe_inter_size = 512 if tp == 1 else 1024
        model.attn_config.head_num = 16 if tp == 1 else 32
        model.attn_config.kv_head_num = 2
        model.attn_config.size_per_head = 256
        model.mm_model_config = SimpleNamespace(is_multimodal=True)
        model.vit_separation = VitSeparation.VIT_SEPARATION_LOCAL
        model.linear_attention_config = SimpleNamespace(
            ssm_state_dtype=DataType.TYPE_FP32
        )
        self._set_tp_size(tp)

    def test_hybrid_activation_default_is_opt_in(self):
        for enabled, model_type, explicit, expected in (
            (True, "qwen35_moe", None, "bf16"),
            (True, "qwen35_moe", "fp16", "fp16"),
            (False, "qwen35_moe", None, None),
            (True, "qwen_3", None, None),
        ):
            with self.subTest(
                enabled=enabled, model_type=model_type, explicit=explicit
            ):
                config = PyEnvConfigs()
                config.py_hw_kernel_config.batch_invariant = enabled
                config.model_args.model_type = model_type
                config.model_args.act_type = explicit
                with patch("torch.cuda.is_available", return_value=False):
                    setup_default_args(config)
                self.assertEqual(config.model_args.act_type, expected)

    @patch.object(
        bi, "get_config_from_path", return_value={"text_config": {"dtype": "bfloat16"}}
    )
    def test_hybrid_precision_and_expert_layouts(self, read_config):
        for tp in (1, 4):
            for ep in (1, 4):
                for fp8 in (False, True):
                    with self.subTest(tp=tp, ep=ep, fp8=fp8):
                        self._set_hybrid(tp)
                        self.parallel.ep_size = ep
                        dp = ep if tp == 1 else 1
                        self.parallel.dp_size = dp
                        self.parallel.world_size = self.parallel.local_world_size = (
                            tp * dp
                        )
                        moe = self.engine.moe_config
                        moe.use_all_gather = dp == 1
                        moe.use_deepep_moe = dp > 1
                        self.hw.enable_cuda_graph = True
                        self.model.quant_config = (
                            Fp8BlockWiseQuantConfig() if fp8 else None
                        )
                        self.model.config_dtype = None
                        configure_batch_invariant_model(self.engine, self.model)
                        if dp > 1:
                            strategy = (
                                "fp8_per_block_pure_dp" if fp8 else "no_quant_pure_dp"
                            )
                        else:
                            strategy = "fp8_per_block_no_dp" if fp8 else "no_quant_cpp"
                        self.assertEqual(moe.moe_strategy, strategy)
                        self.assertTrue(moe.use_all_gather)
                        self.assertFalse(moe.use_deepep_moe)
                        self.assertTrue(self.hw.enable_cuda_graph)
                        self.assertFalse(self.model.mm_model_config.is_multimodal)
                        self.assertFalse(self.model.enable_fp32_lm_head)
        read_config.assert_called_with(self.model.ckpt_path)

    @patch.object(bi, "get_config_from_path")
    def test_hybrid_rejects_incompatible_checkpoint_state_or_layout(self, read_config):
        self._set_hybrid(4)
        read_config.return_value = {
            "dtype": "bfloat16",
            "text_config": {"dtype": "float16"},
        }
        self.model.quant_config = Fp8BlockWiseQuantConfig(group_size=64)
        self.model.linear_attention_config.ssm_state_dtype = DataType.TYPE_BF16
        self._set_tp_size(1)
        with self.assertRaises(ValueError) as error:
            configure_batch_invariant_model(self.engine, self.model)
        for reason in ("BF16 text checkpoint", "128x128", "FP32 SSM", "TP/DP/EP"):
            self.assertIn(reason, str(error.exception))
        self.assertEqual(self.engine.moe_config.moe_strategy, "auto")
        self.assertTrue(self.model.mm_model_config.is_multimodal)


if __name__ == "__main__":
    unittest.main()
