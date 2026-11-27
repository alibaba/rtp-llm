import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.model_factory import configure_batch_invariant_model
from rtp_llm.models_py.determinism import batch_invariant_config as bi
from rtp_llm.ops import DataType, KvCacheDataType, RoleType, SpeculativeType, TaskType


class BatchInvariantStartupTest(unittest.TestCase):
    def test_older_native_config(self):
        from rtp_llm.config.engine_config import EngineConfig
        from rtp_llm.ops import HWKernelConfig
        from rtp_llm.server.server_args.server_args import setup_args

        # Older pybind objects allow neither reading nor assigning the new field.
        with patch.object(HWKernelConfig, "batch_invariant", property(), create=True):
            for env in ({}, {"RTP_BATCH_INVARIANT": "0"}):
                with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                    configs = setup_args([])
                    configs.parallelism_config.tp_size = 2
                    policy = EngineConfig.create(configs).collective_config
                    self.assertFalse(policy.batch_invariant)

            with patch.dict(os.environ, {"RTP_BATCH_INVARIANT": "1"}, clear=True):
                with self.assertRaisesRegex(
                    RuntimeError, "RTP_BATCH_INVARIANT=1.*native library"
                ):
                    setup_args([])


class BatchInvariantConfigTest(unittest.TestCase):
    def setUp(self):
        self.hw = SimpleNamespace(
            batch_invariant=True,
            enable_cuda_graph=False,
            generation_prefill_capture_token_buckets=[],
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

    def test_decode_cuda_graph_for_supported_models_and_tp_sizes(self):
        self.hw.enable_cuda_graph = True
        for model_type in ("qwen_3", "qwen_3_moe"):
            if model_type == "qwen_3_moe":
                self._set_qwen3_moe()
            for size in (1, 2, 4):
                with self.subTest(model_type=model_type, tp_size=size):
                    self._set_tp_size(size)
                    configure_batch_invariant_model(self.engine, self.model)
                    self.assertTrue(self.hw.enable_cuda_graph)

    def test_prefill_capture_buckets_require_cuda_graph_to_be_disabled(self):
        self.hw.generation_prefill_capture_token_buckets = [64]
        configure_batch_invariant_model(self.engine, self.model)
        self.hw.enable_cuda_graph = True
        with self.assertRaisesRegex(ValueError, "GENERATION_PREFILL_CAPTURE_CONFIG"):
            configure_batch_invariant_model(self.engine, self.model)

    def test_decode_graph_rejects_speculation_and_batch_decode_scheduler(self):
        self.hw.enable_cuda_graph = True
        with patch.object(self.engine.sp_config, "type", SpeculativeType.MTP):
            with self.assertRaisesRegex(ValueError, "speculative decoding"):
                configure_batch_invariant_model(self.engine, self.model)
        self.engine.runtime_config.use_batch_decode_scheduler = True
        with self.assertRaisesRegex(ValueError, "USE_BATCH_DECODE_SCHEDULER"):
            configure_batch_invariant_model(self.engine, self.model)

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

    def test_headwise_checkpoint_is_rejected(self):
        self.model.headwise_config = {"sliding_window": 128}
        with self.assertRaisesRegex(ValueError, "headwise_config"):
            configure_batch_invariant_model(self.engine, self.model)

    def test_conflicts_are_reported_before_applying_overrides(self):
        self.hw.enable_cuda_graph = True
        self.hw.generation_prefill_capture_token_buckets = [64]
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
            "GENERATION_PREFILL_CAPTURE_CONFIG",
            "prefix reuse",
        ):
            self.assertIn(reason, str(raised.exception))
        self.assertTrue(self.hw.enable_dynamic_decode_backend)
        self.assertTrue(self.model.enable_fp32_lm_head)


if __name__ == "__main__":
    unittest.main()
