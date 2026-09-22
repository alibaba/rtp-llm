import unittest
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.engine_config import finalize_scheduler_config
from rtp_llm.device.device_type import DeviceType
from rtp_llm.ops import RoleType


class DummyFIFOSchedulerConfig:
    def __init__(self):
        self.max_context_batch_size = 2
        self.max_batch_tokens_size = 0
        self.prefill_chunk_size = 0


class EngineConfigTest(TestCase):
    def _finalize(self, chunk_size=0, **overrides):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = chunk_size
        args = {
            "max_seq_len": 1024,
            "use_hybrid_attention": False,
            "role_type": RoleType.PREFILL,
            "use_batch_decode_scheduler": False,
            "seq_size_per_block": 64,
        }
        args.update(overrides)
        finalize_scheduler_config(cfg, **args)
        return cfg

    def test_finalize_scheduler_config_disabled_by_default(self):
        # prefill_chunk_size <= 0 => chunked prefill disabled, no validation runs.
        cfg = self._finalize(
            use_hybrid_attention=True,  # would raise if chunked prefill were enabled
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_rejects_chunk_size_smaller_than_one_block(self):
        with self.assertRaises(ValueError):
            self._finalize(chunk_size=17)

    def test_finalize_scheduler_config_floor_aligns_chunk_size(self):
        requested_chunk_size = 130
        cfg = self._finalize(chunk_size=requested_chunk_size)

        self.assertEqual(cfg.prefill_chunk_size, 128)
        self.assertLessEqual(cfg.prefill_chunk_size, requested_chunk_size)

    def test_finalize_scheduler_config_allows_int_max_chunk_size(self):
        cfg = self._finalize(
            chunk_size=2**31 - 1,
            seq_size_per_block=1,
        )

        self.assertEqual(cfg.prefill_chunk_size, 2**31 - 1)

    def test_finalize_scheduler_config_rejects_chunk_size_above_int_max(self):
        with self.assertRaises(ValueError):
            self._finalize(
                chunk_size=2**31,
                seq_size_per_block=1,
            )

    def test_finalize_scheduler_config_allows_hybrid_attention(self):
        cfg = self._finalize(chunk_size=64, use_hybrid_attention=True)
        self.assertEqual(cfg.prefill_chunk_size, 64)

    def test_finalize_scheduler_config_disables_chunked_prefill_for_unsupported_role(
        self,
    ):
        # Roles other than PREFILL / PDFUSION never activate chunked prefill in C++; config
        # finalization should not reject their model combination just because the shared env
        # var is present, and it should silently zero prefill_chunk_size so downstream sees a
        # disabled config.
        cfg = self._finalize(
            chunk_size=64,
            use_hybrid_attention=True,
            role_type=RoleType.DECODE,
            use_batch_decode_scheduler=True,
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_rejects_batch_decode_scheduler(self):
        with self.assertRaisesRegex(ValueError, "use_batch_decode_scheduler=True"):
            self._finalize(
                chunk_size=64,
                use_batch_decode_scheduler=True,
            )

    def test_finalize_scheduler_config_allows_supported_roles(self):
        # Both roles execute prefill locally and share the same chunked-prefill gate.
        for role_type in (RoleType.PREFILL, RoleType.PDFUSION):
            with self.subTest(role_type=role_type):
                cfg = self._finalize(
                    chunk_size=64,
                    role_type=role_type,
                )
                self.assertEqual(cfg.prefill_chunk_size, 64)


class MlaChunkedConfigTest(TestCase):
    def setUp(self):
        from rtp_llm.model_factory import ModelFactory

        self.update = ModelFactory.update_engine_config_from_model_config
        self.engine, self.model = self._make_configs()
        cuda = patch(
            "rtp_llm.device.device_type.get_device_type", return_value=DeviceType.Cuda
        )
        cuda.start()
        self.addCleanup(cuda.stop)

    @staticmethod
    def _make_configs():
        from rtp_llm.config.engine_config import EngineConfig
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.config.py_config_modules import PyEnvConfigs

        engine = EngineConfig.create(PyEnvConfigs())
        engine.pd_sep_config.role_type = RoleType.PDFUSION
        engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = 64
        engine.hw_kernel_config.enable_cuda_graph = False
        engine.kv_cache_config.reuse_cache = False
        model = ModelConfig()
        model.max_seq_len = 2048
        model.data_type = "bf16"
        model.attn_config.use_mla = True
        model.attn_config.head_num = 16
        model.attn_config.kv_head_num = 16
        model.attn_config.tokens_per_block = 64
        model.attn_config.kernel_tokens_per_block = 64
        return engine, model

    def test_allows_supported_mla_configurations(self):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.ops import SpeculativeType

        # These are config checks; speculative model compatibility remains
        # the responsibility of the native loading/execution path.
        cases = (
            (1, False, False, False, RoleType.PREFILL, SpeculativeType.NONE),
            (2, False, True, False, RoleType.PDFUSION, SpeculativeType.MTP),
            (16, True, True, True, RoleType.PDFUSION, SpeculativeType.MTP),
            (1, False, False, False, RoleType.PDFUSION, SpeculativeType.EAGLE),
            (1, False, False, False, RoleType.PDFUSION, SpeculativeType.DSPARK),
        )
        for tp, fp8, reuse, graph, role, sp_type in cases:
            with self.subTest(
                tp=tp, fp8=fp8, reuse=reuse, graph=graph, role=role, sp_type=sp_type
            ):
                engine, model = self._make_configs()
                engine.parallelism_config.tp_size = tp
                engine.parallelism_config.world_size = tp
                engine.pd_sep_config.role_type = role
                engine.kv_cache_config.reuse_cache = reuse
                engine.hw_kernel_config.enable_cuda_graph = graph
                engine.sp_config.type = sp_type
                quant = Fp8BlockWiseQuantConfig() if fp8 else None
                model.quant_config = quant
                self.update(engine, model)
                self.assertEqual(
                    engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 64
                )
                self.assertEqual(engine.pd_sep_config.role_type, role)
                self.assertEqual(engine.kv_cache_config.reuse_cache, reuse)
                self.assertEqual(engine.hw_kernel_config.enable_cuda_graph, graph)
                self.assertEqual(engine.sp_config.type, sp_type)
                self.assertIs(model.quant_config, quant)

    def test_fp8_absorb_override_only_for_active_mla_chunking(self):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig

        for role, budget, mla, fp8, disable_absorb in (
            (RoleType.PDFUSION, 64, True, True, True),
            (RoleType.PDFUSION, 0, True, True, False),
            (RoleType.DECODE, 64, True, True, False),
            (RoleType.PDFUSION, 64, True, False, False),
            (RoleType.PDFUSION, 64, False, True, False),
        ):
            with self.subTest(role=role, budget=budget, mla=mla, fp8=fp8):
                engine, model = self._make_configs()
                engine.pd_sep_config.role_type = role
                engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = budget
                engine.fmha_config.absorb_opt_len = 4096
                model.attn_config.use_mla = mla
                model.quant_config = Fp8BlockWiseQuantConfig() if fp8 else None
                self.update(engine, model)
                self.assertEqual(
                    engine.fmha_config.absorb_opt_len, 0 if disable_absorb else 4096
                )

    def test_mla_backend_options_follow_native_validation(self):
        from rtp_llm.ops import CPRotateMethod, KvCacheDataType

        # Admission is not an assertion that every native backend supports these
        # options. Chunking must not add a separate backend whitelist.
        for sparse, kv_dtype, cp in (
            (False, KvCacheDataType.FP8, CPRotateMethod.DISABLED),
            (True, KvCacheDataType.BASE, CPRotateMethod.DISABLED),
            (True, KvCacheDataType.FP8, CPRotateMethod.DISABLED),
            (True, KvCacheDataType.FP8, CPRotateMethod.PREFILL_CP),
            (False, KvCacheDataType.BASE, CPRotateMethod.PREFILL_CP),
        ):
            with self.subTest(sparse=sparse, kv_dtype=kv_dtype, cp=cp):
                engine, model = self._make_configs()
                model.attn_config.is_sparse = sparse
                model.attn_config.kv_cache_dtype = kv_dtype
                engine.parallelism_config.prefill_cp_config.method = cp
                self.update(engine, model)
                self.assertEqual(
                    engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 64
                )

    def test_mla_compute_dtype_follows_native_validation(self):
        import torch

        # The native backend decides whether FP16 execution is supported.
        engine, model = self._make_configs()
        model.data_type = "fp16"
        self.update(engine, model)
        self.assertEqual(model.compute_dtype, torch.float16)
        self.assertEqual(
            engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 64
        )

    def test_mla_config_does_not_restrict_device_type(self):
        # Configuration admission does not imply that a device has an MLA backend.
        for device in (DeviceType.ROCm, DeviceType.Cpu):
            with self.subTest(device=device), patch(
                "rtp_llm.device.device_type.get_device_type", return_value=device
            ):
                engine, model = self._make_configs()
                self.update(engine, model)
                self.assertEqual(
                    engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 64
                )

    def test_inactive_chunking_does_not_restrict_mla(self):
        for role, budget, dtype in (
            (RoleType.PDFUSION, 0, "fp16"),
            (RoleType.DECODE, 64, "bf16"),
        ):
            with self.subTest(role=role, budget=budget):
                engine, model = self._make_configs()
                engine.pd_sep_config.role_type = role
                engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = budget
                model.attn_config.is_sparse = True
                model.data_type = dtype
                self.update(engine, model)
                self.assertEqual(
                    engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 0
                )


class HybridChunkedConfigTest(TestCase):
    setUp = MlaChunkedConfigTest.setUp

    @staticmethod
    def _make_configs():
        engine, model = MlaChunkedConfigTest._make_configs()
        model.attn_config.use_mla = False
        model.model_type = "custom_hybrid"
        model.hybrid_attention_config.enable_hybrid_attention = True
        return engine, model

    def test_config_admission(self):
        from rtp_llm.ops import CPRotateMethod, HybridAttentionType, SpeculativeType

        # Model names/TP counts do not gate admission. Non-64-aligned blocks
        # only exercise reuse=False: admitting them does not prove cache safety.
        none, mtp, eagle = (
            SpeculativeType.NONE,
            SpeculativeType.MTP,
            SpeculativeType.EAGLE,
        )
        cases = (
            # model type, TP, logical block, kernel block, reuse, speculative, chunk
            ("qwen3_next", 1, 64, 64, False, none, 192),
            ("qwen35_dense", 2, 64, 64, True, none, 192),
            ("qwen35_moe", 8, 128, 64, True, none, 192),
            ("custom_hybrid", 16, 128, 64, False, none, 192),
            ("custom_hybrid", 1, 32, 32, False, none, 192),
            ("custom_hybrid", 1, 96, 32, False, none, 192),
            ("custom_hybrid", 1, 64, 64, False, mtp, 192),
            ("qwen35_moe", 2, 64, 64, True, eagle, 192),
            ("custom_hybrid", 1, 64, 64, False, mtp, 0),
        )
        for kind, tp, block, kernel, reuse, speculative, chunk in cases:
            with self.subTest(
                config=(kind, tp, block, kernel, reuse, speculative), chunk=chunk
            ):
                engine, model = self._make_configs()
                model.model_type = kind
                model.hybrid_attention_config.hybrid_attention_types = [
                    HybridAttentionType.LINEAR,
                    HybridAttentionType.NONE,
                ]
                model.attn_config.tokens_per_block = block
                model.attn_config.kernel_tokens_per_block = kernel
                engine.parallelism_config.tp_size = tp
                engine.parallelism_config.world_size = tp
                if tp == 2:
                    # The existing GDN CP implementation also accepts a prefix.
                    engine.parallelism_config.prefill_cp_config.method = (
                        CPRotateMethod.PREFILL_CP
                    )
                engine.kv_cache_config.reuse_cache = reuse
                engine.sp_config.type = speculative
                engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = chunk
                self.update(engine, model)
                self.assertEqual(
                    engine.runtime_config.fifo_scheduler_config.prefill_chunk_size,
                    chunk // block * block,
                )
                self.assertEqual(engine.kv_cache_config.reuse_cache, reuse)
                self.assertEqual(model.attn_config.kernel_tokens_per_block, kernel)
                self.assertEqual(engine.sp_config.type, speculative)

    def test_model_creation_checks_linear_implementations(self):
        import torch

        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearKDA as KDA
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet as GDN,
        )
        from rtp_llm.ops import HybridAttentionType

        class UnregisteredGDN(GDN):
            pass

        def make_layers(*attention_classes):
            # Real attention types without weights; only the layer layout is needed.
            layers = []
            for cls in attention_classes:
                layer = SimpleNamespace()
                if cls is not None:
                    layer.self_attn = cls.__new__(cls)
                    torch.nn.Module.__init__(layer.self_attn)
                layers.append(layer)
            return SimpleNamespace(layers=layers)

        gdn = make_layers(GDN)
        mixed = make_layers(GDN, KDA)
        subclass = make_layers(UnregisteredGDN)
        missing = make_layers(None)
        linear, full = HybridAttentionType.LINEAR, HybridAttentionType.NONE
        cases = (
            # chunk, case, constructed model, layer types, expected error
            (64, "gdn", make_layers(GDN, torch.nn.Identity), [linear, full], None),
            (64, "mixed", mixed, [linear, linear], "layer 1.*KimiLinearKDA"),
            (64, "subclass", subclass, [linear], "layer 0.*UnregisteredGDN"),
            (64, "missing_attention", missing, [linear], "layer 0.*missing self_attn"),
            (64, "missing_layer", gdn, [linear, linear], "cannot match"),
            (64, "missing_model", None, [linear], "cannot match"),
            (64, "missing_layer_types", gdn, [], "cannot match"),
            (64, "ordinary_attention", None, None, None),
            (0, "chunk_disabled", None, [linear], None),
        )
        for chunk, name, py_model, layer_types, message in cases:
            with self.subTest(case=name, chunk=chunk):
                engine, model = self._make_configs()
                model.num_layers = max(1, len(layer_types or []))
                model.hybrid_attention_config.enable_hybrid_attention = (
                    layer_types is not None
                )
                model.hybrid_attention_config.hybrid_attention_types = layer_types or []
                engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = chunk
                self.update(engine, model)
                loaded_model = SimpleNamespace(py_model=py_model)
                # Stub loading only; call the real factory to check integration.
                model_class = SimpleNamespace(
                    __name__="TestHybridModel",
                    from_config=lambda **kwargs: loaded_model,
                )
                with patch.object(
                    ModelFactory, "get_model_cls", return_value=model_class
                ):
                    if message:
                        with self.assertRaisesRegex(ValueError, message):
                            ModelFactory._create_model(model, engine)
                    else:
                        self.assertIs(
                            ModelFactory._create_model(model, engine), loaded_model
                        )


if __name__ == "__main__":
    unittest.main()
