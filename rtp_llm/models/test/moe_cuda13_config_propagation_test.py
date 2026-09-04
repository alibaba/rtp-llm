import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch
import torch.nn as nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.deepseek_v4 import DeepSeekV4, _require_n_shared_experts
from rtp_llm.models_py.model_desc.deepseek_v4_model import (
    DeepSeekV4Model,
    _args_from_model_config,
    _resolve_dsv4_moe_strategy,
)
from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.chunk_env import (
    chunked_moe_enabled,
    moe_chunk_tokens_from_env,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    resolve_moe_max_tokens_per_rank,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


class Dsv4MoeConfigTest(unittest.TestCase):
    def test_global_zero_disables_chunking_without_shrinking_capacity(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CHUNK_TOKENS": "0",
                "DSV4_MOE_CHUNK_PREFILL": "1",
                "DSV4_MOE_CHUNK_TOKENS": "4096",
            },
            clear=True,
        ):
            self.assertFalse(chunked_moe_enabled())
            self.assertEqual(moe_chunk_tokens_from_env(), 0)
            self.assertEqual(
                resolve_moe_max_tokens_per_rank(
                    1048576,
                    65536,
                    4,
                    8,
                    chunking_enabled=chunked_moe_enabled(),
                    chunk_tokens=moe_chunk_tokens_from_env(),
                ),
                16384,
            )

    def test_global_chunk_capacity_takes_priority(self):
        with mock.patch.dict(
            os.environ,
            {
                "DSV4_CHUNK_TOKENS": "2048",
                "DSV4_MOE_CHUNK_TOKENS": "4096",
            },
            clear=True,
        ):
            self.assertEqual(moe_chunk_tokens_from_env(), 2048)
            self.assertEqual(
                resolve_moe_max_tokens_per_rank(
                    1048576,
                    65536,
                    1,
                    8,
                    chunking_enabled=chunked_moe_enabled(),
                    chunk_tokens=moe_chunk_tokens_from_env(),
                ),
                2048,
            )

    def test_block_decode_marks_the_actual_moe_forward_phase(self):
        class PassThroughHC(nn.Module):
            def pre(self, x, **_kwargs):
                return x, None, None

            def post(self, output, *_args):
                return output

        class DecodeAttention(nn.Module):
            def forward_decode(self, x, _metadata, kv_cache=None):
                return x

        class RecordingFfn(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_decode_forward = None

            def forward(self, x, _input_ids, *, is_decode_forward=False):
                self.is_decode_forward = is_decode_forward
                return x

        block = Block.__new__(Block)
        nn.Module.__init__(block)
        block.layer_id = 0
        block.attn_hc = PassThroughHC()
        block.ffn_hc = PassThroughHC()
        block.attn_norm = nn.Identity()
        block.ffn_norm = nn.Identity()
        block.attn = DecodeAttention()
        block.ffn = RecordingFfn()
        x = torch.ones(2, 1, 4)

        output = block.forward_decode(
            x,
            SimpleNamespace(),
            torch.arange(2).view(2, 1),
        )

        self.assertTrue(block.ffn.is_decode_forward)
        self.assertTrue(torch.equal(output, x))

    def test_debug_observer_preserves_tensor_names_and_position_filter(self):
        layer = Block.__new__(Block)
        nn.Module.__init__(layer)
        layer.layer_id = 3
        positions = torch.tensor([4, 9])
        records = []

        def record(level, name, tensor):
            records.append((level, name, tensor.clone()))

        with mock.patch.object(_rt, "should_record_layer", return_value=True):
            with mock.patch.object(_rt, "_DBG_GLOBAL_POS", 9):
                with mock.patch.object(_rt, "record_if_level", side_effect=record):
                    observer = layer._moe_observer(positions)
                    self.assertIsNotNone(observer)
                    for kind in (
                        "input",
                        "topk_weights",
                        "topk_indices",
                        "routed_y",
                        "shared_y",
                        "final_y",
                    ):
                        observer(kind, torch.arange(4).view(2, 2))

        names = [name for _, name, _ in records]
        for suffix in (
            "moe_x_in",
            "moe_topk_weights",
            "moe_topk_indices",
            "moe_routed_y",
            "moe_shared_y",
            "moe_y",
        ):
            self.assertIn(f"L03_{suffix}", names)
            self.assertIn(f"L03_{suffix}_pos9", names)
        position_records = [
            tensor for _, name, tensor in records if name.endswith("pos9")
        ]
        self.assertTrue(all(tensor.shape == (1, 2) for tensor in position_records))

    def test_dsv4_preserves_explicit_builtin_for_registry_validation(self):
        config = SimpleNamespace(moe_strategy="fp8_per_block_no_dp")
        self.assertEqual(
            _resolve_dsv4_moe_strategy(config),
            "fp8_per_block_no_dp",
        )

    def test_dsv4_preserves_external_strategy_name(self):
        config = SimpleNamespace(moe_strategy=" external_test_strategy ")
        self.assertEqual(
            _resolve_dsv4_moe_strategy(config),
            "external_test_strategy",
        )

    @staticmethod
    def _minimal_config_json(n_shared_experts: int) -> dict:
        return {
            "num_hidden_layers": 1,
            "hidden_size": 512,
            "vocab_size": 64,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "compress_ratios": [0],
            "o_groups": 1,
            "o_lora_rank": 0,
            "index_head_dim": 128,
            "index_n_heads": 1,
            "index_topk": 4,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "num_experts_per_tok": 4,
            "n_routed_experts": 16,
            "moe_intermediate_size": 256,
            "n_shared_experts": n_shared_experts,
        }

    @classmethod
    def _load_minimal_model_config(cls, n_shared_experts: int) -> ModelConfig:
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(cls._minimal_config_json(n_shared_experts), writer)
            return DeepSeekV4._create_config(ckpt_path)

    def test_missing_shared_expert_count_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "missing required field"):
            _require_n_shared_experts({})

    def test_scheduler_multi_request_capacity_reaches_unchunked_dsv4_moe(self):
        for configured_cap, expected_cap in ((0, 12288), (7000, 7000)):
            with self.subTest(configured_cap=configured_cap), patch.dict(
                os.environ, {"DSV4_CHUNK_TOKENS": "0"}, clear=True
            ):
                model_config = self._load_minimal_model_config(1)
                model_config.max_seq_len = 4096
                scheduler = SimpleNamespace(
                    max_context_batch_size=3,
                    max_batch_tokens_size=configured_cap,
                )
                engine_config = SimpleNamespace(
                    runtime_config=SimpleNamespace(
                        fifo_scheduler_config=scheduler, model_name=""
                    )
                )
                ModelFactory.update_engine_config_from_model_config(
                    engine_config, model_config
                )
                args = _args_from_model_config(model_config)
                capacity = resolve_moe_max_tokens_per_rank(
                    args.max_seq_len,
                    args.max_tokens_per_rank,
                    cp_size=1,
                    max_generate_batch_size=8,
                    chunking_enabled=chunked_moe_enabled(),
                    chunk_tokens=moe_chunk_tokens_from_env(),
                )
                self.assertEqual(capacity, expected_cap)
                self.assertGreater(capacity, args.max_seq_len)
                for executor_cls in (MegaMoeExecutor, MegaMoeSEExecutor):
                    executor = executor_cls.__new__(executor_cls)
                    nn.Module.__init__(executor)
                    executor._mega_buf = SimpleNamespace(
                        num_max_tokens_per_rank=capacity
                    )
                    executor._mega_y = torch.empty(capacity, 1)
                    executor._validate_capacity(expected_cap)
                    with self.assertRaisesRegex(RuntimeError, "exceeds"):
                        executor._validate_capacity(expected_cap + 1)

    def test_dsv4_prefill_capacity_falls_back_only_when_unspecified(self):
        model_config = self._load_minimal_model_config(1)
        for seq_len, prefill_cap, expected in (
            (4096, None, 4096),
            (0, None, 4096),
            (4096, 0, 0),
            (4096, 7000, 7000),
        ):
            with self.subTest(seq_len=seq_len, prefill_cap=prefill_cap):
                model_config.max_seq_len = seq_len
                model_config.moe_prefill_max_tokens_per_rank = prefill_cap
                self.assertEqual(
                    _args_from_model_config(model_config).max_tokens_per_rank,
                    expected,
                )

    def test_zero_prefill_budget_keeps_positive_runtime_capacity(self):
        for cp_size in (1, 2, 4):
            for chunking in (False, True):
                with self.subTest(cp_size=cp_size, chunking=chunking):
                    self.assertEqual(
                        resolve_moe_max_tokens_per_rank(
                            4096, 0, cp_size, 8, chunking_enabled=chunking
                        ),
                        1,
                    )
                    self.assertEqual(
                        resolve_moe_max_tokens_per_rank(
                            4096,
                            0,
                            cp_size,
                            8,
                            is_decode_role=True,
                            is_speculative=True,
                            gen_num_per_cycle=2,
                            chunking_enabled=chunking,
                        ),
                        24,
                    )

    def test_cp_multi_request_capacity_reaches_transformer_construction(self):
        class ConstructionReached(Exception):
            pass

        for cp_size, seq_len, budget, expected in (
            (2, 4096, 0, 6144),
            (2, 4097, 0, 6150),
            (4, 4097, 0, 3078),
            (2, 4096, 7000, 3504),
            (4, 1, 0, 6),
        ):
            with self.subTest(cp=cp_size, seq_len=seq_len, budget=budget), patch.dict(
                os.environ, {"DSV4_CHUNK_TOKENS": "0"}, clear=True
            ):
                config = self._load_minimal_model_config(1)
                config.max_seq_len = seq_len
                scheduler = SimpleNamespace(
                    max_context_batch_size=3, max_batch_tokens_size=budget
                )
                ModelFactory.update_engine_config_from_model_config(
                    SimpleNamespace(
                        runtime_config=SimpleNamespace(
                            fifo_scheduler_config=scheduler, model_name=""
                        )
                    ),
                    config,
                )
                model = DeepSeekV4Model(
                    config,
                    SimpleNamespace(
                        tp_size=cp_size,
                        prefill_cp_config=SimpleNamespace(is_enabled=lambda: True),
                    ),
                    SimpleNamespace(global_weights={}, weights=[]),
                    MoeConfig(),
                    max_generate_batch_size=8,
                )
                self.assertEqual(
                    model._v4_args.max_tokens_per_rank,
                    config.moe_prefill_max_tokens_per_rank,
                )
                # Stop only at weight materialization: constructor + runtime
                # resource propagation above use the production code paths.
                with patch(
                    "rtp_llm.models_py.model_desc.deepseek_v4_model.V4Transformer",
                    side_effect=ConstructionReached,
                ) as construct:
                    with self.assertRaises(ConstructionReached):
                        model._initialize_impl(
                            SimpleNamespace(
                                kv_cache=None,
                                is_speculative=False,
                                is_decode_role=False,
                                max_context_batch_size=3,
                            )
                        )
                self.assertEqual(
                    construct.call_args.args[0].max_tokens_per_rank, expected
                )
                for executor_cls in (MegaMoeExecutor, MegaMoeSEExecutor):
                    executor = executor_cls.__new__(executor_cls)
                    nn.Module.__init__(executor)
                    executor._mega_buf = SimpleNamespace(
                        num_max_tokens_per_rank=expected
                    )
                    executor._mega_y = torch.empty(expected, 1)
                    executor._validate_capacity(expected)
                    with self.assertRaisesRegex(RuntimeError, "exceeds"):
                        executor._validate_capacity(expected + 1)

    def test_explicit_zero_keeps_routed_only_checkpoint_supported(self):
        self.assertEqual(_require_n_shared_experts({"n_shared_experts": 0}), 0)

    def test_one_shared_expert_is_supported(self):
        self.assertEqual(_require_n_shared_experts({"n_shared_experts": 1}), 1)

    def test_multiple_shared_experts_are_supported(self):
        self.assertEqual(_require_n_shared_experts({"n_shared_experts": 2}), 2)

    def test_shared_expert_count_rejects_invalid_values(self):
        for value in (True, False, 0.5, 1.0, "1", None, -1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "non-negative integer"
            ):
                _require_n_shared_experts({"n_shared_experts": value})

    def test_routed_only_config_propagates_through_model_build_inputs(self):
        model_config = self._load_minimal_model_config(0)

        self.assertEqual(model_config.n_shared_experts, 0)
        self.assertEqual(model_config.inter_size, 0)
        args = _args_from_model_config(model_config)
        self.assertEqual(args.n_shared_experts, 0)
        self.assertEqual(args.moe_inter_dim, 256)

    def test_routed_only_config_ignores_dense_intermediate_width(self):
        model_config = self._load_minimal_model_config(0)
        model_config.inter_size = 1024

        args = _args_from_model_config(model_config)

        self.assertEqual(args.n_shared_experts, 0)
        self.assertEqual(args.moe_inter_dim, 256)

    def test_multiple_shared_experts_keep_routed_expert_width(self):
        model_config = self._load_minimal_model_config(2)

        self.assertEqual(model_config.n_shared_experts, 2)
        self.assertEqual(model_config.inter_size, 512)
        args = _args_from_model_config(model_config)
        self.assertEqual(args.n_shared_experts, 2)
        self.assertEqual(args.moe_inter_dim, 256)

    def test_eplb_physical_expert_count_reaches_v4_block_inputs(self):
        from rtp_llm.models_py.modules.dsv4.transformer import _block_kwargs
        from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
            Fp8Fp4MoeRuntimeConfig,
        )

        model_config = self._load_minimal_model_config(1)
        model_config.eplb_config.redundant_expert = 8

        args = _args_from_model_config(model_config)
        kwargs = _block_kwargs(0, args, layer_weights={})

        self.assertEqual(args.n_routed_experts, 16)
        self.assertEqual(args.n_physical_experts, 24)
        self.assertEqual(kwargs["n_physical_experts"], 24)
        # Propagation preserves the unsupported request so the production
        # factory rejects it, rather than silently dropping redundant experts.
        runtime = Fp8Fp4MoeRuntimeConfig(
            layer_id=0,
            hidden_size=args.dim,
            moe_inter_dim=args.moe_inter_dim,
            expert_num=args.n_routed_experts,
            physical_expert_num=kwargs["n_physical_experts"],
            moe_k=args.n_activated_experts,
            n_shared_experts=args.n_shared_experts,
            swiglu_limit=args.swiglu_limit,
            ep_size=8,
            ep_rank=0,
            max_tokens_per_rank=32,
            moe_strategy="auto",
        )
        with self.assertRaisesRegex(
            ValueError, "do not support EPLB.*logical_experts=16, physical_experts=24"
        ):
            FusedMoeFactory().create_fused_moe(runtime, {})

    def test_zero_routed_expert_width_fails_fast(self):
        model_config = self._load_minimal_model_config(0)
        model_config.moe_inter_size = 0

        with self.assertRaisesRegex(ValueError, "positive routed expert width"):
            _args_from_model_config(model_config)

    def test_inconsistent_shared_expert_width_fails_fast(self):
        model_config = self._load_minimal_model_config(2)
        model_config.inter_size = 256

        with self.assertRaisesRegex(ValueError, "shared-expert width is inconsistent"):
            _args_from_model_config(model_config)

    def test_shared_width_can_recover_missing_routed_width(self):
        model_config = self._load_minimal_model_config(2)
        model_config.moe_inter_size = 0

        self.assertEqual(_args_from_model_config(model_config).moe_inter_dim, 256)

    def test_production_weight_graph_tracks_shared_expert_config(self):
        import torch

        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig
        from rtp_llm.utils.database import CkptDatabase
        from rtp_llm.utils.model_weight import W

        for n_shared_experts in (0, 1, 2):
            with self.subTest(n_shared_experts=n_shared_experts):
                model_config = self._load_minimal_model_config(n_shared_experts)
                parallelism_config = ParallelismConfig()
                parallelism_config.tp_size = 1
                parallelism_config.tp_rank = 0
                parallelism_config.dp_size = 1
                parallelism_config.dp_rank = 0
                parallelism_config.ep_size = 1
                parallelism_config.ep_rank = 0
                parallelism_config.world_size = 1
                parallelism_config.world_rank = 0
                parallelism_config.local_world_size = 1

                with tempfile.TemporaryDirectory() as checkpoint_path:
                    # A real checkpoint database drives the same constructor,
                    # metadata processing, descriptor construction, and file
                    # filtering path used by BaseModel.create_model_loader().
                    torch.save(
                        {"embed.weight": torch.ones(1)},
                        os.path.join(checkpoint_path, "model.bin"),
                    )
                    database = CkptDatabase(checkpoint_path)
                    weight_builder = DeepSeekV4.get_weight_cls()(
                        model_config=model_config,
                        parallelism_config=parallelism_config,
                        hw_kernel_config=HWKernelConfig(),
                        kv_cache_config=KVCacheConfig(),
                    )
                    weight_info = weight_builder.create_model_weight_info(database)

                self.assertEqual(weight_builder._n_shared_experts, n_shared_experts)
                layer_by_name = {
                    weight.name: weight for weight in weight_info.layer_weights[0]
                }
                shared_names = {W.v4_shared_w13_w, W.v4_shared_w2_w}
                if n_shared_experts == 0:
                    self.assertTrue(shared_names.isdisjoint(layer_by_name))
                else:
                    self.assertTrue(shared_names.issubset(layer_by_name))
                    self.assertEqual(
                        {
                            ckpt.tensor_name(0)
                            for ckpt in layer_by_name[W.v4_shared_w13_w].weights
                        },
                        {
                            "layers.0.ffn.shared_experts.w1.weight",
                            "layers.0.ffn.shared_experts.w3.weight",
                        },
                    )
                    self.assertEqual(
                        {
                            ckpt.tensor_name(0)
                            for ckpt in layer_by_name[W.v4_shared_w2_w].weights
                        },
                        {"layers.0.ffn.shared_experts.w2.weight"},
                    )

    def test_public_strategy_is_forwarded_to_the_shared_registry(self):
        from rtp_llm.server.server_args import server_args

        with patch.dict(os.environ, {}, clear=True):
            public_config = server_args.setup_args(
                ["--moe_strategy", "fp8_per_block_no_dp"]
            ).moe_config
        args = _args_from_model_config(
            self._load_minimal_model_config(1),
            moe_config=public_config,
        )

        self.assertEqual(public_config.moe_strategy, "fp8_per_block_no_dp")
        self.assertEqual(args.moe_strategy, "fp8_per_block_no_dp")

    def test_public_strategy_reaches_v4_block_and_moe_selection(self):
        import torch
        import torch.nn as nn

        from rtp_llm.models_py.modules.dsv4.block import Block
        from rtp_llm.models_py.modules.dsv4.transformer import _block_kwargs
        from rtp_llm.server.server_args import server_args
        from rtp_llm.utils.model_weight import W

        class DummyModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class DummyMoe(DummyModule):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.strategy_name = kwargs["strategy"]
                self.received_kwargs = kwargs

        weight_keys = (
            W.v4_attn_norm,
            W.v4_ffn_norm,
            W.v4_hc_attn_fn,
            W.v4_hc_attn_base,
            W.v4_hc_attn_scale,
            W.v4_hc_ffn_fn,
            W.v4_hc_ffn_base,
            W.v4_hc_ffn_scale,
        )
        cases = (("mega_moe", 0), ("mega_moe_se", 1), ("mega_moe_se", 3))
        for strategy, n_shared_experts in cases:
            with (
                self.subTest(strategy=strategy),
                patch.dict(os.environ, {}, clear=True),
            ):
                public_config = server_args.setup_args(
                    ["--moe_strategy", strategy]
                ).moe_config
                model_config = self._load_minimal_model_config(n_shared_experts)
                args = _args_from_model_config(
                    model_config,
                    moe_config=public_config,
                )
                args.ep_size = 2
                args.ep_rank = 0
                layer_weights = {key: torch.ones(1) for key in weight_keys}
                experts = args.n_routed_experts // args.ep_size
                inter, dim = args.moe_inter_dim, args.dim
                for key, value in ((W.v4_routed_w1_w, 1), (W.v4_routed_w3_w, 3)):
                    layer_weights[key] = torch.full(
                        (experts, inter, dim // 2), value, dtype=torch.int8
                    )
                for key, value in ((W.v4_routed_w1_s, 11), (W.v4_routed_w3_s, 13)):
                    layer_weights[key] = torch.full(
                        (experts, inter, dim // 32), value, dtype=torch.uint8
                    )
                layer_weights.update(
                    {
                        W.v4_router_w: torch.zeros(args.n_routed_experts, dim),
                        W.v4_routed_w2_w: torch.zeros(
                            experts, dim, inter // 2, dtype=torch.int8
                        ),
                        W.v4_routed_w2_s: torch.ones(
                            experts, dim, inter // 32, dtype=torch.uint8
                        ),
                    }
                )
                if n_shared_experts:
                    for key in (
                        W.v4_shared_w13_w,
                        W.v4_shared_w13_s,
                        W.v4_shared_w2_w,
                        W.v4_shared_w2_s,
                    ):
                        layer_weights[key] = torch.ones(1)
                kwargs = _block_kwargs(0, args, layer_weights)

                with (
                    patch(
                        "rtp_llm.models_py.modules.dsv4.block.AttentionFP8",
                        DummyModule,
                    ),
                    patch(
                        "rtp_llm.models_py.modules.dsv4.block.RMSNorm",
                        DummyModule,
                    ),
                    patch(
                        "rtp_llm.models_py.modules.dsv4.block.build_hc_unit",
                        side_effect=lambda *args, **kwargs: DummyModule(),
                    ),
                    patch.object(
                        Block,
                        "_resolve_prefill_fast_hc_impls",
                        return_value=(lambda: None,) * 4,
                    ),
                    patch(
                        "rtp_llm.models_py.modules.dsv4.block.ChunkedFp8Fp4MoeLayer",
                        DummyMoe,
                    ),
                ):
                    block = Block(**kwargs)

                self.assertEqual(public_config.moe_strategy, strategy)
                self.assertEqual(args.moe_strategy, strategy)
                self.assertEqual(kwargs["moe_strategy"], strategy)
                self.assertEqual(block.ffn.strategy_name, strategy)
                received = block.ffn.received_kwargs
                self.assertEqual(received["moe_w1_layout"], "gate_up")
                self.assertEqual(received["model_type"], "deepseek_v4")
                self.assertTrue(received["chunking_enabled"])
                self.assertEqual(received["observer_factory"], block._moe_observer)
                self.assertTrue(torch.all(layer_weights[W.moe_w1][:, :inter] == 1))
                self.assertTrue(torch.all(layer_weights[W.moe_w1][:, inter:] == 3))
                self.assertNotIn(W.v4_routed_w1_w, layer_weights)
                self.assertEqual(W.ffn_w13 in layer_weights, n_shared_experts > 0)


class MoeCuda13ConfigPropagationTest(unittest.TestCase):
    def test_real_adapter_constructs_mega_moe_executors(self):
        model_config = ModelConfig()
        model_config.expert_num = 64
        model_config.moe_k = 6
        model_config.moe_inter_size = 1408
        model_config.n_shared_experts = 2
        model_config.inter_size = 2816
        model_config.routed_scaling_factor = 2.5
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 2
        parallelism_config.world_size = 2
        adapter = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )
        quant_config = FusedMoEQuantConfig(quant_dtype="fp8_fp4", block_shape=[128, 32])

        for executor_class in (MegaMoeExecutor, MegaMoeSEExecutor):
            with self.subTest(executor=executor_class.__name__), mock.patch.object(
                executor_class, "setup_weights", autospec=True
            ) as setup_weights:
                executor = executor_class(adapter, quant_config, {})
                setup_weights.assert_called_once()
                self.assertIs(executor.cfg, adapter)
                self.assertEqual(executor.cfg.route_scale, 2.5)


if __name__ == "__main__":
    unittest.main()
