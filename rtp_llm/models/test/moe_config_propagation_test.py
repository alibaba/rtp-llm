import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.moe_capacity import include_system_prompt_capacity
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.ffn_weight import FfnWeight, MoeWeight
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorSource
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight
from rtp_llm.models.kimi_k25.kimi_k25 import KimiK25
from rtp_llm.models.kimi_k25.kimi_k25_weight import KimiK25Weight
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Moe
from rtp_llm.models.qwen_v2_moe import Qwen2Moe
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import (
    CPRotateMethod,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    RoleType,
    SpeculativeExecutionConfig,
    SpeculativeType,
    TaskType,
)
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


def _deepseek_text_config(n_shared_experts: int = 2) -> dict:
    return {
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 4,
        "vocab_size": 32000,
        "hidden_size": 4096,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "routed_scaling_factor": 2.5,
        "num_experts_per_tok": 6,
        "n_routed_experts": 64,
        "moe_intermediate_size": 1408,
        "n_shared_experts": n_shared_experts,
        "first_k_dense_replace": 1,
    }


def _assert_moe_config_is_constructible(
    test: unittest.TestCase, config, expected_n_shared_experts: int = 2
) -> None:
    adapter = MoEConfigAdapter(
        model_config=config,
        parallelism_config=ParallelismConfig(),
        moe_config=MoeConfig(),
    )
    test.assertEqual(adapter.moe_inter_dim, 1408)
    test.assertEqual(adapter.n_shared_experts, expected_n_shared_experts)
    test.assertEqual(adapter.route_scale, 2.5)


class _RoutedOnlyTensorSource(TensorSource):
    def __init__(self):
        self.requested_names = []

    def load_tensor(self, name, data_type=torch.float16):
        if "shared_experts" in name:
            raise AssertionError(f"unexpected shared-expert lookup: {name}")
        self.requested_names.append(name)
        return [torch.ones((2, 2), dtype=data_type)]

    def has_tensor(self, name):
        return False

    def get_database(self):
        return None


class MoeConfigPropagationTest(unittest.TestCase):
    def _prefill_capacity_configs(self, configured_cap=0, chunk_size=0):
        model_config = ModelConfig()
        model_config.max_seq_len = 4096
        model_config.model_name = "test-model"
        model_config.task_type = TaskType.LANGUAGE_MODEL
        model_config.attn_config.tokens_per_block = 64
        engine_config = SimpleNamespace(
            runtime_config=SimpleNamespace(
                fifo_scheduler_config=SimpleNamespace(
                    max_context_batch_size=3,
                    max_batch_tokens_size=configured_cap,
                    prefill_chunk_size=chunk_size,
                ),
                model_name="",
                use_batch_decode_scheduler=False,
                warm_up=False,
                warm_up_with_loss=False,
                max_generate_batch_size=16,
            ),
            pd_sep_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=ParallelismConfig(),
            kv_cache_config=KVCacheConfig(),
            hw_kernel_config=HWKernelConfig(),
            sp_config=SpeculativeExecutionConfig(),
        )
        return engine_config, model_config

    def test_scheduler_prefill_batch_capacity_reaches_generic_moe(self):
        for configured_cap, expected_cap in ((0, 12288), (7000, 7000)):
            with self.subTest(configured_cap=configured_cap):
                engine_config, model_config = self._prefill_capacity_configs(
                    configured_cap=configured_cap
                )

                ModelFactory.update_engine_config_from_model_config(
                    engine_config, model_config
                )
                adapter = MoEConfigAdapter(
                    model_config=model_config,
                    parallelism_config=ParallelismConfig(),
                    moe_config=MoeConfig(),
                )

                self.assertEqual(
                    model_config.moe_prefill_max_tokens_per_rank, expected_cap
                )
                self.assertEqual(adapter.prefill_max_tokens_per_rank, expected_cap)
                self.assertEqual(adapter.max_tokens_per_rank, expected_cap)

    def test_chunked_prefill_capacity_reaches_generic_moe(self):
        for role, chunk, batch_cap, expected_chunk, expected_cap in (
            (RoleType.PREFILL, 130, 0, 128, 128),
            (RoleType.PDFUSION, 130, 0, 128, 128),
            (RoleType.DECODE, 130, 0, 0, 12288),
            # The singleton admission path bypasses the batch cap.
            (RoleType.PREFILL, 8192, 7000, 8192, 8192),
        ):
            with self.subTest(role=role, chunk=chunk, batch_cap=batch_cap):
                engine, model = self._prefill_capacity_configs(batch_cap, chunk)
                engine.pd_sep_config.role_type = role
                ModelFactory.update_engine_config_from_model_config(engine, model)
                moe_config = MoeConfig()
                moe_config.ll_num_max_token = 256
                adapter = MoEConfigAdapter(model, engine.parallelism_config, moe_config)

                scheduler = engine.runtime_config.fifo_scheduler_config
                self.assertEqual(scheduler.prefill_chunk_size, expected_chunk)
                self.assertEqual(scheduler.max_batch_tokens_size, batch_cap or 12288)
                self.assertEqual(model.max_seq_len, 4096)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, expected_cap)
                self.assertEqual(adapter.prefill_max_tokens_per_rank, expected_cap)
                self.assertEqual(adapter.max_tokens_per_rank, max(256, expected_cap))

    def test_page_rr_chunk_capacity_with_standard_warmup(self):
        for method in (CPRotateMethod.DISABLED, CPRotateMethod.PREFILL_CP):
            with self.subTest(method=method):
                engine, model = self._prefill_capacity_configs(1048576, 65536)
                model.max_seq_len = 1048576
                scheduler = engine.runtime_config.fifo_scheduler_config
                scheduler.max_context_batch_size = 1
                engine.runtime_config.warm_up = True
                engine.parallelism_config.tp_size = 8
                engine.parallelism_config.enable_sp = True
                cp_config = engine.parallelism_config.prefill_cp_config
                cp_config.method = method
                cp_config.kv_cache_sharded = True
                cp_config.prefill_cp_size = 8

                ModelFactory.update_engine_config_from_model_config(engine, model)

                self.assertEqual(model.moe_prefill_max_tokens_per_rank, 65536)
                self.assertEqual(model.max_seq_len, 1048576)
                self.assertEqual(scheduler.max_batch_tokens_size, 1048576)

    def test_page_rr_capacity_includes_sp_tail_padding(self):
        engine, model = self._prefill_capacity_configs(chunk_size=130)
        model.attn_config.tokens_per_block = 1
        engine.parallelism_config.tp_size = 8
        engine.parallelism_config.enable_sp = True
        engine.parallelism_config.prefill_cp_config.kv_cache_sharded = True
        ModelFactory.update_engine_config_from_model_config(engine, model)
        self.assertEqual(model.moe_prefill_max_tokens_per_rank, 136)
        self.assertEqual(
            engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 130
        )

    def test_query_cp_capacity_covers_ragged_rows_and_return_sequences(self):
        # Enumerate packed chunk layouts independently of the capacity formula.
        # A request can contribute multiple rows through num_return_sequences.
        def layouts(tokens):
            if tokens == 0:
                yield ()
            for first in range(1, tokens + 1):
                for tail in layouts(tokens - first):
                    yield (first, *tail)

        for method in (
            CPRotateMethod.ALL_GATHER,
            CPRotateMethod.ALL_GATHER_WITH_OVERLAP,
            CPRotateMethod.ALLTOALL,
        ):
            for tp in (2, 4, 8):
                with self.subTest(method=method, tp=tp):
                    engine, model = self._prefill_capacity_configs(chunk_size=8)
                    model.attn_config.tokens_per_block = 1
                    engine.parallelism_config.tp_size = tp
                    engine.parallelism_config.prefill_cp_config.method = method
                    engine.runtime_config.fifo_scheduler_config.max_context_batch_size = (
                        1
                    )
                    ModelFactory.update_engine_config_from_model_config(engine, model)
                    alignment = 2 * tp
                    physical_sizes = [
                        sum(
                            ((length + alignment - 1) // alignment) * alignment
                            for length in lengths
                        )
                        for tokens in range(1, 9)
                        for lengths in layouts(tokens)
                    ]
                    self.assertEqual(
                        model.moe_prefill_max_tokens_per_rank, max(physical_sizes)
                    )

    def test_non_language_tasks_keep_non_chunk_capacity(self):
        for task in (TaskType.DENSE_EMBEDDING, TaskType.RERANKER):
            with self.subTest(task=task):
                engine, model = self._prefill_capacity_configs(chunk_size=128)
                model.task_type = task
                ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, 12288)

    def test_loss_warmup_uses_actual_rows_and_speculative_reserve(self):
        for kind, gamma, expected in (
            (SpeculativeType.NONE, 0, 12285),
            (SpeculativeType.MTP, 3, 12276),
            (SpeculativeType.DSPARK, 3, 12261),
            (SpeculativeType.DSPARK, 0, 12285),
        ):
            with self.subTest(kind=kind, gamma=gamma):
                engine, model = self._prefill_capacity_configs(7000, 128)
                engine.runtime_config.warm_up = True
                engine.runtime_config.warm_up_with_loss = True
                engine.sp_config.type = kind
                engine.sp_config.gen_num_per_cycle = gamma
                with patch("rtp_llm.config.moe_capacity.is_cuda", return_value=True):
                    ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, expected)

    def test_loss_warmup_padding_is_per_sequence_for_cp_and_per_batch_for_sp(self):
        for query_cp, expected in ((False, 12280), (True, 12288)):
            with self.subTest(query_cp=query_cp):
                engine, model = self._prefill_capacity_configs(chunk_size=128)
                engine.parallelism_config.tp_size = 8
                engine.parallelism_config.enable_sp = True
                if query_cp:
                    engine.parallelism_config.prefill_cp_config.method = (
                        CPRotateMethod.ALL_GATHER
                    )
                engine.runtime_config.warm_up = True
                engine.runtime_config.warm_up_with_loss = True
                engine.sp_config.type = SpeculativeType.MTP
                engine.sp_config.gen_num_per_cycle = 3
                with patch("rtp_llm.config.moe_capacity.is_cuda", return_value=True):
                    ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, expected)

    def test_skipped_loss_warmup_does_not_expand_capacity(self):
        for skip in ("disabled", "multimodal", "ffn_disaggregate", "non_cuda"):
            with self.subTest(skip=skip):
                engine, model = self._prefill_capacity_configs(chunk_size=128)
                engine.runtime_config.warm_up = skip != "disabled"
                engine.runtime_config.warm_up_with_loss = True
                model.mm_model_config.is_multimodal = skip == "multimodal"
                engine.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = (
                    skip == "ffn_disaggregate"
                )
                with patch(
                    "rtp_llm.config.moe_capacity.is_cuda",
                    return_value=skip != "non_cuda",
                ):
                    ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, 128)

    def test_decode_graph_does_not_force_max_seq_capacity(self):
        for enabled, buckets, expected in (
            (False, [512], 128),
            (True, [], 128),
            (True, [32, 512], 512),
        ):
            with self.subTest(enabled=enabled, buckets=buckets):
                engine, model = self._prefill_capacity_configs(chunk_size=128)
                engine.hw_kernel_config.enable_cuda_graph = enabled
                engine.hw_kernel_config.decode_capture_batch_sizes = buckets
                ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, expected)

    def test_mtp_graph_capacity_pads_complete_requests(self):
        engine, model = self._prefill_capacity_configs(chunk_size=64)
        engine.hw_kernel_config.enable_cuda_graph = True
        engine.hw_kernel_config.decode_capture_batch_sizes = [17]
        engine.parallelism_config.tp_size = 8
        engine.parallelism_config.enable_sp = True
        engine.sp_config.type = SpeculativeType.MTP
        engine.sp_config.gen_num_per_cycle = 4
        ModelFactory.update_engine_config_from_model_config(engine, model)
        # 17 requests pad to 24 complete rows of width 5, rather than 88 tokens.
        self.assertEqual(model.moe_prefill_max_tokens_per_rank, 120)

    def test_query_cp_decode_graph_accounts_for_replicated_rows(self):
        engine, model = self._prefill_capacity_configs(chunk_size=64)
        engine.hw_kernel_config.enable_cuda_graph = True
        engine.hw_kernel_config.decode_capture_batch_sizes = [256]
        engine.parallelism_config.tp_size = 4
        engine.parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        ModelFactory.update_engine_config_from_model_config(engine, model)
        self.assertEqual(model.moe_prefill_max_tokens_per_rank, 1024)

    def test_dspark_proposal_graph_padding_can_exceed_verify_padding(self):
        engine, model = self._prefill_capacity_configs(chunk_size=8)
        model.attn_config.tokens_per_block = 1
        engine.runtime_config.max_generate_batch_size = 3
        engine.hw_kernel_config.enable_cuda_graph = True
        engine.parallelism_config.tp_size = 8
        engine.parallelism_config.enable_sp = True
        engine.sp_config.type = SpeculativeType.DSPARK
        engine.sp_config.gen_num_per_cycle = 3
        engine.sp_config.sp_dspark_sample_from_anchor = True
        ModelFactory.update_engine_config_from_model_config(engine, model)
        # Width 3 pads to eight requests (24 tokens); width 4 needs four (16).
        self.assertEqual(model.moe_prefill_max_tokens_per_rank, 24)

    def test_generation_prefill_graph_uses_token_bucket_only_on_owning_role(self):
        for role, kind, expected in (
            (RoleType.PREFILL, SpeculativeType.NONE, 128),
            (RoleType.PDFUSION, SpeculativeType.NONE, 1024),
            (RoleType.PDFUSION, SpeculativeType.MTP, 128),
        ):
            with self.subTest(role=role, kind=kind):
                engine, model = self._prefill_capacity_configs(chunk_size=128)
                engine.pd_sep_config.role_type = role
                engine.sp_config.type = kind
                engine.sp_config.gen_num_per_cycle = 1
                engine.hw_kernel_config.enable_cuda_graph = True
                engine.hw_kernel_config.generation_prefill_capture_token_buckets = [
                    256,
                    1024,
                ]
                ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, expected)

    def test_system_prompt_capacity_is_set_after_tokenization_before_loading(self):
        for source in ("file", "string", "tokens"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as root:
                engine, model = self._prefill_capacity_configs(chunk_size=128)
                engine.parallelism_config.tp_size = 8
                engine.parallelism_config.enable_sp = True
                kv = engine.kv_cache_config
                prompts = json.dumps([{"task_id": "task", "prompt": "prompt"}])
                if source == "file":
                    kv.multi_task_prompt = os.path.join(root, "prompts.json")
                    with open(kv.multi_task_prompt, "w") as out:
                        out.write(prompts)
                elif source == "string":
                    kv.multi_task_prompt_str = prompts
                else:
                    kv.insertMultiTaskPromptTokens("task", list(range(257)))
                ModelFactory.update_engine_config_from_model_config(engine, model)
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, 128)
                tokenizer = SimpleNamespace(encode=lambda _: list(range(257)))
                with patch.object(
                    BaseModel,
                    "load_tokenizer",
                    lambda obj: setattr(obj, "tokenizer", tokenizer),
                ), patch.object(BaseModel, "_finalize_output_vocab_config"):
                    BaseModel(
                        model,
                        engine.parallelism_config,
                        engine.hw_kernel_config,
                        kv,
                        None,
                        MoeConfig(),
                        16,
                        None,
                        None,
                        False,
                        None,
                    )
                self.assertEqual(model.moe_prefill_max_tokens_per_rank, 264)

    def test_system_prompts_execute_separately_and_query_cp_pads_each(self):
        engine, model = self._prefill_capacity_configs(chunk_size=8)
        model.attn_config.tokens_per_block = 1
        engine.parallelism_config.tp_size = 4
        engine.parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        ModelFactory.update_engine_config_from_model_config(engine, model)
        engine.kv_cache_config.insertMultiTaskPromptTokens("short", list(range(60)))
        engine.kv_cache_config.insertMultiTaskPromptTokens("long", list(range(129)))
        include_system_prompt_capacity(
            model, engine.kv_cache_config, engine.parallelism_config
        )
        self.assertEqual(model.moe_prefill_max_tokens_per_rank, 136)

    def test_declared_prefill_capacity_reaches_adapter(self):
        config = ModelConfig()
        self.assertIsNone(config.moe_prefill_max_tokens_per_rank)
        config.max_seq_len = 4096
        moe_config = MoeConfig()
        moe_config.ll_num_max_token = 32
        for capacity, expected in ((None, 4096), (0, 0), (7000, 7000)):
            with self.subTest(capacity=capacity):
                config.moe_prefill_max_tokens_per_rank = capacity
                adapter = MoEConfigAdapter(
                    model_config=config,
                    parallelism_config=ParallelismConfig(),
                    moe_config=moe_config,
                )
                self.assertEqual(adapter.prefill_max_tokens_per_rank, expected)
                self.assertEqual(adapter.max_tokens_per_rank, max(32, expected))

    def test_declared_moe_w1_layout_reaches_adapter(self):
        config = ModelConfig()
        self.assertEqual(config.moe_w1_layout, "up_gate")
        for layout in ("gate_up", "up_gate"):
            with self.subTest(layout=layout):
                config.moe_w1_layout = layout
                adapter = MoEConfigAdapter(
                    model_config=config,
                    parallelism_config=ParallelismConfig(),
                    moe_config=MoeConfig(),
                )
                self.assertEqual(adapter.moe_w1_layout, layout)

    def test_physical_experts_must_partition_evenly_across_ep_ranks(self):
        config = ModelConfig()
        config.expert_num = 64
        parallelism = ParallelismConfig()
        parallelism.ep_size = 8
        for redundant in (0, 5, 8):
            with self.subTest(redundant=redundant):
                config.eplb_config.redundant_expert = redundant
                if redundant == 5:
                    with self.assertRaisesRegex(ValueError, "69.*divisible.*8"):
                        MoEConfigAdapter(config, parallelism, MoeConfig())
                else:
                    ranges = []
                    for rank in range(8):
                        parallelism.ep_rank = rank
                        adapter = MoEConfigAdapter(config, parallelism, MoeConfig())
                        ranges.extend(
                            range(adapter.local_expert_start, adapter.local_expert_end)
                        )
                    self.assertEqual(ranges, list(range(64 + redundant)))

    def _assert_independently_sized_qwen_shared_expert(
        self, config, model_cls, shared_width, routed_width, *, stacked=False
    ):
        # Keep the real expert counts and independently specified FFN widths;
        # a small hidden dimension makes the checkpoint inexpensive on CPU.
        config.hidden_size = 8
        parallelism_config = ParallelismConfig()
        adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )

        has_shared = shared_width > 0
        self.assertEqual(config.moe_style, 2 if has_shared else 1)
        self.assertEqual(config.n_shared_experts, int(has_shared))
        self.assertEqual(adapter.n_shared_experts, int(has_shared))
        self.assertEqual(config.inter_size, shared_width)
        self.assertEqual(config.moe_inter_size, routed_width)
        self.assertEqual(adapter.moe_inter_dim, routed_width)

        prefix = "model.language_model." if model_cls is Qwen35Moe else "model."
        mlp = prefix + "layers.0.mlp."
        checkpoint = {
            prefix + "layers.0.input_layernorm.weight": torch.ones(8),
            mlp + "gate.weight": torch.ones(config.expert_num, 8),
        }
        if has_shared:
            checkpoint.update(
                {
                    mlp + "shared_expert_gate.weight": torch.ones(1, 8),
                    mlp
                    + "shared_expert.gate_proj.weight": torch.full(
                        (shared_width, 8), 1.0
                    ),
                    mlp
                    + "shared_expert.up_proj.weight": torch.full(
                        (shared_width, 8), 3.0
                    ),
                    mlp
                    + "shared_expert.down_proj.weight": torch.full(
                        (8, shared_width), 2.0
                    ),
                }
            )
        if stacked:
            checkpoint[mlp + "experts.gate_up_proj"] = torch.cat(
                (
                    torch.full((config.expert_num, routed_width, 8), 1.0),
                    torch.full((config.expert_num, routed_width, 8), 3.0),
                ),
                dim=1,
            )
            checkpoint[mlp + "experts.down_proj"] = torch.full(
                (config.expert_num, 8, routed_width), 2.0
            )
        else:
            for expert_id in range(config.expert_num):
                expert = mlp + f"experts.{expert_id}."
                checkpoint[expert + "gate_proj.weight"] = torch.full(
                    (routed_width, 8), 1.0
                )
                checkpoint[expert + "up_proj.weight"] = torch.full(
                    (routed_width, 8), 3.0
                )
                checkpoint[expert + "down_proj.weight"] = torch.full(
                    (8, routed_width), 2.0
                )

        weight_builder = model_cls.get_weight_cls()(
            model_config=config,
            parallelism_config=parallelism_config,
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight_builder._process_meta([{}], set(checkpoint))
        layer_weights = (
            weight_builder._get_hf_ffn_layer_weight_info(0)
            if model_cls is Qwen2Moe
            else weight_builder._create_ffn_weight()
        )
        # Exercise production checkpoint lookup, merge/transpose and load.
        # Device-specific packing is outside this CPU loader contract.
        exported_device = SimpleNamespace(
            shuffle_moe_weight=lambda tensor, *_args: tensor,
            maybe_rewrite_weight_by_key=lambda _name, tensor, use_swizzle_a=None: tensor,
        )
        with tempfile.TemporaryDirectory() as checkpoint_path:
            save_file(checkpoint, os.path.join(checkpoint_path, "model.safetensors"))
            database = CkptDatabase(checkpoint_path)
            source = DatabaseTensorSource(database)
            load_config = weight_builder.create_load_config(
                torch.float32, database, exported_device=exported_device
            )
            loaded = {}
            for weight in layer_weights:
                loaded.update(weight.load(source, 0, "cpu", load_config))

            if has_shared:
                self.assertEqual(
                    source.load_tensor(mlp + "shared_expert.gate_proj.weight")[0].shape,
                    (config.inter_size, config.hidden_size),
                )
            routed_key = mlp + (
                "experts.gate_up_proj" if stacked else "experts.0.gate_proj.weight"
            )
            self.assertEqual(
                source.load_tensor(routed_key)[0].shape,
                (
                    (config.expert_num, 2 * config.moe_inter_size, config.hidden_size)
                    if stacked
                    else (config.moe_inter_size, config.hidden_size)
                ),
            )

        self.assertEqual(
            loaded[W.moe_w1].shape, (config.expert_num, 2 * routed_width, 8)
        )
        self.assertEqual(loaded[W.moe_w2].shape, (config.expert_num, 8, routed_width))
        self.assertEqual(loaded[W.moe_gate].shape, (8, config.expert_num))
        if has_shared:
            self.assertEqual(loaded[W.ffn_w13].shape, (8, 2 * shared_width))
            self.assertEqual(loaded[W.ffn_w2].shape, (shared_width, 8))
            self.assertEqual(loaded[W.shared_expert_gate].shape, (8, 1))
            # Shared W13 merges gate|up; generic routed W1 loads up|gate.
            self.assertTrue(torch.all(loaded[W.ffn_w13][:, :shared_width] == 1))
            self.assertTrue(torch.all(loaded[W.ffn_w13][:, shared_width:] == 3))
        else:
            self.assertEqual(set(loaded), {W.moe_gate, W.moe_w1, W.moe_w2})
        self.assertTrue(torch.all(loaded[W.moe_w1][:, :routed_width] == 3))
        self.assertTrue(torch.all(loaded[W.moe_w1][:, routed_width:] == 1))

    def _assert_routed_only_weight_descriptors(self, weight_cls, config):
        weight = weight_cls.__new__(weight_cls)
        weight._align_size = 0
        weight._is_gated_activation = True
        weight.model_config = config
        weight.moe_layer_index_ = {1}
        weight.expert_num_ = config.expert_num
        weight.has_e_score_correction_bias = False

        layer_weights = weight._get_hf_ffn_layer_weight_info(1)

        self.assertEqual(len(layer_weights), 1)
        self.assertIsInstance(layer_weights[0], MoeWeight)
        self.assertFalse(any(isinstance(item, FfnWeight) for item in layer_weights))

        exported_device = SimpleNamespace(
            shuffle_moe_weight=lambda tensor, *_args: tensor,
            maybe_rewrite_weight_by_key=lambda _name, tensor, use_swizzle_a=None: tensor,
        )
        load_config = SimpleNamespace(
            compute_dtype=torch.float32,
            merge_lora=False,
            moe_pure_tp_preshard=False,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            use_swizzleA=False,
            exported_device=exported_device,
            get_selected_experts=lambda _layer_id, expert_num: range(expert_num),
        )
        source = _RoutedOnlyTensorSource()
        loaded = layer_weights[0].load(source, 1, "cpu", load_config)

        self.assertEqual(set(loaded), {W.moe_gate, W.moe_w1, W.moe_w2})
        self.assertTrue(source.requested_names)
        self.assertFalse(
            any("shared_experts" in name for name in source.requested_names)
        )

    def test_deepseek_v2_propagates_moe_dimensions(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        self.assertEqual(config.moe_inter_size, 1408)
        self.assertEqual(config.n_shared_experts, 2)
        self.assertEqual(config.inter_size, 2816)
        _assert_moe_config_is_constructible(self, config)

    def test_kimi_k25_propagates_moe_dimensions(self):
        config = ModelConfig()
        KimiK25._populate_text_config(config, _deepseek_text_config())

        self.assertEqual(config.moe_inter_size, 1408)
        self.assertEqual(config.n_shared_experts, 2)
        self.assertEqual(config.inter_size, 2816)
        _assert_moe_config_is_constructible(self, config)

    def test_deepseek_v2_routed_only_config_is_constructible(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(n_shared_experts=0), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        self.assertEqual(config.moe_style, 1)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.inter_size, 0)
        self.assertEqual(config.dense_inter_size, 11008)
        _assert_moe_config_is_constructible(self, config, 0)

    def test_routed_only_param_count_keeps_preceding_dense_layers(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(n_shared_experts=0), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        moe_layer_count = len(config.moe_layer_index)
        dense_layer_count = config.num_layers - moe_layer_count
        ffn_w_count = 3 if config.isGatedActivation() else 2
        expected_moe_weights = (
            moe_layer_count
            * config.moe_inter_size
            * config.hidden_size
            * ffn_w_count
            * config.expert_num
        )
        self.assertEqual(config.moe_weight_param_count(), expected_moe_weights)

        count_with_dense_layers = config.layer_weight_param_count()
        config.dense_inter_size = 0
        count_without_dense_layers = config.layer_weight_param_count()
        self.assertEqual(
            count_with_dense_layers - count_without_dense_layers,
            dense_layer_count * 11008 * config.hidden_size * ffn_w_count,
        )

    def test_kimi_k25_routed_only_config_is_constructible(self):
        config = ModelConfig()
        KimiK25._populate_text_config(config, _deepseek_text_config(n_shared_experts=0))

        self.assertEqual(config.moe_style, 1)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.inter_size, 0)
        self.assertEqual(config.dense_inter_size, 11008)
        _assert_moe_config_is_constructible(self, config, 0)

    def test_deepseek_v2_routed_only_weights_skip_shared_expert_keys(self):
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(_deepseek_text_config(n_shared_experts=0), writer)
            config = ModelConfig()
            DeepSeekV2._from_hf(config, ckpt_path)

        self._assert_routed_only_weight_descriptors(DeepSeekV2Weight, config)

    def test_kimi_k25_routed_only_weights_skip_shared_expert_keys(self):
        config = ModelConfig()
        KimiK25._populate_text_config(config, _deepseek_text_config(n_shared_experts=0))

        self._assert_routed_only_weight_descriptors(KimiK25Weight, config)

    def test_qwen3_next_propagates_independently_sized_shared_expert(self):
        config = ModelConfig()
        config.num_layers = 2
        Qwen3Next._parse_moe_config(
            {
                "num_experts_per_tok": 8,
                "num_experts": 512,
                "moe_intermediate_size": 768,
                "shared_expert_intermediate_size": 2048,
                "decoder_sparse_step": 1,
            },
            config,
        )

        self._assert_independently_sized_qwen_shared_expert(
            config, Qwen3Next, 2048, 768
        )

    def test_qwen35_propagates_independently_sized_shared_expert(self):
        config = ModelConfig()
        config.num_layers = 2
        Qwen35Moe._parse_moe_config(
            {
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "moe_intermediate_size": 768,
                "shared_expert_intermediate_size": 2048,
                "decoder_sparse_step": 1,
            },
            config,
        )

        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                self._assert_independently_sized_qwen_shared_expert(
                    config, Qwen35Moe, 2048, 768, stacked=stacked
                )

    def test_qwen2_moe_propagates_independently_sized_shared_expert(self):
        config = ModelConfig()
        config.num_layers = 2
        Qwen2Moe.load_moe_config(
            config,
            {
                "num_experts_per_tok": 4,
                "num_experts": 60,
                "moe_intermediate_size": 1408,
                "shared_expert_intermediate_size": 5632,
                "decoder_sparse_step": 1,
            },
        )

        self._assert_independently_sized_qwen_shared_expert(
            config, Qwen2Moe, 5632, 1408
        )

    def test_qwen_routed_only_checkpoint_has_no_shared_expert_lookups(self):
        for model_cls in (Qwen2Moe, Qwen3Next, Qwen35Moe):
            for shared_width in (None, 0):
                with self.subTest(model=model_cls.__name__, shared_width=shared_width):
                    config = ModelConfig()
                    config.num_layers = 1
                    hf_config = {
                        "num_experts_per_tok": 1,
                        "num_experts": 2,
                        "moe_intermediate_size": 16,
                        "decoder_sparse_step": 1,
                    }
                    if shared_width is not None:
                        hf_config["shared_expert_intermediate_size"] = shared_width
                    if model_cls is Qwen2Moe:
                        model_cls.load_moe_config(config, hf_config)
                    else:
                        model_cls._parse_moe_config(hf_config, config)
                    layouts = (False, True) if model_cls is Qwen35Moe else (False,)
                    for stacked in layouts:
                        self._assert_independently_sized_qwen_shared_expert(
                            config, model_cls, 0, 16, stacked=stacked
                        )


if __name__ == "__main__":
    unittest.main()
