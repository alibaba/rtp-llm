"""CPU contract tests for the experimental TRT sparse Decode integration.

The service package imports CUDA extensions at module import time. These tests
compile unchanged function/class AST nodes from the production files with small
dependency stubs instead. They exercise real dispatch/prepare/CMP methods, not
copies of their logic. They do not validate CUDA stream ordering or numerics;
those require the separate GPU tests. No CUDA context is created here.
"""

import ast
import logging
import os
import sys
import unittest
from abc import abstractmethod
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

ATTENTION = Path(__file__).resolve().parents[2]
MODULES = ATTENTION.parents[1]
NEW_MODULE = (
    "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.trtllm_sparse_impl"
)
ENV = "GLM5_SPARSE_DECODE_BACKEND"


def compile_nodes(path, namespace, names):
    """Retain real method bodies; omit only the extension-loading imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != set(names):
        raise AssertionError(
            f"Missing production nodes in {path}: {set(names) - found}"
        )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def sparse_namespace():
    namespace = {
        "__name__": "_cpu_sparse_contract",
        "torch": torch,
        "abstractmethod": abstractmethod,
        "os": os,
        "logging": logging,
    }
    compile_nodes(ATTENTION / "fmha_impl_base.py", namespace, {"MlaImplBase"})
    compile_nodes(
        ATTENTION / "cuda_mla_impl/flashmla_sparse_impl.py",
        namespace,
        {
            "_topk_2d",
            "_is_multi_token_decode",
            "SparseMlaOp",
            "SparseMlaFp8Op",
            "SparseMlaImpl",
        },
    )
    return namespace


def trt_namespace():
    namespace = sparse_namespace()
    namespace["KvCacheDataType"] = SimpleNamespace(FP8="fp8", BASE="base")
    compile_nodes(
        ATTENTION / "cuda_mla_impl/trtllm_sparse_impl.py",
        namespace,
        {"TrtllmSparseMlaFp8Op", "TrtllmSparseMlaImpl"},
    )
    return namespace


class CpuOnlyTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # An accidental CUDA allocation or stream operation must fail, not turn
        # this CPU test into a GPU test when run on a CUDA-enabled machine.
        for name in ("_lazy_init", "synchronize", "stream", "set_device"):
            guard = patch.object(
                torch.cuda,
                name,
                side_effect=AssertionError(f"CPU test called CUDA {name}"),
            )
            guard.start()
            self.addCleanup(guard.stop)


class FactoryDispatchTest(CpuOnlyTest):
    def setUp(self):
        super().setUp()
        self.namespace = {
            "__name__": "_cpu_attention_factory",
            "os": os,
            "logging": logging,
            "KvCacheDataType": SimpleNamespace(FP8="fp8", BASE="base"),
            "W": SimpleNamespace(rope_cos_sin_cache="rope"),
            "get_mla_impl": Mock(),
            "get_fmha_impl": Mock(),
        }
        compile_nodes(
            ATTENTION / "attn_factory.py",
            self.namespace,
            {"_get_glm5_trtllm_impl", "AttnImplFactory"},
        )
        self.selector = self.namespace["_get_glm5_trtllm_impl"]
        self.factory = self.namespace["AttnImplFactory"]
        self.model = SimpleNamespace(
            model_type="glm_5", quant_config=None, max_seq_len=8192
        )
        self.config = SimpleNamespace(
            is_sparse=True,
            use_mla=True,
            kv_cache_dtype="fp8",
            head_num=64,
            kv_lora_rank=512,
            nope_head_dim=192,
            rope_head_dim=64,
            kernel_tokens_per_block=64,
            indexer_topk=2048,
        )
        self.parallelism = SimpleNamespace(
            tp_size=1,
            get_attn_tp_size=lambda: 1,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False, kv_cache_sharded=False, prefill_cp_size=0
            ),
        )
        self.model.getAttentionConfigs = Mock(return_value=self.config)
        self.weight = SimpleNamespace(weights=[{}], get_global_weight=Mock())
        self.inputs = self.inputs_for("decode")
        self.impl = Mock(name="constructed_trt_impl")
        self.impl.support_cuda_graph.return_value = True
        self.backend = Mock(name="TrtllmSparseMlaImpl", return_value=self.impl)
        self.backend.support = Mock(
            wraps=trt_namespace()["TrtllmSparseMlaImpl"].support
        )
        self.backend.support_parallelism_config = Mock(
            wraps=trt_namespace()["TrtllmSparseMlaImpl"].support_parallelism_config
        )
        self.module = ModuleType(NEW_MODULE)
        self.module.TrtllmSparseMlaImpl = self.backend
        module_patch = patch.dict(sys.modules, {NEW_MODULE: self.module})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    @staticmethod
    def inputs_for(role):
        return SimpleNamespace(
            is_prefill=role != "decode",
            is_target_verify=role == "verify",
            is_draft_extend=role == "extend",
        )

    def select(self, graph=False):
        return self.selector(
            self.model,
            self.config,
            self.parallelism,
            self.weight,
            self.inputs,
            None,
            graph,
        )

    def test_default_and_explicit_flashmla_do_not_construct_trt(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            self.assertIsNone(self.select())
        with patch.dict(os.environ, {ENV: "flashmla"}):
            self.assertIsNone(self.select())
        self.backend.assert_not_called()

    def test_opt_in_covers_decode_verify_and_draft_extend_for_both_models(self):
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            for model in ("glm_5", "glm_5_mtp"):
                for role in ("decode", "verify", "extend"):
                    with self.subTest(model=model, role=role):
                        self.model.model_type = model
                        self.inputs = self.inputs_for(role)
                        self.assertIs(self.select(graph=True), self.impl)
        self.assertEqual(self.backend.call_count, 6)
        for call in self.backend.call_args_list:
            self.assertIs(call.kwargs["is_cuda_graph"], True)

    def test_ordinary_prefill_stays_on_existing_backend(self):
        self.inputs = self.inputs_for("prefill")
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            self.assertIsNone(self.select(graph=False))
            self.assertIsNone(self.select(graph=True))
        self.backend.assert_not_called()

    def test_actual_cp_and_rr_sharding_rejected_before_construction(self):
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            for enabled in (False, True):
                for tp_size in (1, 2, 8):
                    for sharded in (False, True):
                        if not (enabled or (sharded and tp_size > 1)):
                            continue
                        self.parallelism.tp_size = tp_size
                        self.parallelism.prefill_cp_config = SimpleNamespace(
                            is_enabled=lambda enabled=enabled: enabled,
                            kv_cache_sharded=sharded,
                            prefill_cp_size=8,
                        )
                        for role in ("decode", "verify", "extend"):
                            self.inputs = self.inputs_for(role)
                            for graph in (False, True):
                                with self.subTest(
                                    enabled=enabled,
                                    tp=tp_size,
                                    sharded=sharded,
                                    role=role,
                                    graph=graph,
                                ):
                                    with self.assertRaisesRegex(
                                        ValueError, "parallelism"
                                    ):
                                        self.select(graph=graph)
        # Fail before reading weights, constructing native metadata or scratch.
        self.backend.assert_not_called()
        self.weight.get_global_weight.assert_not_called()

    def test_cp_metadata_only_and_unsharded_tp_remain_supported(self):
        configurations = [None]
        for tp_size, sharded in ((1, False), (1, True), (2, False), (8, False)):
            configurations.append(
                SimpleNamespace(
                    tp_size=tp_size,
                    prefill_cp_config=SimpleNamespace(
                        is_enabled=lambda: False,
                        kv_cache_sharded=sharded,
                        # Prefill topology can exceed the Decode TP size.
                        prefill_cp_size=8,
                    ),
                )
            )
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            for config in configurations:
                self.parallelism = config
                for role in ("decode", "verify", "extend"):
                    self.inputs = self.inputs_for(role)
                    for graph in (False, True):
                        with self.subTest(config=config, role=role, graph=graph):
                            self.assertIs(self.select(graph=graph), self.impl)
                            self.backend.support_parallelism_config.assert_called_with(
                                config
                            )

    def test_parallelism_rejection_does_not_fall_back(self):
        self.backend.support_parallelism_config.return_value = False
        fallback = Mock()
        self.factory.FMHA_IMPL_REGISTRY = {"mla": fallback}
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            with self.assertRaisesRegex(ValueError, "parallelism"):
                self.factory.get_fmha_impl(
                    self.model,
                    self.parallelism,
                    self.weight,
                    self.inputs,
                    None,
                    True,
                )
        fallback.assert_not_called()
        self.backend.assert_not_called()
        self.weight.get_global_weight.assert_not_called()

    def test_cp_guard_does_not_change_default_backend_or_ordinary_prefill(self):
        self.parallelism.tp_size = 8
        self.parallelism.prefill_cp_config = SimpleNamespace(
            is_enabled=lambda: True, kv_cache_sharded=True
        )
        self.backend.support_parallelism_config.side_effect = AssertionError(
            "unselected TRT backend must not check parallelism"
        )
        with patch.dict(os.environ, {ENV: "flashmla"}):
            self.assertIsNone(self.select(graph=True))
        self.inputs = self.inputs_for("prefill")
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            self.assertIsNone(self.select(graph=False))
            self.assertIsNone(self.select(graph=True))
        self.backend.assert_not_called()
        self.backend.support_parallelism_config.assert_not_called()

    def test_unsupported_model_or_attention_does_not_fallback(self):
        for field, value in (
            ("model_type", "deepseek_v3"),
            ("is_sparse", False),
            ("use_mla", False),
        ):
            owner = self.model if field == "model_type" else self.config
            old = getattr(owner, field)
            with self.subTest(field=field), patch.dict(os.environ, {ENV: "trtllm_gen"}):
                setattr(owner, field, value)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.select()
            setattr(owner, field, old)
        self.backend.assert_not_called()

    def test_backend_rejection_and_constructor_error_are_not_swallowed(self):
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            self.backend.support.return_value = False
            with self.assertRaises((ValueError, RuntimeError)):
                self.select()
            self.backend.assert_not_called()
            self.backend.support.return_value = True
            self.backend.side_effect = RuntimeError("deliberate constructor failure")
            with self.assertRaisesRegex(RuntimeError, "deliberate constructor failure"):
                self.select()

    def test_unknown_switch_is_rejected(self):
        with patch.dict(os.environ, {ENV: "trtllm_typo"}):
            with self.assertRaises((ValueError, RuntimeError)):
                self.select()

    def test_incompatible_shapes_or_cache_are_not_selected(self):
        for field, value in (
            ("kv_cache_dtype", "base"),
            ("head_num", 4),
            ("kv_lora_rank", 256),
            ("nope_head_dim", 128),
            ("rope_head_dim", 32),
            ("kernel_tokens_per_block", 128),
            ("indexer_topk", 1024),
        ):
            original = getattr(self.config, field)
            with self.subTest(field=field), patch.dict(os.environ, {ENV: "trtllm_gen"}):
                setattr(self.config, field, value)
                with self.assertRaises(ValueError):
                    self.select()
            setattr(self.config, field, original)
        self.backend.assert_not_called()

    def test_disabled_flashinfer_does_not_import_or_construct_backend(self):
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            with self.assertRaisesRegex(ValueError, "disable_flash_infer"):
                self.selector(
                    self.model,
                    self.config,
                    self.parallelism,
                    self.weight,
                    self.inputs,
                    SimpleNamespace(disable_flash_infer=True),
                    False,
                )
        self.backend.assert_not_called()

    def test_outer_factory_returns_selected_backend_and_preserves_default(self):
        old_impl = SimpleNamespace()
        old_dispatch = Mock(return_value=old_impl)
        self.factory.FMHA_IMPL_REGISTRY = {"mla": old_dispatch}
        arguments = (self.model, self.parallelism, self.weight, self.inputs, None, True)
        with patch.dict(os.environ, {ENV: "trtllm_gen"}):
            self.assertIs(self.factory.get_fmha_impl(*arguments), self.impl)
        old_dispatch.assert_not_called()
        with patch.dict(os.environ, {ENV: "flashmla"}):
            self.assertIs(self.factory.get_fmha_impl(*arguments), old_impl)
        old_dispatch.assert_called_once()


class ReplayPrepareContractTest(CpuOnlyTest):
    def setUp(self):
        super().setUp()
        self.namespace = trt_namespace()

    def prepare_case(self, role, empty_kernel_table=False, indexer_metadata=True):
        namespace = self.namespace
        # Use the real new subclass, not a synthetic type with matching names.
        # CUDA compute is irrelevant to this CPU metadata branch contract.
        op_type = namespace["TrtllmSparseMlaFp8Op"]
        op = object.__new__(op_type)
        op.plan = Mock()
        instance = object.__new__(namespace["TrtllmSparseMlaImpl"])
        instance.fmha_impl = op
        instance.seq_size_per_block = 64
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cpu")
        inputs = SimpleNamespace(
            is_prefill=role != "decode",
            is_target_verify=role == "verify",
            is_draft_extend=role == "extend",
            input_lengths=torch.tensor([4, 4], device="cpu", dtype=torch.int32),
            prefix_lengths=torch.tensor([30, 70], device="cpu", dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.tensor(
                [31, 71], device="cpu", dtype=torch.int32
            ),
            kv_cache_kernel_block_id_device=(
                torch.empty((0, 0), dtype=torch.int32, device="cpu")
                if empty_kernel_table
                else table
            ),
            kv_cache_block_id_device=table,
        )
        lengths = torch.tensor([31, 71], device="cpu", dtype=torch.int32)
        params = SimpleNamespace(
            multi_token_decode_total_tokens=8 if role != "decode" else 0,
            expanded_seq_lens=lengths,
            positions_d=torch.tensor([30, 70], device="cpu", dtype=torch.int32),
            fill_multi_token_decode_cuda_graph_params=Mock(),
            fill_sparse_mla_decode_cuda_graph_params=Mock(),
            fill_target_verify_cuda_graph_params=Mock(),
        )
        original_positions = params.positions_d
        instance.fmha_params = params
        instance.prepare = Mock(
            side_effect=AssertionError("fell back to eager prepare")
        )
        instance._refresh_paged_mqa_schedule_metadata = Mock()
        instance._cuda_dag_indexer_metadata = (
            SimpleNamespace(prepare=Mock()) if indexer_metadata else None
        )
        namespace["SparseMlaImpl"].prepare_cuda_graph(instance, inputs)
        self.assertIs(instance.fmha_params, params)
        self.assertIs(params.expanded_seq_lens, lengths)
        self.assertIs(params.positions_d, original_positions)
        op.plan.assert_called_once_with(params, table, attn_inputs=inputs)
        instance._refresh_paged_mqa_schedule_metadata.assert_called_once_with(
            inputs, forbid_realloc=True
        )
        instance.prepare.assert_not_called()
        params.fill_target_verify_cuda_graph_params.assert_not_called()
        if role == "decode":
            params.fill_sparse_mla_decode_cuda_graph_params.assert_called_once_with(
                inputs.sequence_lengths_plus_1_d, table, 64
            )
            params.fill_multi_token_decode_cuda_graph_params.assert_not_called()
        else:
            params.fill_multi_token_decode_cuda_graph_params.assert_called_once_with(
                inputs.input_lengths, inputs.prefix_lengths, table, 64
            )
            params.fill_sparse_mla_decode_cuda_graph_params.assert_not_called()
        if indexer_metadata:
            instance._cuda_dag_indexer_metadata.prepare.assert_called_once_with()

    def test_decode_verify_and_extend_keep_device_only_prepare(self):
        for role in ("decode", "verify", "extend"):
            for empty_table in (False, True):
                with self.subTest(role=role, empty_table=empty_table):
                    self.prepare_case(role, empty_table)

    def test_optional_indexer_metadata_can_be_absent(self):
        self.prepare_case("decode", indexer_metadata=False)


class BackendPlanContractTest(CpuOnlyTest):
    def setUp(self):
        super().setUp()
        self.namespace = trt_namespace()
        self.op_type = self.namespace["TrtllmSparseMlaFp8Op"]

    def make_op(self, heads=8):
        decode_module = ModuleType("flashinfer.decode")
        decode_module.trtllm_batch_decode_with_kv_cache_mla = Mock()
        with patch.dict(sys.modules, {"flashinfer.decode": decode_module}):
            return self.op_type(
                num_heads=heads,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                qk_nope_head_dim=192,
                page_size=64,
                softmax_extra_scale=1.25,
                top_k=2048,
                use_cuda_graph=True,
            )

    @staticmethod
    def metadata():
        params = SimpleNamespace(
            batch_indice_d=torch.tensor([0, 1], dtype=torch.int32, device="cpu"),
            expanded_seq_lens=torch.tensor([31, 71], dtype=torch.int32, device="cpu"),
            kvlen_d=torch.tensor([31, 71], dtype=torch.int32, device="cpu"),
            positions_d=torch.tensor([30, 70], dtype=torch.int32, device="cpu"),
        )
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cpu")
        return params, table

    def test_real_constructor_retains_type_and_injects_op(self):
        op = self.make_op()
        self.assertIsInstance(op, self.namespace["SparseMlaFp8Op"])
        self.assertTrue(op.expects_paged_kv)
        self.assertTrue(op.use_cuda_graph)
        self.assertEqual(op.scale, 1.25 / 16)
        base_init = Mock(return_value=None)
        with patch.object(self.namespace["SparseMlaImpl"], "__init__", base_init):
            self.namespace["TrtllmSparseMlaImpl"]("configs", "inputs", marker="kept")
        base_init.assert_called_once_with(
            "configs", "inputs", fmha_impl=self.op_type, marker="kept"
        )

    def test_plan_preserves_metadata_without_flash_scheduler_or_gather(self):
        for role in ("decode", "verify", "extend"):
            with self.subTest(role=role):
                op = self.make_op()
                params, table = self.metadata()
                lengths, positions = params.expanded_seq_lens, params.positions_d
                op._reserve = Mock()
                op._reset_sched_meta = Mock(
                    side_effect=AssertionError("Flash scheduler")
                )
                op._build_gather_workspace = Mock(
                    side_effect=AssertionError("Prefill gather")
                )
                with patch.object(
                    self.namespace["SparseMlaFp8Op"],
                    "plan",
                    side_effect=AssertionError("called Flash parent plan"),
                ):
                    op.plan(params, table, FactoryDispatchTest.inputs_for(role))
                self.assertIs(op.mla_params, params)
                self.assertIs(op.block_table, table)
                self.assertIs(op._seq_lens, lengths)
                self.assertIs(params.expanded_seq_lens, lengths)
                self.assertIs(params.positions_d, positions)
                self.assertIsNone(op._sched_meta)
                self.assertIsNone(op._gather)
                op._reserve.assert_called_once_with(2, torch.device("cpu"))
                op._reset_sched_meta.assert_not_called()
                op._build_gather_workspace.assert_not_called()

    def test_plan_uses_kv_lengths_only_when_expanded_lengths_are_empty(self):
        op = self.make_op()
        params, table = self.metadata()
        params.expanded_seq_lens = torch.empty(0, dtype=torch.int32, device="cpu")
        empty = params.expanded_seq_lens
        op._reserve = Mock()
        op.plan(params, table, FactoryDispatchTest.inputs_for("decode"))
        self.assertIs(op._seq_lens, params.kvlen_d)
        self.assertIs(params.expanded_seq_lens, empty)

    def test_plan_rejects_prefill_and_invalid_metadata_before_allocation(self):
        op = self.make_op()
        params, table = self.metadata()
        op._reserve = Mock()
        with self.assertRaises(ValueError):
            op.plan(params, table, FactoryDispatchTest.inputs_for("prefill"))
        params.expanded_seq_lens = params.expanded_seq_lens.long()
        with self.assertRaises(ValueError):
            op.plan(params, table, FactoryDispatchTest.inputs_for("decode"))
        op._reserve.assert_not_called()

    def test_graph_shape_change_cannot_replace_existing_scratch(self):
        op = self.make_op()
        op._capacity = 2
        op._device = torch.device("cpu")
        marker = object()
        op.selected_kv = marker
        # Same already-reserved metadata is a no-op before any CUDA check.
        op._reserve(2, torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "captured scratch"):
            op._reserve(4, torch.device("cpu"))
        self.assertIs(op.selected_kv, marker)
        self.assertEqual(op._capacity, 2)


class CmpDispatchContractTest(CpuOnlyTest):
    def test_paged_backend_receives_unmodified_absorbed_query_cache_and_topk(self):
        namespace = {"__name__": "_cpu_cmp_contract", "torch": torch}
        compile_nodes(MODULES / "hybrid/glm5_cmp.py", namespace, {"Glm5Cmp"})
        cmp = object.__new__(namespace["Glm5Cmp"])
        cmp.layer_idx = 7
        result = torch.empty((2, 8, 512), dtype=torch.bfloat16, device="cpu")
        op = object.__new__(trt_namespace()["TrtllmSparseMlaFp8Op"])
        op.forward = Mock(return_value=result)
        implementation = SimpleNamespace(
            weights=[{}], fmha_params=object(), fmha_impl=op
        )
        query = torch.empty((2, 8, 576), dtype=torch.bfloat16, device="cpu")
        topk = torch.zeros((2, 2048), dtype=torch.int32, device="cpu")
        cache = torch.empty((3, 64, 656), dtype=torch.uint8, device="cpu")
        layer_cache = SimpleNamespace(kv_cache_base=cache)
        for wrapper in (implementation, SimpleNamespace(fmha_impl=implementation)):
            with self.subTest(nested=wrapper is not implementation):
                actual = cmp.sparse_mla(query, topk, wrapper, layer_cache)
                self.assertIs(actual, result)
                args, kwargs = op.forward.call_args
                self.assertIs(args[0], query)
                self.assertIs(args[1], cache)
                self.assertIs(args[2], topk)
                self.assertEqual(kwargs, {"layer_id": 7})
        self.assertEqual(op.forward.call_count, 2)


class ForwardCallContractTest(CpuOnlyTest):
    def make_forward_state(self, tokens=2):
        namespace = trt_namespace()
        op = object.__new__(namespace["TrtllmSparseMlaFp8Op"])
        op._capacity = tokens
        op._device = torch.device("cpu")
        op.num_heads, op.kv_lora_rank = 8, 512
        op.qk_nope_head_dim, op.qk_rope_head_dim = 192, 64
        op.token_per_block = 64
        op.top_k, op.scale = 2048, 1.25 / 16
        op._seq_lens = torch.full((tokens,), 71, dtype=torch.int32, device="cpu")
        op.mla_params = SimpleNamespace(
            batch_indice_d=torch.zeros(tokens, dtype=torch.int32, device="cpu"),
            expanded_seq_lens=op._seq_lens,
        )
        op.block_table = torch.ones((1, 2), dtype=torch.int32, device="cpu")
        op.output = torch.empty((tokens, 8, 512), dtype=torch.bfloat16, device="cpu")
        # CUDA producer/consumer buffers are opaque objects to these call-boundary
        # tests. No surrogate attention or quantization algorithm is run on CPU.
        for name in (
            "q_fp8",
            "selected_kv",
            "source_indices",
            "physical_indices",
            "valid_counts",
            "trt_seq_lens",
            "_query_view",
            "_selected_paged",
            "workspace_buffer",
            "_indices_view",
            "_output_view",
        ):
            setattr(op, name, object())
        calls = Mock()
        namespace["convert_selected_kv"] = calls.convert
        namespace["mask_empty_output"] = calls.mask
        op._decode = calls.decode
        query = torch.zeros((tokens, 8, 576), dtype=torch.bfloat16, device="cpu")
        cache = torch.zeros((3, 64, 656), dtype=torch.uint8, device="cpu")
        topk = torch.zeros((tokens, 2048), dtype=torch.int32, device="cpu")
        return op, calls, query, cache, topk

    def test_actual_forward_converts_then_decodes_then_masks(self):
        for expanded in (False, True):
            with self.subTest(expanded_topk=expanded):
                op, calls, query, cache, topk = self.make_forward_state()
                logical = topk.unsqueeze(1) if expanded else topk
                lengths = op.mla_params.expanded_seq_lens
                result = op.forward(query, cache, logical, layer_id=7)
                self.assertIs(result, op.output)
                self.assertEqual(result.shape, (2, 8, 512))
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertEqual(
                    [call[0] for call in calls.mock_calls],
                    ["convert", "decode", "mask"],
                )
                args, kw = calls.convert.call_args
                self.assertIs(args[0], query)
                self.assertIs(args[1], cache)
                torch.testing.assert_close(args[2], topk)
                self.assertIs(args[3], op.mla_params.batch_indice_d)
                self.assertIs(args[4], op.block_table)
                self.assertIs(args[5], lengths)
                self.assertIs(kw["lengths_out"], op.trt_seq_lens)
                self.assertIsNot(kw["lengths_out"], lengths)
                trt = calls.decode.call_args.kwargs
                self.assertIs(trt["seq_lens"], op.trt_seq_lens)
                self.assertIs(trt["block_tables"], op._indices_view)
                self.assertIs(trt["kv_cache"], op._selected_paged)
                self.assertIs(trt["out"], op._output_view)
                self.assertEqual(trt["max_seq_len"], 2048)
                self.assertEqual(trt["sparse_mla_top_k"], 2048)
                self.assertEqual(trt["bmm1_scale"], 1.25 / 16)
                self.assertEqual(trt["bmm2_scale"], 1.0)
                self.assertEqual(trt["backend"], "trtllm-gen")
                calls.mask.assert_called_once_with(op.output, op.valid_counts)
                self.assertIs(op.mla_params.expanded_seq_lens, lengths)
                self.assertTrue(torch.equal(lengths, torch.full_like(lengths, 71)))

    def test_empty_token_set_does_not_launch_conversion_or_decode(self):
        op, calls, query, cache, topk = self.make_forward_state(tokens=0)
        self.assertIs(op.forward(query, cache, topk), op.output)
        self.assertEqual(calls.mock_calls, [])

    def test_resident_physical_indices_reach_converter_without_replacing_topk(self):
        for expanded in (False, True):
            with self.subTest(expanded=expanded):
                op, calls, query, cache, topk = self.make_forward_state()
                physical = torch.full_like(topk, 129)
                supplied = physical.unsqueeze(1) if expanded else physical
                self.assertIs(
                    op.forward(query, cache, topk, physical_indices=supplied), op.output
                )
                args, kwargs = calls.convert.call_args
                self.assertIs(args[2], topk)
                self.assertEqual(
                    kwargs["physical_indices"].data_ptr(), physical.data_ptr()
                )
                self.assertEqual(kwargs["physical_indices"].shape, topk.shape)
                self.assertIs(kwargs["indices_out"], op.physical_indices)
                self.assertIsNot(kwargs["physical_indices"], kwargs["indices_out"])
                self.assertEqual(
                    [call[0] for call in calls.mock_calls],
                    ["convert", "decode", "mask"],
                )

    def test_invalid_resident_indices_fail_before_any_launch(self):
        for tokens in (0, 2):
            op, calls, query, cache, topk = self.make_forward_state(tokens=tokens)
            bad_inputs = [
                topk.long(),
                topk[:, :1024],
                torch.empty((tokens, 2, 2048), dtype=torch.int32),
                torch.empty((tokens, 2048), dtype=torch.int32, device="meta"),
            ]
            if tokens:
                bad_inputs.append(
                    torch.empty((tokens, 4096), dtype=torch.int32)[:, ::2]
                )
            for physical in bad_inputs:
                with self.subTest(
                    tokens=tokens, shape=physical.shape, dtype=physical.dtype
                ):
                    with self.assertRaisesRegex(ValueError, "physical_indices"):
                        op.forward(query, cache, topk, physical_indices=physical)
            self.assertEqual(calls.mock_calls, [])

    def test_invalid_query_or_topk_shape_is_rejected_before_launch(self):
        op, calls, query, cache, topk = self.make_forward_state()
        for bad_query, bad_topk in (
            (query.float(), topk),
            (query[:, :4], topk),
            (query, topk[:, :1024]),
            (query, topk[:, None, :].expand(-1, 2, -1)),
        ):
            with self.subTest(query=bad_query.shape, topk=bad_topk.shape):
                with self.assertRaises(ValueError):
                    op.forward(bad_query, cache, bad_topk)
        self.assertEqual(calls.mock_calls, [])

    def test_invalid_physical_page_shape_is_rejected_before_launch(self):
        op, calls, query, _, topk = self.make_forward_state()
        for shape in (
            (3, 128, 656),
            (3, 32, 656),
            (3, 64, 576),
            (3, 64, 2, 656),
            (3, 64 * 656),
        ):
            with self.subTest(shape=shape):
                cache = torch.zeros(shape, dtype=torch.uint8, device="cpu")
                with self.assertRaisesRegex(ValueError, "paged KV"):
                    op.forward(query, cache, topk)
        self.assertEqual(calls.mock_calls, [])

    def test_fp8_typed_cache_is_reinterpreted_without_changing_packed_bytes(self):
        op, calls, query, _, topk = self.make_forward_state()
        # Include NaN encodings: a numeric cast would not preserve these scale
        # and BF16 bytes. This is a byte-view contract, not FP8 arithmetic.
        raw = torch.arange(3 * 64 * 656, dtype=torch.int64, device="cpu")
        raw = raw.to(torch.uint8).view(3, 64, 1, 656)
        exposed = raw.view(torch.float8_e4m3fn)
        op.forward(query, exposed, topk)
        passed = calls.convert.call_args.args[1]
        self.assertEqual(passed.dtype, torch.uint8)
        self.assertEqual(passed.shape, raw.shape)
        self.assertEqual(passed.data_ptr(), raw.data_ptr())
        self.assertTrue(torch.equal(passed, raw))


if __name__ == "__main__":
    unittest.main()
