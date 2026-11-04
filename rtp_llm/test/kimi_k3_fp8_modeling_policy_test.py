"""Offline dispatch contracts; no Torch, CUDA or compiled RTP imports.

Compile the production methods without their accelerator imports. These tests
exercise modeling selection and producer/consumer boundaries, not GPU numerics.
"""

import ast
from contextlib import nullcontext
import logging
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "models_py/model_desc/kimi_k3.py"
DSA = ROOT / "models_py/modules/kimi_k3/dsa.py"
COLLECTIVE = "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion"


def method(class_name, name, namespace, path=MODELING):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class KimiK3Model:
    pass


class KimiK3DenseMLP:
    pass


class ProjectionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.created = []
        owner = self

        class Collective:
            def __init__(self, group, device, **kwargs):
                self.group, self.device = group, device
                self.__dict__.update(kwargs)
                owner.created.append(self)

            def reduce_scatter_bf16(self, value):
                return value

        self.module = ModuleType(COLLECTIVE)
        self.module.Fp8CollectiveProjection = Collective
        self.initialize = method("KimiK3Model", "_initialize_fp8_collective", {
            "KimiK3Model": KimiK3Model,
            "KimiK3DenseMLP": KimiK3DenseMLP,
            "RoleType": NS(DECODE="DECODE"),
            "Group": NS(TP="TP"),
            "_get_group": lambda group: group,
            "logging": logging,
        })

    def model(self, role="PREFILL", quantized=True, fp8_input=True):
        result = KimiK3Model()
        result._fp8_collective = None
        result.tp_size = 8
        result.fp8_collective_max_tokens = 65536
        result._max_moe_tokens = 192 if role == "DECODE" else 65536
        result.parallelism_config = NS(role_type=role)
        result.config = NS(
            hidden_size=7168,
            k3_runtime_config=NS(attn_res_block_size=1),
            attention_projection_quant_config=(
                NS(get_method=lambda: "FP8_PER_BLOCK") if quantized else None
            ),
        )
        attention = NS(
            input=NS(scale_ue8m0=True, weight=NS(device="cuda:0")),
            output=NS(scale_ue8m0=True),
            supports_fp8_activation=fp8_input,
        )
        residual = NS(configure_verify_fp8=lambda enabled: setattr(residual, "verify", enabled))
        result.layers = [NS(index=0, block_size=1, attention=attention,
                            attention_residual=residual, mlp=KimiK3DenseMLP())]
        return result

    def initialize_model(self, model):
        with patch.dict(sys.modules, {COLLECTIVE: self.module}):
            self.initialize(model)

    def test_fp8_projection_automatically_initializes_prefill_fusions(self):
        model = self.model()
        self.initialize_model(model)
        collective, = self.created
        self.assertTrue(collective.enable_ag)
        self.assertTrue(collective.enable_rs)
        self.assertFalse(collective.decode_staging)
        self.assertEqual(collective.max_m, 65536)
        self.assertIs(model.layers[0].attention._fp8_collective, collective)
        self.assertFalse(model.layers[0].attention_residual.verify)

    def test_decode_keeps_existing_verify_staging_and_resolved_capacity(self):
        model = self.model(role="DECODE")
        self.initialize_model(model)
        collective, = self.created
        self.assertTrue(collective.decode_staging)
        self.assertEqual(collective.max_m, 192)
        self.assertTrue(model.layers[0].attention_residual.verify)
        self.assertEqual(model.layers[0].mlp.output_transition, collective.reduce_scatter_bf16)

    def test_bf16_and_single_rank_do_not_allocate_fp8_collectives(self):
        bf16, single = self.model(quantized=False), self.model()
        single.tp_size = 1
        for model in (bf16, single):
            self.initialize_model(model)
            self.assertIsNone(model._fp8_collective)
        self.assertEqual(self.created, [])

    def test_native_mtp_never_inherits_target_fp8_collectives(self):
        class Draft(KimiK3Model):
            pass

        model = self.model()
        model.__class__ = Draft
        self.initialize_model(model)
        self.assertIsNone(model._fp8_collective)
        self.assertEqual(self.created, [])

    def test_requested_fp8_rejects_incompatible_projection_scales(self):
        model = self.model()
        model.layers[0].attention.output.scale_ue8m0 = False
        with self.assertRaisesRegex(ValueError, "grouped E4M3"):
            self.initialize_model(model)
        self.assertEqual(self.created, [])

    def test_initialize_is_idempotent(self):
        model = self.model()
        self.initialize_model(model)
        self.initialize_model(model)
        self.assertEqual(len(self.created), 1)

    def test_indexer_hidden_contract_disables_verify_fp8_only_producer(self):
        model = self.model(role="DECODE", fp8_input=False)
        self.initialize_model(model)
        self.assertFalse(model.layers[0].attention_residual.verify)
        self.assertIsNotNone(model._fp8_collective)


class AttnResConsumerTest(unittest.TestCase):
    def test_prefill_fp8_producer_requires_a_matching_consumer(self):
        forward = method("KimiK3DecoderLayer", "forward", {
            "torch": NS(cuda=NS(is_current_stream_capturing=lambda: False)),
            "profile_scope": lambda _: nullcontext(),
        })

        class Hidden:
            is_cuda = True
            shape = (4096, 7168)

        class Residual:
            def __init__(self):
                self.calls = []

            def __call__(self, hidden, *args, **kwargs):
                self.calls.append("bf16")
                return hidden

            def forward_fp8(self, hidden, *args, **kwargs):
                self.calls.append("fp8")
                return hidden

        class Attention:
            _fp8_collective = NS(enable_ag=True, can_run_ag=lambda rows: True)

            def __call__(self, hidden, *args):
                return hidden

        for fp8_input in (False, True):
            with self.subTest(fp8_input=fp8_input):
                attention, residual = Attention(), Residual()
                attention.supports_fp8_activation = fp8_input
                layer = NS(index=0, block_size=1, attention=attention,
                           attention_residual=residual, mlp_residual=Residual(),
                           attention_norm=NS(weight=None, variance_epsilon=1e-6),
                           mlp_norm=NS(weight=None, variance_epsilon=1e-6),
                           apply_mlp=lambda hidden, mlp_input, mask: hidden)
                inputs = NS(is_prefill=True, is_mtp_draft_update=False)
                forward(layer, Hidden(), None, None, None, inputs,
                        NS(is_target_verify=False), None)
                self.assertEqual(residual.calls, ["fp8" if fp8_input else "bf16"])

    def test_dsa_declares_its_bf16_hidden_requirement(self):
        tree = ast.parse(DSA.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KimiK31DSA")
        value = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "supports_fp8_activation" for t in n.targets))
        self.assertIs(ast.literal_eval(value), False)


class WeightOwnershipTest(unittest.TestCase):
    def test_input_gap_keeps_replicated_rows_separate_from_rank_local_gate(self):
        class Matrix:
            def __init__(self, rows, columns):
                self.rows, self.shape = rows, (len(rows), columns)

            def __getitem__(self, index):
                return Matrix(self.rows[index], self.shape[1])

            def new_zeros(self, size):
                return Matrix([0] * size[0], size[1])

        prepare = method("KimiK31LoadAttentionFp8Weight", "_prepare_matrix", {
            "W": NS(mla_fusedqkrope_w="input"),
            "torch": NS(cat=lambda pieces, dim: Matrix(
                sum((piece.rows for piece in pieces), []), pieces[0].shape[1])),
        }, ROOT / "models/kimi_k3/fp8_weight.py")
        for tp in (1, 8):
            with self.subTest(projection_tp=tp):
                gate = 96 // tp * 128
                source = Matrix(list(range(1, 2112 + gate + 1)), 7168)
                wrapper = NS(kernel=NS(name="input"), projection_tp_size=tp,
                             source=NS(config=NS(head_num=96, v_head_dim=128)))
                result = prepare(wrapper, source)
                self.assertEqual(result.shape, (2176 + gate, 7168))
                self.assertEqual(result.rows[:2112], source.rows[:2112])
                self.assertEqual(result.rows[2112:2176], [0] * 64)
                self.assertEqual(result.rows[2176:], source.rows[2112:])

    def test_query_cp_and_decode_keep_their_projection_head_ownership(self):
        class Tensor:
            def __init__(self, shape):
                self.shape = tuple(shape)

            def narrow(self, axis, begin, count):
                shape = list(self.shape)
                if begin + count > shape[axis]:
                    raise AssertionError("shard exceeds the source weight")
                shape[axis] = count
                return Tensor(shape)

            def contiguous(self):
                return self

            def storage_offset(self):
                return 0

            def numel(self):
                return math.prod(self.shape)

            def element_size(self):
                return 1

            def untyped_storage(self):
                return NS(nbytes=lambda: self.numel())

        names = NS(mla_q_b_w="qb", mla_kv_b_w="kvb", attn_o_w="o",
                   linear_attn_out_w="kda_o", mla_fusedqkrope_w="input")
        split = method("KimiK3LoadFp8Weight", "_split", {
            "W": names, "K3W": NS(KDA_INPUT="kda_input"),
        }, ROOT / "models/kimi_k3/fp8_weight.py")
        for query_cp, name, shape, scale_shape, expected in (
            (True, "qb", (18432, 1536), (144, 12), (18432, 1536)),
            (False, "qb", (18432, 1536), (144, 12), (2304, 1536)),
            (True, "o", (7168, 12288), (56, 96), (7168, 12288)),
            (False, "o", (7168, 12288), (56, 96), (7168, 1536)),
        ):
            with self.subTest(query_cp=query_cp, name=name):
                wrapper = NS(kernel=NS(name=name), scale=NS(name="scale"),
                             projection_tp_size=1 if query_cp else 8,
                             projection_tp_rank=0 if query_cp else 7,
                             replicate_q_weights=query_cp, derive_mla=False)
                result = split(wrapper, {name: Tensor(shape), "scale": Tensor(scale_shape)},
                               NS(tp_size=8, tp_rank=7))
                self.assertEqual(result[name].shape, expected)


class CacheDispatchTest(unittest.TestCase):
    def test_decode_schedules_retain_capture_buckets_and_reject_replay_allocation(self):
        calls = []

        def metadata(**kwargs):
            calls.append(kwargs)
            return object(), None

        module_name = "rtp_llm.models_py.utils.flash_mla_legacy"
        module = ModuleType(module_name)
        module.get_mla_metadata = metadata
        plan = method("PageRRSparseMlaOp", "plan", {}, ROOT / (
            "models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_page_rr_impl.py"
        ))
        op = NS(workspace_owner=NS(sfa_use_fp8=True), kernel_heads=128, top_k=2048,
                _prefill=object(), _decode=object(), _fp8_schedules={})

        def prepare(rows, replay=False):
            plan(op, NS(is_prefill=False, forbid_realloc=replay, kernel_page_size=128,
                        positions_d=NS(numel=lambda: rows)), None)

        with patch.dict(sys.modules, {module_name: module}):
            prepare(4)
            first = op._fp8_metadata
            prepare(8)
            self.assertIsNot(op._fp8_metadata, first)
            prepare(4, replay=True)
            self.assertIs(op._fp8_metadata, first)
            with self.assertRaisesRegex(ValueError, "before graph replay"):
                prepare(16, replay=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(op._fp8_schedules), 2)

    def test_power_of_two_cache_scales_are_an_internal_writer_policy(self):
        calls = []
        # Supply annotation names without importing the CUDA extension.
        from typing import Optional

        write = method("MlaKVCacheWriteOp", "forward", {
            "torch": NS(Tensor=object), "Optional": Optional, "LayerKVCache": object,
            "rtp_llm_ops": NS(SparseMlaParams=object),
            "compute_ops": NS(concat_and_cache_mla=lambda *args, **kwargs: calls.append(kwargs)),
        }, ROOT / (
            "models_py/modules/factory/attention/cuda_mla_impl/mla_kv_cache_write_op.py"
        ))
        for enabled in (False, True):
            write(NS(kv_cache_type="fp8_ds_mla", scale=object(), use_ue8m0=enabled),
                  object(), object(), NS(kv_cache_base=object()), NS(slot_mapping=object()))
        self.assertEqual(calls, [{}, {"use_ue8m0": True}])

    def test_fp8_decode_preserves_bf16_query_output_and_natural_lse_shape(self):
        class Tensor:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = tuple(shape), dtype

            def unsqueeze(self, axis):
                shape = list(self.shape)
                shape.insert(axis, 1)
                return Tensor(shape, self.dtype)

            def squeeze(self, axis):
                shape = list(self.shape)
                if shape[axis] != 1:
                    raise AssertionError("squeeze must remove the query axis")
                del shape[axis]
                return Tensor(shape, self.dtype)

            def view(self, *shape):
                if shape == ("uint8",):
                    return Tensor(self.shape, "uint8")
                shape = list(shape)
                if -1 in shape:
                    i = shape.index(-1)
                    shape[i] = math.prod(self.shape) // -math.prod(shape)
                return Tensor(shape, self.dtype)

        calls = []

        def decode(**kwargs):
            calls.append(kwargs)
            return Tensor((4, 1, 128, 512), "bf16"), Tensor((4, 128, 1), "fp32")

        module_name = "rtp_llm.models_py.utils.flash_mla_legacy"
        module = ModuleType(module_name)
        module.flash_mla_with_kvcache = decode
        attend = method("PageRRSparseMlaOp", "_attention", {
            "torch": NS(uint8="uint8"),
        }, ROOT / "models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_page_rr_impl.py")
        op = NS(workspace_owner=NS(sfa_use_fp8=True),
                params=NS(is_prefill=False, kernel_page_size=128, block_table=object()),
                kernel_heads=128, kv_lora_rank=512, scale=192 ** -.5,
                _fp8_metadata=object(), _fp8_num_splits=None)
        with patch.dict(sys.modules, {module_name: module}):
            output, lse = attend(op, Tensor((4, 128, 576), "bf16"),
                                 Tensor((256, 1, 656), "fp8"), Tensor((4, 2048), "int32"))
        self.assertEqual(output.shape, (4, 128, 512))
        self.assertEqual(output.dtype, "bf16")
        self.assertEqual(lse.shape, (4, 128))
        self.assertEqual(calls[0]["q"].dtype, "bf16")
        self.assertEqual(calls[0]["k_cache"].shape, (2, 128, 1, 656))
        self.assertEqual(calls[0]["k_cache"].dtype, "uint8")
        self.assertTrue(calls[0]["is_fp8_kvcache"])

    def test_k31_cache_storage_and_core_precision_are_independent(self):
        class Base:
            def init_precision_config(self, cache, act_type):
                self.attention_projection_quant_config = self.projection

        path = ROOT / "models/kimi_k3/kimi_k31.py"
        original = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.ClassDef) and n.name == "KimiK31ModelConfig")
        body = [n for n in original.body if isinstance(n, ast.FunctionDef)
                and n.name in ("init_precision_config", "_bind_indexer_precision")]
        cls = ast.ClassDef(name="Config", bases=[ast.Name(id="Base", ctx=ast.Load())],
                           keywords=[], body=body, decorator_list=[])
        namespace = {"Base": Base, "KvCacheDataType": NS(FP8="fp8", BASE="base"),
                     "Fp8BlockWiseQuantConfig": lambda: "indexer_fp8", "logging": logging}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     str(path), "exec"), namespace)
        for projections in (None, object()):
            for fp8_cache in (False, True):
                cfg = namespace["Config"]()
                cfg.projection = projections
                cfg.is_mtp, cfg.model_type = False, "kimi_k31"
                cfg.k3_runtime_config = NS(dsa=NS(indexer_use_fp8=True))
                cfg.attn_config = NS(mla_fp8_compute=False)
                cfg.init_precision_config(NS(fp8_kv_cache=fp8_cache), "bf16")
                self.assertIs(cfg.attention_projection_quant_config, projections)
                self.assertEqual(cfg.attn_config.kv_cache_dtype, "fp8" if fp8_cache else "base")
                self.assertFalse(cfg.attn_config.mla_fp8_compute)
                self.assertEqual(cfg.indexer_projection_quant_config, "indexer_fp8")
        cfg.attn_config.mla_fp8_compute = True
        with self.assertRaisesRegex(ValueError, "BF16 SFA compute"):
            cfg._bind_indexer_precision()


class DsaFusionTest(unittest.TestCase):
    def test_gate_quant_reuse_preserves_local_query_and_tp_output_layouts(self):
        class Tensor:
            def __init__(self, rows, width, dtype="bf16", masked=False):
                self.shape, self.dtype, self.masked = (rows, width), dtype, masked

        class Mask:
            def __getitem__(self, key):
                return self

        class Projected:
            def __init__(self, rows, gate_width):
                self.rows, self.gate_width = rows, gate_width

            def split(self, sizes, dim):
                if sizes != [2112, 64, self.gate_width]:
                    raise AssertionError("Q/KV padding must be discarded before the gate")
                return (Tensor(self.rows, 2112), Tensor(self.rows, 64),
                        Tensor(self.rows, self.gate_width))

        for local_query in (False, True):
            with self.subTest(local_query=local_query):
                calls = []
                scales = object()

                def quantize(output, gate):
                    self.assertEqual(output.masked, not local_query)
                    self.assertEqual(gate.masked, not local_query)
                    return Tensor(*output.shape, dtype="fp8"), scales

                def gather(hidden, group):
                    calls.append("gather")
                    return Tensor(hidden.shape[0] * 8, hidden.shape[1])

                def project(values, supplied_scales):
                    self.assertIs(supplied_scales, scales)
                    self.assertEqual(values.dtype, "fp8")
                    calls.append("project")
                    return Tensor(values.shape[0], 7168)

                def project_rs(values, supplied_scales, projection):
                    self.assertIs(supplied_scales, scales)
                    calls.append("project_rs")
                    return Tensor(values.shape[0] // 8, 7168)

                module_name = "rtp_llm.models_py.triton_kernels.common.activation"
                module = ModuleType(module_name)
                module.sigmoid_mul_per_token_group_quant_fp8 = quantize
                forward = method("KimiK31DSA", "forward", {
                    "torch": NS(where=lambda mask, value, zero: Tensor(
                        *value.shape, dtype=value.dtype, masked=True)),
                    "all_gather": gather,
                    "Group": NS(TP="TP"),
                    "profile_scope": lambda _: nullcontext(),
                }, DSA)
                heads = 96 if local_query else 12

                def attend(qkv, fmha, cache, hidden):
                    self.assertEqual(hidden.dtype, "bf16")
                    return Tensor(qkv.shape[0], heads * 128)

                layer = NS(prefill_query_cp=local_query, tp_size=8,
                           _bf16_collective=object(),
                           _mtp_bf16_collectives=None, q_rank=1536, kv_rank=512,
                           suffix_dim=64, heads=heads, v_dim=128, _input_padding=64,
                           _fp8_output_gate=True,
                           input=lambda hidden: Projected(hidden.shape[0], heads * 128),
                           output=NS(forward_quantized=project), _attend=attend,
                           _fp8_collective=NS(eligible_rs=lambda *args: True,
                                              gemm_reduce_scatter=project_rs))
                metadata = NS(query_layout=NS(all_valid=True) if local_query else None)
                owner = NS(release_eager_buffers=lambda: calls.append("release"))
                with patch.dict(sys.modules, {module_name: module}):
                    output = forward(layer, Tensor(4, 7168),
                                     {"full": NS(workspace_owner=owner)}, None,
                                     NS(valid_token_mask=Mask()), metadata)
                self.assertEqual(output.shape, (4, 7168))
                self.assertEqual(output.dtype, "bf16")
                self.assertEqual(calls, ["project", "release"] if local_query
                                 else ["gather", "project_rs", "release"])


if __name__ == "__main__":
    unittest.main()
