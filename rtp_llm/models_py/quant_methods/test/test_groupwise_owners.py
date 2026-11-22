"""Real tiny CPU scratch/FST loads, not GPU/multicard/serving/golden tests."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import AWQConfig, GPTQConfig
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.groupwise_checkpoint import (
    GroupwiseCheckpointLayout,
    PackedGroupwiseWeights,
)
from rtp_llm.models_py.quant_methods.test.staged_checkpoint_fixture import (
    save_with_late_shard,
)
from rtp_llm.models_py.quant_methods.test.test_groupwise_checkpoint import (
    checkpoint,
    reconstruct,
)
from rtp_llm.models_py.registry import register_model


@register_model("_groupwise_dense_cpu_owner")
class DenseOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        common = dict(
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            quant_config=load.quant_config,
            params_dtype=torch.float16,
            prefix="proj",
            bias=config.bias,
        )
        if config.kind == "qkv":
            self.proj = QKVParallelLinear(64, 4, 1, 16, **common)
        elif config.kind == "merged":
            self.proj = MergedColumnParallelLinear(
                64, 128, shard_names=["gate_proj", "up_proj"], **common
            )
        else:
            cls = RowParallelLinear if config.kind == "row" else ColumnParallelLinear
            self.proj = cls(64, 64, **common)


@register_model("_groupwise_moe_cpu_owner")
class ExpertOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.experts = BaseMoEExperts(
            num_experts=4,
            top_k=1,
            hidden_size=64,
            moe_intermediate_size=64,
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            ep_size=load.ep_size,
            ep_rank=load.ep_rank,
            params_dtype=torch.float16,
            model_config=config,
            parallelism_config=SimpleNamespace(
                dp_size=1,
                dp_rank=0,
                world_size=load.tp_size,
                local_world_size=load.tp_size,
            ),
            moe_config=None,
            quant_config=load.quant_config,
            layer_idx=0,
            prefix="experts",
        )


def source_config(method, bits):
    return (AWQConfig if method == "awq" else GPTQConfig)(
        bits=bits, group_size=16, is_quanted=True
    )


def material(method, bits, k, n, offset=0):
    layout = GroupwiseCheckpointLayout(method, bits, 16, k, n)
    values, dense = checkpoint(layout)
    # Distinguish experts/projections without touching packed integer semantics.
    values["scales"] *= offset + 1
    # The legacy signed-weight + FP16 offset ABI has two rounding steps;
    # evaluate that exact reference after changing scales, not dense*factor.
    return values, reconstruct(
        PackedGroupwiseWeights.from_tensors(
            **values, method=method, bits=bits, group_size=16
        )
    )


def distributed_load(rank, world, directory, config, load, selected):
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + directory + "/rendezvous",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=45),
    )
    try:
        load = replace(
            load,
            tp_rank=rank,
            ep_rank=rank if load.ep_size > 1 else 0,
            attn_tp_rank=rank,
            ffn_tp_rank=rank,
            lm_head_tp_rank=rank,
        )
        loader = NewModelLoader(config, load, model_path=directory)
        owner = loader.load()
        assert loader.loaded_method.value == "fastsafetensors"
        if rank == selected:
            torch.save(owner, directory + "/loaded_owner.pt")
    finally:
        torch.distributed.destroy_process_group()


def decode(layer):
    metadata = layer._groupwise_layout
    return reconstruct(
        PackedGroupwiseWeights.from_tensors(
            layer.qweight,
            layer.qzeros,
            layer.scales,
            method=metadata["method"],
            bits=metadata["bits"],
            group_size=metadata["group_size"],
        )
    )


class GroupwiseOwnerTest(unittest.TestCase):
    def load(
        self,
        values,
        config,
        method,
        bits,
        backend,
        tp=1,
        rank=0,
        ep=1,
        ep_rank=0,
        file_format="safetensors",
        late=None,
    ):
        source = source_config(method, bits)
        config.quant_config = source
        config.activation_type = "SiGLU"
        if config.model_type == "_groupwise_moe_cpu_owner":
            config.expert_num = 4
            config.moe_k = 1
        with tempfile.TemporaryDirectory() as directory:
            if late is not None:
                save_with_late_shard(directory, values, late)
            elif file_format == "bin":
                torch.save(values, Path(directory) / "pytorch_model.bin")
            else:
                save_file(values, str(Path(directory) / "model.safetensors"))
            load = NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.float16,
                load_method=backend,
                tp_size=tp,
                tp_rank=rank,
                ep_size=ep,
                ep_rank=ep_rank,
                quant_config=QuantizationConfig(method, source),
            )
            if backend == "fastsafetensors" and tp > 1:
                torch.multiprocessing.start_processes(
                    distributed_load,
                    args=(tp, directory, config, load, rank),
                    nprocs=tp,
                    join=True,
                    start_method="fork",
                )
                return torch.load(directory + "/loaded_owner.pt", weights_only=False)
            loader = NewModelLoader(config, load, model_path=directory)
            owner = loader.load()
            self.assertEqual(loader.loaded_method.value, backend)
            self.assertTrue(all(p.device.type == "cpu" for p in owner.parameters()))
            return owner

    def test_qwen2_gptq_checkpoint_biases_use_production_owner_and_pp_stage(self):
        from rtp_llm.ops import ParallelismConfig

        # Use the production Qwen2 decoder: Q/K/V biases are real parameters;
        # o_proj/down_proj/gate_proj/up_proj have no bias in the old or new model.
        source = source_config("gptq", 4)
        config = SimpleNamespace(
            model_type="qwen_2",
            num_layers=2,
            is_mtp=False,
            vocab_size=64,
            hidden_size=64,
            inter_size=64,
            attn_config=SimpleNamespace(head_num=4, kv_head_num=1, size_per_head=16),
            layernorm_eps=1e-6,
            enable_fp32_lm_head=False,
            tie_word_embeddings=True,
            capture_aux_hidden_layer_ids=[],
            hidden_state_capture_layer_ids=[],
            quant_config=source,
        )
        values = {
            "model.embed_tokens.weight": torch.arange(4096).reshape(64, 64).half(),
            "model.norm.weight": torch.ones(64, dtype=torch.float16),
        }
        expected_biases = {}
        for layer in range(2):
            prefix = f"model.layers.{layer}."
            for norm in ("input_layernorm", "post_attention_layernorm"):
                values[prefix + norm + ".weight"] = torch.ones(64, dtype=torch.float16)
            biases = []
            for index, (name, width) in enumerate(
                (
                    ("self_attn.q_proj", 64),
                    ("self_attn.k_proj", 16),
                    ("self_attn.v_proj", 16),
                    ("self_attn.o_proj", 64),
                    ("mlp.gate_proj", 64),
                    ("mlp.up_proj", 64),
                    ("mlp.down_proj", 64),
                )
            ):
                packed, _ = material("gptq", 4, 64, width, index)
                values.update({prefix + name + "." + k: v for k, v in packed.items()})
                bias = (
                    torch.arange(width).half() + layer * 10 + index
                    if index < 3
                    else torch.zeros(width).half()
                )
                values[prefix + name + ".bias"] = bias
                if index < 3:
                    biases.append(bias)
            expected_biases[layer] = torch.cat(biases)
        for backend in ("scratch", "fastsafetensors"):
            # Scratch checks each real PP ownership interval; the separate
            # leaf regression below exercises real FST TP groups with Gloo.
            for stage in (0, 1) if backend == "scratch" else (0,):
                with self.subTest(
                    backend=backend, stage=stage
                ), tempfile.TemporaryDirectory() as root:
                    save_file(values, str(Path(root) / "model.safetensors"))
                    pc = ParallelismConfig()
                    pc.tp_size = pc.ep_size = pc.dp_size = 1
                    pc.tp_rank = pc.ep_rank = pc.dp_rank = 0
                    pc.pp_size, pc.pp_rank = (2 if backend == "scratch" else 1), stage
                    pc.world_size = pc.local_world_size = pc.pp_size
                    pc.world_rank = pc.local_rank = stage
                    pc.pp_stage_layer_counts = [1, 1] if pc.pp_size == 2 else []
                    load = NewLoaderConfig(
                        device="cpu",
                        compute_dtype=torch.float16,
                        load_method=backend,
                        parallelism_config=pc,
                        quant_config=QuantizationConfig("gptq", source),
                    )
                    loader = NewModelLoader(config, load, model_path=root)
                    model = loader.load()
                    self.assertEqual(loader.loaded_method.value, backend)
                    owned = (stage,) if pc.pp_size == 2 else (0, 1)
                    if pc.pp_size == 2:
                        self.assertEqual(set(model.layers), {str(i) for i in owned})
                    else:
                        self.assertEqual(len(model.layers), len(owned))
                    for i in owned:
                        layer = (
                            model.layers[str(i)] if pc.pp_size == 2 else model.layers[i]
                        )
                        torch.testing.assert_close(
                            layer.self_attn.qkv_proj.bias,
                            expected_biases[i],
                            rtol=0,
                            atol=0,
                        )
                        for projection in (
                            layer.self_attn.o_proj,
                            layer.mlp.gate_up_proj,
                            layer.mlp.down_proj,
                        ):
                            self.assertIsNone(projection.bias)
                    # Missing a semantic QKV bias is still a load-time failure.
                    broken = dict(values)
                    del broken[f"model.layers.{stage}.self_attn.q_proj.bias"]
                    save_file(broken, str(Path(root) / "model.safetensors"))
                    with self.assertRaisesRegex(
                        RuntimeError, "[Mm]issing|not loaded|complete checkpoint"
                    ):
                        NewModelLoader(config, load, model_path=root).load()

    def test_gptq_zero_bias_buffers_for_biasless_modules(self):
        for backend in ("scratch", "fastsafetensors"):
            for kind in ("column", "row", "merged"):
                parts = ("gate_proj", "up_proj") if kind == "merged" else ("",)
                values, dense_parts = {}, []
                for index, part in enumerate(parts):
                    packed, dense = material("gptq", 4, 64, 64, index)
                    prefix = "proj." + (part + "." if part else "")
                    values.update({prefix + k: v for k, v in packed.items()})
                    values[prefix + "bias"] = torch.zeros(64, dtype=torch.float16)
                    dense_parts.append(dense)
                for rank in range(2):
                    with self.subTest(backend=backend, kind=kind, rank=rank):
                        owner = self.load(
                            values,
                            SimpleNamespace(
                                model_type="_groupwise_dense_cpu_owner",
                                kind=kind,
                                bias=False,
                            ),
                            "gptq",
                            4,
                            backend,
                            tp=2,
                            rank=rank,
                        )
                        self.assertIsNone(owner.proj.bias)
                        self.assertNotIn("proj.bias", dict(owner.named_parameters()))
                        expected = torch.cat(
                            [
                                dense.chunk(2, 0 if kind == "row" else 1)[rank]
                                for dense in dense_parts
                            ],
                            dim=1,
                        )
                        torch.testing.assert_close(
                            decode(owner.proj), expected, rtol=0, atol=0
                        )

    def test_gptq_zero_bias_buffer_dtype_checked_before_zero_acceptance(self):
        packed, _ = material("gptq", 4, 64, 64)
        values = {"proj." + name: tensor for name, tensor in packed.items()}
        config = SimpleNamespace(
            model_type="_groupwise_dense_cpu_owner", kind="row", bias=False
        )
        for backend in ("scratch", "fastsafetensors"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(backend=backend, dtype=dtype):
                    owner = self.load(
                        dict(values, **{"proj.bias": torch.zeros(64, dtype=dtype)}),
                        config,
                        "gptq",
                        4,
                        backend,
                    )
                    self.assertIsNone(owner.proj.bias)
            for dtype in (
                torch.bool,
                torch.uint8,
                torch.int32,
                torch.int64,
                torch.float64,
            ):
                with self.subTest(backend=backend, dtype=dtype):
                    bias = torch.zeros(64, dtype=dtype)
                    source = source_config("gptq", 4)
                    owner = DenseOwner(
                        config,
                        NewLoaderConfig(
                            device="cpu",
                            quant_config=QuantizationConfig("gptq", source),
                        ),
                    )
                    # Independently prove the owner checks dtype before it
                    # accepts zero values, without the FST coordinator.
                    with self.assertRaisesRegex(
                        TypeError, "Grouped scales/bias must be floating point"
                    ):
                        owner.proj.load_weights({"bias": bias})
                    with self.assertRaisesRegex(
                        TypeError, "Grouped scales/bias must be floating point"
                    ):
                        self.load(
                            dict(values, **{"proj.bias": bias}),
                            config,
                            "gptq",
                            4,
                            backend,
                        )

    def test_staged_fst_late_dense_metadata(self):
        cfg = SimpleNamespace(
            model_type="_groupwise_dense_cpu_owner", kind="merged", bias=False
        )
        identity = torch.arange(64, dtype=torch.int32) // 16
        for method in ("gptq", "awq"):
            required = {}
            dense = []
            for index, part in enumerate(("gate_proj", "up_proj")):
                packed, projection = material(method, 4, 64, 64, index)
                required.update(
                    {f"proj.{part}.{key}": value for key, value in packed.items()}
                )
                dense.append(projection)

            def load_late(late):
                return self.load(required, cfg, method, 4, "fastsafetensors", late=late)

            late = {"proj.g_idx": identity}
            if method == "gptq":
                late["proj.bias"] = torch.zeros(128, dtype=torch.float32)
            owner = load_late(late)
            self.assertIsNone(owner.proj.bias)
            torch.testing.assert_close(
                decode(owner.proj), torch.cat(dense, dim=1), rtol=0, atol=0
            )
            # Fused and per-projection aliases must not validate a logical key twice.
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                owner.proj.load_weights({"gate_proj.g_idx": identity})
            if method == "gptq":
                with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                    owner.proj.load_weights({"gate_proj.bias": torch.zeros(64)})
                with self.assertRaisesRegex(ValueError, "all zero"):
                    load_late({"proj.bias": torch.ones(128)})
            with self.assertRaisesRegex(ValueError, "activation-order"):
                load_late({"proj.up_proj.g_idx": identity.flip(0)})
            # A fused alias that would write packed parameters is still forbidden.
            with self.assertRaisesRegex(RuntimeError, "after staged postprocessing"):
                load_late({"proj.scales": torch.cat([packed["scales"]] * 2, dim=1)})

    def test_staged_fst_late_expert_g_idx(self):
        cfg = SimpleNamespace(model_type="_groupwise_moe_cpu_owner")
        packed, _ = material("gptq", 4, 64, 64)
        required = {
            f"experts.{expert}.{proj}.{key}": value.clone()
            for expert in range(4)
            for proj in ("gate_proj", "up_proj", "down_proj")
            for key, value in packed.items()
        }
        key = "experts.3.down_proj.g_idx"
        identity = torch.arange(64, dtype=torch.int32) // 16
        def load_index(index):
            return self.load(
                required, cfg, "gptq", 4, "fastsafetensors", late={key: index}
            )

        owner = load_index(identity)
        torch.testing.assert_close(owner.experts.w2[3], packed["qweight"])
        with self.assertRaisesRegex(ValueError, "activation-order"):
            load_index(identity.flip(0))

    def test_gptq_biasless_buffers_are_strict_not_silently_discarded(self):
        packed, _ = material("gptq", 4, 64, 64)
        base = {"proj." + k: v for k, v in packed.items()}
        config = SimpleNamespace(
            model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
        )
        bad_values = {
            "nonzero": torch.cat((torch.zeros(63), torch.ones(1))).half(),
            "nan": torch.full((64,), float("nan"), dtype=torch.float16),
            "shape": torch.zeros(63, dtype=torch.float16),
            "dtype": torch.zeros(64, dtype=torch.int32),
        }
        for backend in ("scratch", "fastsafetensors"):
            for case, bias in bad_values.items():
                with self.subTest(backend=backend, case=case):
                    with self.assertRaises((ValueError, TypeError)):
                        self.load(
                            dict(base, **{"proj.bias": bias}),
                            config,
                            "gptq",
                            4,
                            backend,
                        )
        # A nonzero value outside TP rank 0's output rows must still fail.
        with self.assertRaisesRegex(ValueError, "must be all zero"):
            self.load(
                dict(base, **{"proj.bias": bad_values["nonzero"]}),
                config,
                "gptq",
                4,
                "scratch",
                tp=2,
                rank=0,
            )
        # The compatibility rule is GPTQ-specific; AWQ remains strict.
        awq, _ = material("awq", 4, 64, 64)
        awq = {"proj." + k: v for k, v in awq.items()}
        awq["proj.bias"] = torch.zeros(64, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "Unexpected grouped bias"):
            self.load(awq, config, "awq", 4, "scratch")
        from rtp_llm.models_py.read_spec import make_sharded_read_spec

        source = source_config("gptq", 4)
        owner = DenseOwner(
            config,
            NewLoaderConfig(
                device="cpu",
                tp_size=2,
                tp_rank=0,
                quant_config=QuantizationConfig("gptq", source),
            ),
        )
        spec = owner.proj.checkpoint_read_spec("bias", (64,))
        self.assertIsNone(spec.slices)
        self.assertTrue(spec.load)
        partial = make_sharded_read_spec("bias", (64,), 0, 2, 0).materialize(
            torch.zeros(64).half()
        )
        with self.assertRaisesRegex(ValueError, "another projection/rank"):
            owner.proj.load_weights({"bias": partial})

    def test_dense_real_scratch_and_fst_all_formats_tp_and_gqa(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                for backend in ("scratch", "fastsafetensors"):
                    for kind in ("column", "row", "merged", "qkv"):
                        tp = 4 if kind == "qkv" else 2
                        parts = (
                            [("q_proj", 64), ("k_proj", 16), ("v_proj", 16)]
                            if kind == "qkv"
                            else (
                                [("gate_proj", 64), ("up_proj", 64)]
                                if kind == "merged"
                                else [("", 64)]
                            )
                        )
                        values, full = {}, []
                        for index, (part, width) in enumerate(parts):
                            packed, dense = material(method, bits, 64, width, index)
                            prefix = "proj." + (part + "." if part else "")
                            values.update(
                                {prefix + key: value for key, value in packed.items()}
                            )
                            values[prefix + "bias"] = (
                                torch.arange(width).half() + index * 100
                            )
                            full.append(dense)
                        for rank in range(tp):
                            with self.subTest(
                                method=method,
                                bits=bits,
                                backend=backend,
                                kind=kind,
                                rank=rank,
                            ):
                                owner = self.load(
                                    values,
                                    SimpleNamespace(
                                        model_type="_groupwise_dense_cpu_owner",
                                        kind=kind,
                                        bias=True,
                                    ),
                                    method,
                                    bits,
                                    backend,
                                    tp,
                                    rank,
                                )
                                expected = []
                                biases = []
                                for index, ((part, width), dense) in enumerate(
                                    zip(parts, full)
                                ):
                                    expected.append(
                                        dense.chunk(tp, 0)[rank]
                                        if kind == "row"
                                        else (
                                            dense
                                            if kind == "qkv" and part != "q_proj"
                                            else dense.chunk(tp, 1)[rank]
                                        )
                                    )
                                    bias = torch.arange(width).half() + index * 100
                                    biases.append(
                                        bias
                                        if kind == "row"
                                        or (kind == "qkv" and part != "q_proj")
                                        else bias.chunk(tp)[rank]
                                    )
                                torch.testing.assert_close(
                                    decode(owner.proj),
                                    torch.cat(expected, 1),
                                    rtol=0,
                                    atol=0,
                                )
                                torch.testing.assert_close(
                                    owner.proj.bias, torch.cat(biases), rtol=0, atol=0
                                )
                                self.assertTrue(owner.proj.checkpoint_load_complete())

    def test_dense_fused_qkv_and_gate_up_sources(self):
        for kind, widths in (("qkv", (64, 16, 16)), ("merged", (64, 64))):
            for method in ("awq", "gptq"):
                packs, dense = zip(
                    *(material(method, 4, 64, n, i) for i, n in enumerate(widths))
                )
                values = {
                    "proj." + key: torch.cat([p[key] for p in packs], dim=1)
                    for key in packs[0]
                }
                for backend in ("scratch", "fastsafetensors"):
                    owner = self.load(
                        values,
                        SimpleNamespace(
                            model_type="_groupwise_dense_cpu_owner",
                            kind=kind,
                            bias=False,
                        ),
                        method,
                        4,
                        backend,
                        2,
                        1,
                    )
                    expected = [
                        v if kind == "qkv" and i else v.chunk(2, 1)[1]
                        for i, v in enumerate(dense)
                    ]
                    torch.testing.assert_close(
                        decode(owner.proj), torch.cat(expected, 1), rtol=0, atol=0
                    )

    def test_grouped_bin_real_scratch(self):
        packed, dense = material("gptq", 4, 64, 64)
        owner = self.load(
            {"proj." + k: v for k, v in packed.items()},
            SimpleNamespace(
                model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
            ),
            "gptq",
            4,
            "scratch",
            file_format="bin",
        )
        torch.testing.assert_close(decode(owner.proj), dense, rtol=0, atol=0)

    def test_moe_real_scratch_and_fst_tp_ep(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                values, full = {}, {}
                for expert in range(4):
                    for index, projection in enumerate(
                        ("gate_proj", "up_proj", "down_proj")
                    ):
                        packed, dense = material(
                            method, bits, 64, 64, expert * 3 + index
                        )
                        values.update(
                            {
                                f"experts.{expert}.{projection}.{key}": value
                                for key, value in packed.items()
                            }
                        )
                        full[expert, projection] = dense
                for backend in ("scratch", "fastsafetensors"):
                    for tp, ep in ((1, 1), (2, 1), (2, 2)):
                        for rank in range(tp):
                            with self.subTest(
                                method=method,
                                bits=bits,
                                backend=backend,
                                tp=tp,
                                ep=ep,
                                rank=rank,
                            ):
                                owner = self.load(
                                    values,
                                    SimpleNamespace(
                                        model_type="_groupwise_moe_cpu_owner"
                                    ),
                                    method,
                                    bits,
                                    backend,
                                    tp,
                                    rank,
                                    ep,
                                    rank if ep > 1 else 0,
                                )
                                layer = owner.experts
                                self.assertTrue(layer.checkpoint_load_complete())
                                for expert in range(4):
                                    for local_id in layer._remap_expert_ids(expert):
                                        for stem in ("w13", "w2"):
                                            decoded = reconstruct(
                                                PackedGroupwiseWeights.from_tensors(
                                                    getattr(layer, stem)[local_id],
                                                    getattr(layer, stem + "_qzeros")[
                                                        local_id
                                                    ],
                                                    getattr(layer, stem + "_scales")[
                                                        local_id
                                                    ],
                                                    method=method,
                                                    bits=bits,
                                                    group_size=16,
                                                )
                                            )
                                            expected = (
                                                full[expert, "down_proj"]
                                                if stem == "w2"
                                                else torch.cat(
                                                    [
                                                        full[expert, "up_proj"],
                                                        full[expert, "gate_proj"],
                                                    ],
                                                    1,
                                                )
                                            )
                                            if ep == 1 and tp > 1:
                                                expected = (
                                                    full[expert, "down_proj"].chunk(
                                                        tp, 0
                                                    )[rank]
                                                    if stem == "w2"
                                                    else torch.cat(
                                                        [
                                                            full[expert, p].chunk(
                                                                tp, 1
                                                            )[rank]
                                                            for p in (
                                                                "up_proj",
                                                                "gate_proj",
                                                            )
                                                        ],
                                                        1,
                                                    )
                                                )
                                            torch.testing.assert_close(
                                                decoded, expected, rtol=0, atol=0
                                            )

    def test_missing_aux_and_activation_order_rejected(self):
        packed, _ = material("gptq", 4, 64, 64)
        cfg = SimpleNamespace(
            model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
        )
        for backend in ("scratch", "fastsafetensors"):
            missing = {
                "proj." + key: value for key, value in packed.items() if key != "qzeros"
            }
            with self.assertRaisesRegex(
                (RuntimeError, ValueError), "missing|not loaded|complete checkpoint"
            ):
                self.load(missing, cfg, "gptq", 4, backend)
            values = {"proj." + key: value for key, value in packed.items()}
            values["proj.g_idx"] = torch.arange(64, dtype=torch.int32).flip(0) // 16
            with self.assertRaisesRegex(ValueError, "activation-order"):
                self.load(values, cfg, "gptq", 4, backend)

    def test_gptq_max_stored_zero_remains_effective_sixteen(self):
        packed, _ = material("gptq", 4, 64, 64)
        packed["qzeros"].fill_(-1)
        for backend in ("scratch", "fastsafetensors"):
            owner = self.load(
                {"proj." + key: value for key, value in packed.items()},
                SimpleNamespace(
                    model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
                ),
                "gptq",
                4,
                backend,
            )
            self.assertTrue(torch.equal(owner.proj.qzeros, packed["qzeros"]))
            reference = (torch.arange(64 * 64).reshape(64, 64) % 16 - 16) * packed[
                "scales"
            ].repeat_interleave(16, 0)
            torch.testing.assert_close(decode(owner.proj), reference, rtol=0, atol=0)

    def test_packed_dtype_and_duplicate_not_silently_converted(self):
        packed, _ = material("awq", 4, 64, 64)
        cfg = SimpleNamespace(
            model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
        )
        for backend in ("scratch", "fastsafetensors"):
            for parameter in ("qweight", "qzeros", "scales"):
                values = {"proj." + key: value for key, value in packed.items()}
                values["proj." + parameter] = packed[parameter].to(torch.int64)
                with self.assertRaisesRegex(TypeError, "int32|floating point"):
                    self.load(values, cfg, "awq", 4, backend)
            owner = self.load(
                {"proj." + key: value for key, value in packed.items()},
                cfg,
                "awq",
                4,
                backend,
            )
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                owner.proj.load_weights({"qweight": packed["qweight"]})
            with self.assertRaisesRegex(RuntimeError, "not a native inference backend"):
                owner.proj(torch.zeros(1, 64, dtype=torch.float16))

    def test_wrong_rank_readspec_rejected_before_copy(self):
        source = source_config("gptq", 4)
        layer = ColumnParallelLinear(
            64,
            64,
            tp_size=2,
            tp_rank=1,
            quant_config=QuantizationConfig("gptq", source),
            params_dtype=torch.float16,
        )
        packed, _ = material("gptq", 4, 64, 64)
        layout = GroupwiseCheckpointLayout("gptq", 4, 16, 64, 64)
        wrong = layout.read_spec("qweight", tuple(packed["qweight"].shape), 1, 2, 0)
        with self.assertRaisesRegex(ValueError, "another projection/rank"):
            layer.load_weights({"qweight": wrong.materialize(packed["qweight"])})


if __name__ == "__main__":
    unittest.main()
