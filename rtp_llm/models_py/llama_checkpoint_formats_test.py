"""Rank-local FT and consolidated .pth archives must not require HF config.json."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import QuantizationConfig as CheckpointQuantization
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


def values(*shape):
    return torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)


def load(root, tp=1, rank=0, kv_heads=1, dtype=torch.float32, task_type=None,
         *, online_quant=None, device="cpu", force_cpu=False, startup_lora=None,
         load_method="scratch"):
    config = SimpleNamespace(
        ckpt_path=str(root), model_type="llama", num_layers=1, vocab_size=8,
        hidden_size=8, inter_size=8, layernorm_eps=1e-6, is_mtp=False,
        enable_fp32_lm_head=False, tie_word_embeddings=False,
        attn_config=SimpleNamespace(head_num=2, kv_head_num=kv_heads, size_per_head=4),
        capture_aux_hidden_layer_ids=[], hidden_state_capture_layer_ids=[],
    )
    if task_type is not None:
        config.task_type = task_type
    parallel = SimpleNamespace(
        tp_size=tp, tp_rank=rank, dp_size=1, dp_rank=0, ep_size=1, ep_rank=0,
        pp_size=1, pp_rank=0, world_size=tp, world_rank=rank,
        get_attn_tp_size=lambda: tp, get_attn_tp_rank=lambda: rank,
        get_ffn_tp_size=lambda: tp, get_ffn_tp_rank=lambda: rank,
    )
    quant = CheckpointQuantization.load_from_ckpt(str(root))
    options = NewLoaderConfig(
        tp_size=tp, tp_rank=rank, compute_dtype=dtype, device=device,
        force_cpu_load_weights=force_cpu,
        parallelism_config=parallel, quant_config=QuantizationConfig(
            online_quant or (quant.get_method() if quant else "none"), source_config=quant
        ),
        load_method=load_method,
    )
    return NewModelLoader(config, options, model_path=str(root)).load(startup_lora=startup_lora)


def write_q_adapter(root, gain):
    from safetensors.torch import save_file

    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}))
    stem = "base_model.model.model.layers.0.self_attn.q_proj"
    a = torch.ones(1, 8)
    b = (values(8, 1) - 4) * gain
    save_file({stem + ".lora_A.weight": a, stem + ".lora_B.weight": b}, str(adapter / "adapter_model.safetensors"))
    return ("q", str(adapter)), b @ a


def write_meta_source(root):
    params = {"dim": 8, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "norm_eps": 1e-6}
    (root / "params.json").write_text(json.dumps(params))
    tensors = {
        "tok_embeddings.weight": values(8, 8),
        "output.weight": values(8, 8) + 99,
        "norm.weight": values(8) + 1,
        "layers.0.attention_norm.weight": values(8) + 7,
        "layers.0.ffn_norm.weight": values(8) + 9,
    }
    for i, name in enumerate(("wq", "wk", "wv", "wo")):
        tensors[f"layers.0.attention.{name}.weight"] = values(
            4 if name in ("wk", "wv") else 8, 8
        ) + 20 * i
    for i in (1, 2, 3):
        tensors[f"layers.0.feed_forward.w{i}.weight"] = values(8, 8) + i
    for rank in range(2):
        part = {}
        for name, tensor in tensors.items():
            axis = 1 if name.endswith(("wo.weight", "w2.weight", "tok_embeddings.weight")) else 0
            part[name] = tensor if tensor.ndim == 1 else tensor.chunk(2, axis)[rank].clone()
        torch.save(part, root / f"consolidated.{rank:02d}.pth")
    return tensors


class LlamaCheckpointFormatsTest(unittest.TestCase):
    def test_full_depth_static_lora_with_reduced_llama_layer_count(self):
        from safetensors.torch import save_file

        from rtp_llm.config.model_config import apply_layer_num_override
        from rtp_llm.models.llama import Llama
        from rtp_llm.ops import ParallelismConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = dict(
                model_type="mistral",
                architectures=["MistralForCausalLM"],
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                hidden_size=4,
                intermediate_size=8,
                vocab_size=10,
                rms_norm_eps=1e-5,
                max_position_embeddings=32,
                sliding_window=32,
                tie_word_embeddings=False,
            )
            (root / "config.json").write_text(json.dumps(hf))
            config = Llama._create_config(tmp)
            config.model_type = "llama"
            config.max_seq_len = 16
            apply_layer_num_override(config, 1)
            base = {
                "model.embed_tokens.weight": values(10, 4),
                "lm_head.weight": values(10, 4),
                "model.norm.weight": torch.ones(4),
            }
            factors = {}
            for layer in range(2):
                prefix = f"model.layers.{layer}."
                for norm in ("input_layernorm", "post_attention_layernorm"):
                    base[prefix + norm + ".weight"] = torch.ones(4)
                for target, shape in (
                    ("self_attn.q_proj", (4, 4)),
                    ("self_attn.k_proj", (2, 4)),
                    ("self_attn.v_proj", (2, 4)),
                    ("self_attn.o_proj", (4, 4)),
                    ("mlp.gate_proj", (8, 4)),
                    ("mlp.up_proj", (8, 4)),
                    ("mlp.down_proj", (4, 8)),
                ):
                    base[prefix + target + ".weight"] = values(*shape)
                for target in ("q_proj", "o_proj"):
                    peft = "base_model.model." + prefix + "self_attn." + target
                    factors[peft + ".lora_A.weight"] = values(1, 4) + 1
                    factors[peft + ".lora_B.weight"] = values(4, 1) + layer + 1
            save_file(base, str(root / "model.safetensors"))
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 1, "lora_alpha": 2})
            )
            adapter_file = adapter / "adapter_model.bin"
            torch.save(factors, adapter_file)

            def load_adapter(method):
                options = NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.float32,
                    load_method=method,
                    parallelism_config=ParallelismConfig(),
                )
                loader = NewModelLoader(config, options, model_path=tmp)
                model = loader.load(startup_lora=("test", str(adapter)))
                self.assertEqual(loader.loaded_method.value, method)
                return model

            delta = 2 * ((values(4, 1) + 1) @ (values(1, 4) + 1))
            attn = "model.layers.0.self_attn."
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    model = load_adapter(method)
                    self.assertEqual(len(model.layers), 1)
                    qkv = torch.cat([
                        base[attn + "q_proj.weight"] + delta,
                        base[attn + "k_proj.weight"],
                        base[attn + "v_proj.weight"],
                    ])
                    layer = model.layers[0].self_attn
                    torch.testing.assert_close(layer.qkv_proj.weight, qkv)
                    torch.testing.assert_close(
                        layer.o_proj.weight, base[attn + "o_proj.weight"] + delta
                    )

            # An adapter referring beyond the original checkpoint is still invalid.
            torch.save(
                {name.replace("layers.1.", "layers.2."): v for name, v in factors.items()},
                adapter_file,
            )
            with self.assertRaisesRegex(NotImplementedError, r"layers\.2\.self_attn"):
                load_adapter("scratch")

    def test_explicitly_disabled_window_does_not_limit_full_attention(self):
        from rtp_llm.models_py.new_models.qwen2.llama import _check_window

        for family in ("llama", "yi"):
            hf = dict(model_type=family, sliding_window=4, use_sliding_window=False)
            _check_window(hf, 8)
            for change in (
                {"use_sliding_window": True},
                {"use_sliding_window": None},
                {"layer_types": ["full_attention", "sliding_attention"]},
                {"model_type": "mistral"},
            ):
                active = {**hf, **change}
                _check_window(active, 4)
                with self.assertRaisesRegex(ValueError, "sliding_window"):
                    _check_window(active, 8)

    def test_calibrated_decoder_does_not_quantize_routed_experts(self):
        from rtp_llm.models_py.quant_methods.unquantized import UnquantizedFusedMoEMethod

        cfg = QuantizationConfig("pertensor_quant")
        experts = SimpleNamespace(PROJ_NAMES=("gate_proj", "up_proj", "down_proj"))
        self.assertIsInstance(
            cfg.get_moe_quant_method(experts, "model.layers.0.mlp.experts"),
            UnquantizedFusedMoEMethod,
        )

    def test_real_llama_smooth_converter_projection_layout(self):
        from rtp_llm.utils.smooth_quant_convert.llama.convert import (
            split_and_save_weight,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text('{"model_type":"llama"}')
            (root / "smoothquant.ini").touch()
            tensors = {
                "model.embed_tokens.weight": values(8, 8),
                "lm_head.weight": values(8, 8),
                "model.norm.weight": values(8) + 1,
                "model.layers.0.input_layernorm.weight": values(8) + 1,
                "model.layers.0.post_layernorm.weight": values(8) + 1,
            }
            for proj in (
                "attention.query_key_value",
                "attention.dense",
                "mlp.fc",
                "mlp.gate",
                "mlp.proj",
            ):
                width = 24 if proj == "attention.query_key_value" else 8
                weight = values(8, width) / 128 - 0.5
                calibration = {
                    "x": torch.ones(8),
                    "w": weight.abs().amax(0),
                    "y": torch.ones(width),
                }
                opts = {"int8_outputs": "all", "multi_query_mode": False}
                tensors.update(
                    split_and_save_weight(
                        0,
                        root,
                        1,
                        f"layers.0.{proj}.weight",
                        weight.numpy(),
                        calibration,
                        opts,
                    )
                )
                if proj in ("attention.dense", "mlp.proj"):
                    tensors.update(
                        split_and_save_weight(
                            0,
                            root,
                            1,
                            f"layers.0.{proj}.smoother",
                            torch.ones(8).numpy(),
                            None,
                            opts,
                        )
                    )
            torch.save(tensors, root / "llama2.pt")
            model = load(root, kv_heads=2, dtype=torch.float16)
            attention = model.layers[0].self_attn.qkv_proj
            source = "model.layers.0.attention.query_key_value."
            torch.testing.assert_close(
                attention.weight, tensors[source + "weight.int8.col"].reshape(8, 24).T
            )
            torch.testing.assert_close(
                attention.weight_scale.flatten(),
                tensors[source + "scale_w_quant_orig.col"].flatten(),
            )

    def test_calibrated_decoder_int8_legacy_suffixes(self):
        from safetensors.torch import save_file
        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.models.llama import Llama
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig, TaskType
        from rtp_llm.tools.convert.ft_export_job import create_ft_export_job

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(
                json.dumps(
                    dict(
                        model_type="llama",
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        hidden_size=8,
                        intermediate_size=8,
                        vocab_size=8,
                        rms_norm_eps=1e-6,
                    )
                )
            )
            (root / "pertensorquant.ini").touch()
            layer = "model.layers.0."
            tensors = {
                "model.embed_tokens.weight": values(8, 8),
                "lm_head.weight": values(8, 8),
                "model.norm.weight": values(8) + 1,
                layer + "input_layernorm.weight": values(8) + 1,
                layer + "post_attention_layernorm.weight": values(8) + 1,
            }
            for i, (proj, width) in enumerate(
                (
                    ("self_attn.q_proj", 8),
                    ("self_attn.k_proj", 4),
                    ("self_attn.v_proj", 4),
                    ("self_attn.o_proj", 8),
                    ("mlp.gate_proj", 8),
                    ("mlp.up_proj", 8),
                    ("mlp.down_proj", 8),
                )
            ):
                base = layer + proj
                tensors[base + ".qweight"] = (values(8, width) % 11 - 5 + i).to(
                    torch.int8
                )
                tensors[base + ".scales"] = (
                    torch.tensor([0.25 + i / 8])
                    if width == 4
                    else (values(1, width) + 1 + i) / 8
                )
            for proj, step in (
                ("self_attn.qkv", 0.5),
                ("self_attn.o_proj", 0.125),
                ("mlp.gate_proj", 0.75),
                ("mlp.down_proj", 0.875),
            ):
                tensors[layer + proj + "_input_scale"] = torch.tensor([step])
            path = root / "model.safetensors"
            save_file(tensors, path)
            ft = root / "ft"
            config = Llama.create_config(str(root))
            config.data_type, config.task_type, config.lora_infos = (
                "fp32",
                TaskType.LANGUAGE_MODEL,
                {},
            )
            config.init_precision_config(KVCacheConfig(), "fp32")
            for rank in range(2):
                pc = ParallelismConfig()
                pc.tp_size, pc.tp_rank, pc.ep_size = 2, rank, 1
                pc.ffn_tp_size, pc.ffn_tp_rank = 2, rank
                pc.world_size, pc.world_rank, pc.local_world_size = 2, rank, 2
                descriptor = SimpleNamespace(
                    model_config=config,
                    parallelism_config=pc,
                    get_weight_cls=Llama.get_weight_cls,
                    hw_kernel_config=HWKernelConfig(),
                    kv_cache_config=KVCacheConfig(),
                    merge_lora=False,
                    load_method="scratch",
                    force_cpu_load_weights=True,
                    moe_pure_tp_preshard=False,
                    loader_recycle_handles=False,
                    _init_custom_module=lambda: None,
                )
                create_ft_export_job(descriptor).dump_weight_as_ft_style("cpu", str(ft))
            generate_ft_index(str(ft), {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 1})
            (ft / "config.json").write_bytes((root / "config.json").read_bytes())
            (ft / "pertensorquant.ini").touch()
            adapter, delta = write_q_adapter(root, gain=256)
            for rank in range(2):
                model = load(root, 2, rank)
                restored = load(ft, 2, rank)
                for (name, original), (loaded_name, actual) in zip(
                    model.named_parameters(), restored.named_parameters(), strict=True
                ):
                    self.assertEqual(name, loaded_name)
                    torch.testing.assert_close(original, actual, rtol=0, atol=0)
                qkv = model.layers[0].self_attn.qkv_proj
                # A real PEFT file plus the rank-local FT export, including
                # replicated KV heads. The Q delta exceeds the old INT8 range;
                # only those output rows may change their weight step.
                loaded_qkv = load(ft, 2, rank, startup_lora=adapter).layers[0].self_attn.qkv_proj
                merged = qkv.weight.float() * qkv.weight_scale
                merged[:4] += delta.chunk(2)[rank]
                step = torch.maximum(qkv.weight_scale, torch.maximum(
                    merged.amax(1, keepdim=True) / 127, -merged.amin(1, keepdim=True) / 128
                ))
                torch.testing.assert_close(loaded_qkv.weight_scale, step, rtol=0, atol=0)
                torch.testing.assert_close(loaded_qkv.weight, (merged / step).round().clamp(-128, 127).to(torch.int8))
                torch.testing.assert_close(loaded_qkv.input_scale, qkv.input_scale, rtol=0, atol=0)
                torch.testing.assert_close(loaded_qkv.weight[4:], qkv.weight[4:], rtol=0, atol=0)
                x = (values(3, 8) - 5) / 3
                outputs = []
                for proj in ("q_proj", "k_proj", "v_proj"):
                    base = layer + "self_attn." + proj
                    w, s = tensors[base + ".qweight"], tensors[base + ".scales"]
                    expected = ((x / 0.5).round().int() @ w.int()) * 0.5 * s
                    outputs.append(
                        expected.chunk(2, -1)[rank] if proj == "q_proj" else expected
                    )
                torch.testing.assert_close(
                    qkv(x), torch.cat(outputs, -1), atol=0, rtol=0
                )
                gate = tensors[layer + "mlp.gate_proj.qweight"].T.chunk(2)[rank]
                up = tensors[layer + "mlp.up_proj.qweight"].T.chunk(2)[rank]
                torch.testing.assert_close(
                    model.layers[0].mlp.gate_up_proj.weight, torch.cat([gate, up])
                )
                torch.testing.assert_close(
                    model.layers[0].self_attn.o_proj.weight,
                    tensors[layer + "self_attn.o_proj.qweight"].chunk(2)[rank].T,
                )
            del tensors[layer + "self_attn.qkv_input_scale"]
            save_file(tensors, path)
            with self.assertRaisesRegex(RuntimeError, "input_scale"):
                load(root)
            # A historical archive missing QKV calibration cannot be repaired
            # from weight scales: never invent a static activation step.
            from safetensors import safe_open
            from safetensors.torch import load_file

            shard = next(ft.glob("model-00-00-part-*.safetensors"))
            fields = load_file(shard)
            with safe_open(shard, framework="pt") as handle:
                metadata = handle.metadata()
            gate_scale = "rank_00_00_00.layers.0.ffn_weights.intermediate_weight13.weight_only_quant_scale"
            flat = fields[gate_scale].reshape(-1).clone()
            fields[gate_scale] = flat.reshape(2, -1).T.contiguous()
            save_file(fields, shard, metadata=metadata)
            legacy = load(ft, 2, 0)
            torch.testing.assert_close(
                legacy.layers[0].mlp.gate_up_proj.weight_scale.flatten(), flat
            )
            field = "rank_00_00_00.layers.0.pre_layernorm_weights.static_quant"
            fields[field] = torch.tensor([4.0])
            save_file(fields, shard, metadata=metadata)
            with self.assertRaisesRegex(ValueError, "reciprocal static INT8"):
                load(ft, 2, 0)
            del fields[field]
            save_file(fields, shard, metadata=metadata)
            generate_ft_index(str(ft), {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 1})
            with self.assertRaisesRegex(ValueError, "Missing FT sources.*input_scale"):
                load(ft, 2, 0)

    def test_meta_consolidated_pth_without_hf_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tensors = write_meta_source(root)
            model = load(root)
            attn = model.layers[0].self_attn
            q = tensors["layers.0.attention.wq.weight"][[0, 2, 1, 3, 4, 6, 5, 7]]
            k = tensors["layers.0.attention.wk.weight"][[0, 2, 1, 3]]
            v = tensors["layers.0.attention.wv.weight"]
            torch.testing.assert_close(attn.qkv_proj.weight, torch.cat((q, k, v)))
            from rtp_llm.models_py.quant_methods.groupwise_export import cuda_groupwise_layout

            for rank in range(2):
                proj = load(root, tp=2, rank=rank, dtype=torch.bfloat16,
                            online_quant="INT8").layers[0].self_attn.qkv_proj
                local = torch.cat((q.chunk(2)[rank], k, v)).T.bfloat16()
                scale = local.abs().amax(0) / 128
                codes = (local / scale).round().clamp(-128, 127).to(torch.int8)
                packed = cuda_groupwise_layout(torch.nn.functional.pad(codes, (0, 52, 0, 56)), 8, 80)
                torch.testing.assert_close(proj.weight, packed)
                torch.testing.assert_close(proj.weight_scale[:12], scale)
            torch.testing.assert_close(model.embed_tokens.weight, tensors["tok_embeddings.weight"])
            torch.testing.assert_close(model.lm_head.weight, tensors["output.weight"])
            from rtp_llm.ops import TaskType

            backbone = load(root, task_type=TaskType.DENSE_EMBEDDING)
            self.assertIsNone(backbone.lm_head)
            torch.testing.assert_close(backbone.norm.weight, tensors["norm.weight"])
            torch.testing.assert_close(backbone.layers[0].self_attn.qkv_proj.weight, attn.qkv_proj.weight)
            for rank in range(2):
                path = root / f"consolidated.{rank:02d}.pth"
                part = torch.load(path, weights_only=True)
                del part["output.weight"]
                torch.save(part, path)
            self.assertIsNone(load(root, task_type=TaskType.DENSE_EMBEDDING).lm_head)
            with self.assertRaisesRegex(ValueError, "Missing FT sources for lm_head.weight"):
                load(root)
            for rank in range(2):
                path = root / f"consolidated.{rank:02d}.pth"
                part = torch.load(path, weights_only=True)
                part["unexpected.weight"] = torch.ones(1)
                torch.save(part, path)
            with self.assertRaisesRegex(ValueError, "Unbound FT tensors.*unexpected.weight"):
                load(root, task_type=TaskType.DENSE_EMBEDDING)

    def test_rank_local_ft_without_hf_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = values(8, 8)
            for rank in range(2):
                records = [
                    (None, "embedding", base.chunk(2, 1)[rank]),
                    (None, "lm_head", (base + 91).chunk(2, 0)[rank]),
                    (None, "final_layernorm.gamma", values(8)),
                    (0, "pre_layernorm_weights.gamma", values(8) + 1),
                    (0, "post_layernorm_weights.gamma", values(8) + 2),
                    (0, "self_attention_weights.query_weight.kernel", values(8, 12) + 100 * rank),
                    (0, "self_attention_weights.attention_output_weight.kernel", values(4, 8)),
                    (0, "ffn_weights.intermediate_weight13.kernel", values(8, 8)),
                    (0, "ffn_weights.intermediate_weight2.kernel", values(4, 8)),
                ]
                write_ft_shards(records, str(root), tp_rank=rank, ep_rank=0)
            generate_ft_index(str(root), {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 1})
            for rank in range(2):
                model = load(root, 2, rank)
                self.assertIsNone(model.layers[0].self_attn.qkv_proj.bias)
                torch.testing.assert_close(
                    model.layers[0].self_attn.qkv_proj.weight,
                    (values(8, 12) + 100 * rank).T,
                )
                torch.testing.assert_close(model.embed_tokens.weight, base.chunk(2)[rank])
                torch.testing.assert_close(model.lm_head.weight, (base + 91).chunk(2)[rank])


if __name__ == "__main__":
    unittest.main()
