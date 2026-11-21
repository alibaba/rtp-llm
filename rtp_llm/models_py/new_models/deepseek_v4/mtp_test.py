"""Real V4 MTP checkpoint I/O. GPU service publication is a separate case."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models.deepseek_v4 import DeepSeekV4Mtp
from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v4.mtp import DeepSeekV4MtpForCausalLM
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.ops import KvCacheDataType, ParallelismConfig


def _config(root):
    cfg = DeepSeekV4Mtp._create_config(str(root))
    cfg.model_type = "deepseek_v4_mtp"
    cfg.ckpt_path = str(root)
    cfg.max_seq_len = 128
    cfg.gen_num_per_cycle = 2
    cfg.data_type = "bf16"
    cfg.attn_config.kv_cache_dtype = KvCacheDataType.FP8
    return cfg


def _tiny(root):
    from rtp_llm.models_py.new_models.deepseek_v4.backbone_test import (
        Dsv4FullBackboneCheckpointTest,
    )

    config = dict(
        num_hidden_layers=3,
        hidden_size=128,
        vocab_size=17,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_groups=2,
        o_lora_rank=128,
        compress_ratios=[0, 4, 128, 0],
        sliding_window=128,
        index_head_dim=128,
        index_n_heads=2,
        index_topk=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        num_experts_per_tok=2,
        n_routed_experts=4,
        moe_intermediate_size=128,
        n_shared_experts=1,
        hc_mult=2,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        num_hash_layers=1,
    )
    (root / "config.json").write_text(json.dumps(config))
    source = Dsv4FullBackboneCheckpointTest().weights()
    data = {}
    for name, value in source.items():
        if name in ("embed.weight", "head.weight"):
            data[name] = value
        elif name.startswith("layers.0.") and not name.endswith("tid2eid"):
            data["mtp.0." + name[len("layers.0.") :]] = value
        elif not name.startswith("layers."):
            data["mtp.0." + name] = value
    data["mtp.0.ffn.gate.bias"] = torch.arange(4).float()
    for i, stem in enumerate(("e", "h"), 1):
        data[f"mtp.0.{stem}norm.weight"] = torch.full((128,), i, dtype=torch.bfloat16)
        data[f"mtp.0.{stem}_proj.weight"] = torch.full(
            (128, 128), i, dtype=torch.float8_e4m3fn
        )
        data[f"mtp.0.{stem}_proj.scale"] = torch.tensor(
            [[125 + i]], dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    data["layers.0.attn.wkv.weight"] = torch.zeros(1)
    data["mtp.1.norm.weight"] = torch.zeros(1)
    save_file(data, str(root / "model.safetensors"))
    return _config(root), data


def _service(root, method, device, config):
    instance = service(str(root), method, config.model_type)
    instance.model_config = config
    instance.parallelism_config = ParallelismConfig()
    instance._get_device_str = lambda: device
    return instance


class V4MtpCheckpointTest(unittest.TestCase):
    def test_real_single_layer_io_and_service_cpu_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, data = _tiny(root)
            self.assertEqual(
                DeepSeekV4Mtp.speculative_weight_alias_names(None, cfg), ()
            )
            cls = get_model_class(cfg.model_type)
            self.assertIs(cls, DeepSeekV4MtpForCausalLM)
            self.assertIs(cls.forward, DeepSeekV4MtpModel.forward)
            self.assertIs(cls._build_fused, DeepSeekV4MtpModel._build_fused)
            for method in ("scratch", "fastsafetensors"):
                instance = _service(root, method, "cpu", cfg)
                load = NewLoaderConfig.from_runtime(
                    instance.parallelism_config,
                    device="cpu",
                    compute_dtype=cfg.compute_dtype,
                    load_method=method,
                    max_generate_batch_size=1,
                )
                model = cls(cfg, load)
                self.assertTrue(model.cuda_graph_draft_prefill_uses_request_lanes())
                self.assertEqual(len(model.v4.layers), 1)
                self.assertFalse(model.v4.layers[0].ffn._hash_gate)
                reader = NewModelLoader(cfg, load, model_path=folder)
                include = model.checkpoint_weight_name_filter()
                resolve = reader._checkpoint_read_spec_resolver(model, include, None)
                files = reader._checkpoint_files()
                items = (
                    get_all_weights(
                        files, name_filter=include, read_spec_resolver=resolve
                    )
                    if method == "scratch"
                    else reader._fastsafetensors_weights_iterator(
                        files, "cpu", include, None, resolve
                    )
                )
                model.load_weights(items)
                reader._validate_loaded_weights(model)
                for name in (
                    "embed.weight",
                    "head.weight",
                    "mtp.0.norm.weight",
                    "mtp.0.enorm.weight",
                    "mtp.0.hnorm.weight",
                    "mtp.0.e_proj.weight",
                    "mtp.0.h_proj.weight",
                    "mtp.0.e_proj.scale",
                    "mtp.0.h_proj.scale",
                ):
                    value = model.get_parameter(model.WEIGHTS_MAPPER.map_name(name))
                    self.assertTrue(
                        torch.equal(
                            value.view(torch.uint8), data[name].view(torch.uint8)
                        ),
                        name,
                    )
                with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                    load_owned_model(instance)
                self.assertIsNone(instance.py_model)
                self.assertIsNone(instance.model_weights_loader)

    @unittest.skipUnless(
        os.environ.get("V4_MTP_RELEASE"), "requires actual V4 MTP checkpoint"
    )
    def test_release_complete_plan_and_real_fusion_bytes(self):
        root = Path(os.environ["V4_MTP_RELEASE"])
        cfg = _config(root)
        load = NewLoaderConfig.from_runtime(
            ParallelismConfig(),
            device="cpu",
            compute_dtype=cfg.compute_dtype,
            max_generate_batch_size=1,
        )
        with torch.device("meta"):
            model = get_model_class(cfg.model_type)(cfg, load)
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        names = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        include = model.checkpoint_weight_name_filter()
        selected = {k: v for k, v in names.items() if include(k)}
        self.assertEqual(
            set(selected),
            {k for k in names if k.startswith("mtp.0.")}
            | {"embed.weight", "head.weight"},
        )
        reader = NewModelLoader(cfg, load, model_path=str(root))
        resolve = reader._checkpoint_read_spec_resolver(model, include, None)
        fusion = {f"mtp.0.{name}" for name in model.WEIGHTS_MAPPER.fusion_leaves}
        self.assertTrue(fusion.issubset(selected))
        for leaf in (model.enorm, model.hnorm, model.e_proj, model.h_proj):
            leaf.to_empty(device="cpu")
        for shard in set(selected.values()):
            with safe_open(str(root / shard), framework="pt", device="cpu") as sf:
                for name, src in selected.items():
                    if src != shard:
                        continue
                    shape = tuple(sf.get_slice(name).get_shape())
                    target, _ = model.resolve_checkpoint_load_plan(
                        model.WEIGHTS_MAPPER.map_name(name), shape
                    )
                    self.assertIsNotNone(target, name)
                    self.assertTrue(resolve(name, shape).load)
                    if name in fusion:
                        value = sf.get_tensor(name)
                        model.load_weights([(name, value)])
                        actual = model.get_parameter(
                            model.WEIGHTS_MAPPER.map_name(name)
                        )
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.uint8), value.view(torch.uint8)
                            ),
                            name,
                        )

    @unittest.skipUnless(
        os.environ.get("V4_MTP_SERVICE_GPU"), "requires CUDA13 V4 execution device"
    )
    def test_real_gpu_service_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, data = _tiny(root)
            for method in ("scratch", "fastsafetensors"):
                instance = _service(
                    root, method, os.environ["V4_MTP_SERVICE_GPU"], config
                )
                load_owned_model(instance)
                self.assertIs(type(instance.py_model), DeepSeekV4MtpForCausalLM)
                self.assertEqual(
                    instance.model_weights_loader.loaded_method.value, method
                )
                self.assertIs(
                    instance.weight.get_global_weight("lm_head"),
                    instance.py_model.v4.head.weight,
                )
                self.assertTrue(
                    torch.equal(
                        instance.py_model.e_proj.weight.cpu().view(torch.uint8),
                        data["mtp.0.e_proj.weight"].view(torch.uint8),
                    )
                )


if __name__ == "__main__":
    unittest.main()
