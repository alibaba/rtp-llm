"""Offline HF -> production rank shards -> service parameters, without GPU kernels."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models.downstream_modules.classifier.classifier import ClassifierHandler
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.glm4_moe_lite import Glm4MoeLite
from rtp_llm.models.glm5 import Glm5
from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v3.language import DeepSeekV32ForCausalLM
from rtp_llm.models_py.new_models.deepseek_v3.test.test_glm5_mtp_route import (
    _Draft as GlmDraft,
)
from rtp_llm.models_py.new_models.deepseek_v3.test.test_nextn_route import (
    _CpuCheckpoint,
    _Draft,
    _Score,
    fixture,
    load_service,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops import HWKernelConfig, ParallelismConfig, TaskType
from rtp_llm.tools.convert.ft_export_job import create_ft_export_job
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index


class GlmScore(_CpuCheckpoint, Glm5):
    pass


class GlmLiteScore(_CpuCheckpoint, Glm4MoeLite):
    pass


class BlockScore(_Score):
    @classmethod
    def _create_config(cls, root):
        cfg = super()._create_config(root)
        cfg.quant_config = Fp8BlockWiseQuantConfig(
            bits=8, group_size=128, is_quanted=True
        )
        return cfg


class GpuBlockScore(BlockScore):
    def _get_device_str(self):
        return "cuda:0"


def _two_rank_round_trip(rank, root):
    import torch.distributed as dist

    root = Path(root)
    hf, ft = root / "hf", root / "ft"
    dist.init_process_group(
        "gloo",
        init_method=(root / "dist").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        pc = ParallelismConfig()
        pc.world_size = pc.local_world_size = pc.tp_size = pc.ep_size = 2
        pc.world_rank = pc.local_rank = pc.tp_rank = pc.ep_rank = rank
        pc.ffn_tp_size, pc.ffn_tp_rank = 2, rank
        source = load_service(
            str(hf), "scratch", cls=_Score, tag="deepseek2", parallel=pc
        )
        source.kv_cache_config = KVCacheConfig()
        create_ft_export_job(source).dump_weight_as_ft_style("cpu", str(ft))
        dist.barrier()
        if rank == 0:
            generate_ft_index(str(ft), {"TP_SIZE": "2", "EP_SIZE": "2", "DP_SIZE": "1"})
        dist.barrier()
        actual = load_service(
            str(ft), "scratch", cls=_Score, tag="deepseek2", parallel=pc
        ).py_model
        for key, value in source.py_model.named_parameters():
            torch.testing.assert_close(actual.get_parameter(key), value, atol=0, rtol=0)
        for expected, result in zip(source.py_model.layers, actual.layers):
            for key in ("_fused_qkv_a_w", "_kc_w", "_vc_w"):
                torch.testing.assert_close(
                    getattr(result.self_attn, key),
                    getattr(expected.self_attn, key),
                    atol=0,
                    rtol=0,
                )
    finally:
        dist.destroy_process_group()


class MlaFtRoundTrip(unittest.TestCase):
    def round_trip(self, tag, service_cls, *, sparse=False, direct=False, cropped_draft=False):
        draft = service_cls in (_Draft, GlmDraft)
        cfg, data, draft_prefix = fixture(not draft or cropped_draft)
        if cropped_draft:
            data = {
                key.replace(draft_prefix, "model.layers.78.", 1) if key.startswith(draft_prefix) else key: val
                for key, val in data.items()
            }
        if not draft:
            cfg["num_nextn_predict_layers"] = 0
            data = {k: v for k, v in data.items() if not k.startswith(draft_prefix)}
        if direct:
            cfg["q_lora_rank"] = None
            for i in range(cfg["num_hidden_layers"]):
                prefix = f"model.layers.{i}.self_attn."
                data[prefix + "q_proj.weight"] = data.pop(prefix + "q_a_proj.weight")
                del (
                    data[prefix + "q_b_proj.weight"],
                    data[prefix + "q_a_layernorm.weight"],
                )
        if sparse:
            cfg.update(index_topk=4, index_n_heads=2, index_head_dim=128)
            for i in ([78] if cropped_draft else range(cfg["num_hidden_layers"])):
                for key, shape in (
                    ("wq_b.weight", (256, 32)),
                    ("wk.weight", (128, 32)),
                    ("k_norm.weight", (128,)),
                    ("k_norm.bias", (128,)),
                    ("weights_proj.weight", (2, 32)),
                ):
                    data[f"model.layers.{i}.self_attn.indexer." + key] = (
                        torch.arange(
                            torch.Size(shape).numel(), dtype=torch.float32
                        ).reshape(shape)
                        / 13
                    )
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            hf, ft = root / "hf", root / "ft"
            hf.mkdir()
            ft.mkdir()
            for place in (hf, ft):
                (place / "config.json").write_text(json.dumps(cfg))
            save_file(data, str(hf / "model.safetensors"))
            service = load_service(str(hf), "scratch", cls=service_cls, tag=tag)
            service.kv_cache_config = KVCacheConfig()
            create_ft_export_job(service).dump_weight_as_ft_style("cpu", str(ft))
            generate_ft_index(str(ft), {"TP_SIZE": "1", "EP_SIZE": "1", "DP_SIZE": "1"})
            reference = service.py_model
            readers = (
                ("scratch",)
                if os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0"
                else ("scratch", "fastsafetensors")
            )
            for method in readers:
                actual = load_service(
                    str(ft), method, cls=service_cls, tag=tag
                ).py_model
                for name, weight in reference.named_parameters():
                    with self.subTest(tag=tag, reader=method, parameter=name):
                        torch.testing.assert_close(
                            actual.get_parameter(name), weight, atol=0, rtol=0
                        )
                for expected, result in zip(reference.layers, actual.layers):
                    for name in ("_kc_w", "_vc_w", "_fused_qkv_a_w"):
                        torch.testing.assert_close(
                            getattr(result.self_attn, name),
                            getattr(expected.self_attn, name),
                            atol=0,
                            rtol=0,
                        )

    def test_cropped_glm_draft_ft_keeps_shared_vocab_and_physical_draft_layer(self):
        self.round_trip("glm_5_mtp", GlmDraft, sparse=True, cropped_draft=True)

    def test_score_aliases(self):
        for tag in (
            "deepseek2",
            "deepseek3",
            "deepseek_v31",
            "deepseek_v32",
            "kimi_k2",
            "glm_5",
            "glm4_moe_lite",
        ):
            with self.subTest(tag=tag):
                self.round_trip(
                    tag,
                    {"glm_5": GlmScore, "glm4_moe_lite": GlmLiteScore}.get(tag, _Score),
                    sparse=tag in ("deepseek_v32", "glm_5"),
                    direct=tag == "deepseek2",
                )

    def test_drafts(self):
        for tag, owner in (("deepseek-v3-mtp", _Draft), ("glm_5_mtp", GlmDraft)):
            with self.subTest(tag=tag):
                self.round_trip(tag, owner, sparse=tag == "glm_5_mtp")

    def test_tp2_ep2_direct_q_and_replicated_kv(self):
        cfg, data, prefix = fixture(True)
        cfg.update(num_nextn_predict_layers=0, q_lora_rank=None)
        data = {k: v for k, v in data.items() if not k.startswith(prefix)}
        for i in range(cfg["num_hidden_layers"]):
            src = f"model.layers.{i}.self_attn."
            data[src + "q_proj.weight"] = data.pop(src + "q_b_proj.weight")
            del data[src + "q_a_proj.weight"], data[src + "q_a_layernorm.weight"]
        with tempfile.TemporaryDirectory() as path:
            hf, ft = Path(path) / "hf", Path(path) / "ft"
            for dest in (hf, ft):
                dest.mkdir()
                (dest / "config.json").write_text(json.dumps(cfg))
            save_file(data, str(hf / "model.safetensors"))
            torch.multiprocessing.spawn(_two_rank_round_trip, args=(path,), nprocs=2)

    def test_offline_block_fp8_writer_and_staging(self):
        self.check_block_writer("cpu")

    def check_block_writer(self, device):
        cfg, _, _ = fixture(True)
        cfg.update(
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
            hidden_size=256,
            intermediate_size=256,
            num_attention_heads=2,
            q_lora_rank=128,
            kv_lora_rank=128,
            qk_nope_head_dim=128,
            qk_rope_head_dim=128,
            v_head_dim=128,
            index_topk=8,
            index_n_heads=2,
            index_head_dim=128,
            quantization_config=dict(
                quant_method="fp8",
                fmt="e4m3",
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            ),
        )
        p = "model.layers.0."
        rng = torch.Generator().manual_seed(19)
        data = {
            "model.embed_tokens.weight": torch.randn(32, 256, generator=rng).bfloat16(),
            "lm_head.weight": torch.randn(32, 256, generator=rng).bfloat16(),
            "model.norm.weight": torch.ones(256, dtype=torch.bfloat16),
        }
        for name, size in (
            ("input_layernorm.weight", 256),
            ("post_attention_layernorm.weight", 256),
            ("self_attn.q_a_layernorm.weight", 128),
            ("self_attn.kv_a_layernorm.weight", 128),
            ("self_attn.indexer.k_norm.weight", 128),
            ("self_attn.indexer.k_norm.bias", 128),
        ):
            data[p + name] = torch.randn(size, generator=rng).bfloat16()
        data[p + "self_attn.indexer.weights_proj.weight"] = torch.randn(
            2, 256, generator=rng
        )
        for name, n, k in (
            ("self_attn.q_a_proj", 128, 256),
            ("self_attn.kv_a_proj_with_mqa", 256, 256),
            ("self_attn.q_b_proj", 512, 128),
            ("self_attn.kv_b_proj", 512, 128),
            ("self_attn.o_proj", 256, 256),
            ("self_attn.indexer.wq_b", 256, 128),
            ("self_attn.indexer.wk", 128, 256),
            ("mlp.gate_proj", 256, 256),
            ("mlp.up_proj", 256, 256),
            ("mlp.down_proj", 256, 256),
        ):
            data[p + name + ".weight"] = torch.randint(-8, 9, (n, k), generator=rng).to(
                torch.float8_e4m3fn
            )
            data[p + name + ".weight_scale_inv"] = torch.exp2(
                torch.randint(-3, 4, (n // 128, k // 128), generator=rng).float()
            )
        with tempfile.TemporaryDirectory() as path:
            hf, ft = Path(path) / "hf", Path(path) / "ft"
            for dest in (hf, ft):
                dest.mkdir()
                (dest / "config.json").write_text(json.dumps(cfg))
            save_file(data, str(hf / "model.safetensors"))
            cfg_owner = BlockScore._create_config(str(hf))
            cfg_owner.ckpt_path, cfg_owner.model_type, cfg_owner.max_seq_len = (
                str(hf),
                "deepseek_v32",
                128,
            )
            cfg_owner.attn_config.tokens_per_block = (
                cfg_owner.attn_config.kernel_tokens_per_block
            ) = 64
            cfg_owner.data_type, cfg_owner.enable_fp32_lm_head, cfg_owner.lora_infos = (
                "bf16",
                False,
                {},
            )
            cfg_owner.init_precision_config(KVCacheConfig(), "bf16")
            self.assertTrue(cfg_owner.quant_algo.isQuant())
            BlockScore._post_build_model_config(cfg_owner)
            pc = ParallelismConfig()
            service = BlockScore(
                cfg_owner,
                pc,
                HWKernelConfig(),
                KVCacheConfig(),
                None,
                None,
                1,
                "scratch",
                None,
                False,
            )
            create_ft_export_job(service).dump_weight_as_ft_style(device, str(ft))
            generate_ft_index(str(ft), {"TP_SIZE": "1", "EP_SIZE": "1", "DP_SIZE": "1"})
            if device == "cpu":
                # Staging only: FP8 kernel setup requires an accelerator.
                # Exercise the production importer directly and check the
                # original FP8 bytes, scales and independent BF16 absorb copies.
                cfg_owner.ckpt_path = str(ft)
                options = NewLoaderConfig(
                    device="cpu",
                    parallelism_config=pc,
                    compute_dtype=torch.bfloat16,
                    quant_config=QuantizationConfig(
                        "fp8_block", cfg_owner.quant_config
                    ),
                )
                actual = DeepSeekV32ForCausalLM(cfg_owner, options)
                checkpoint = FTCheckpoint.open(
                    str(ft), CheckpointTopology(), num_layers=1
                )
                actual.load_ft_checkpoint(checkpoint)
                for key in data:
                    if not key.startswith(p + "self_attn."):
                        continue
                    expected = data[key]
                    got = actual.get_parameter(key.removeprefix("model."))
                    if expected.dtype == torch.float8_e4m3fn:
                        expected, got = (
                            expected.view(torch.uint8),
                            got.view(torch.uint8),
                        )
                    torch.testing.assert_close(got, expected, atol=0, rtol=0)
                kv = data[p + "self_attn.kv_b_proj.weight"].float() * data[
                    p + "self_attn.kv_b_proj.weight_scale_inv"
                ].repeat_interleave(128, 0).repeat_interleave(128, 1)
                expanded = kv.bfloat16().reshape(2, 256, 128)
                torch.testing.assert_close(
                    actual.layers[0].self_attn._kc_w,
                    expanded[:, :128, :],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    actual.layers[0].self_attn._vc_w,
                    expanded[:, 128:, :].transpose(1, 2),
                    atol=0,
                    rtol=0,
                )
                return
            source = load_service(
                str(hf), "scratch", cls=GpuBlockScore, tag="deepseek_v32"
            ).py_model
            actual = load_service(
                str(ft), "scratch", cls=GpuBlockScore, tag="deepseek_v32"
            ).py_model
            for key, expected in source.named_parameters():
                got = actual.get_parameter(key)
                if expected.dtype == torch.float8_e4m3fn:
                    expected, got = expected.view(torch.uint8), got.view(torch.uint8)
                torch.testing.assert_close(got, expected, atol=0, rtol=0)
            for key in ("_kc_w", "_vc_w"):
                torch.testing.assert_close(
                    getattr(actual.layers[0].self_attn, key),
                    getattr(source.layers[0].self_attn, key),
                    atol=0,
                    rtol=0,
                )

    def test_native_classifier_archive(self):
        cfg, data, prefix = fixture(True)
        cfg.update(
            num_nextn_predict_layers=0,
            num_labels=3,
            id2label={"0": "a", "1": "b", "2": "c"},
        )
        data = {k: v for k, v in data.items() if not k.startswith(prefix)}
        with tempfile.TemporaryDirectory() as path:
            hf, ft = Path(path) / "hf", Path(path) / "ft"
            hf.mkdir()
            ft.mkdir()
            for place in (hf, ft):
                (place / "config.json").write_text(json.dumps(cfg))
            save_file(data, str(hf / "model.safetensors"))
            service = load_service(str(hf), "scratch", cls=_Score, tag="deepseek3")
            data.update(
                {
                    "classifier.weight": torch.arange(96).reshape(3, 32).bfloat16(),
                    "classifier.bias": torch.tensor([0, 1, 2], dtype=torch.bfloat16),
                }
            )
            save_file(data, str(hf / "model.safetensors"))
            service.kv_cache_config = KVCacheConfig()
            config = service.model_config
            config.task_type = TaskType.SEQ_CLASSIFICATION
            module = CustomModule(config, None)
            module.handler = ClassifierHandler(config)
            service._init_custom_module = lambda: module
            create_ft_export_job(service).dump_weight_as_ft_style("cpu", str(ft))
            generate_ft_index(str(ft), {"TP_SIZE": "1", "EP_SIZE": "1", "DP_SIZE": "1"})
            config.ckpt_path = str(ft)
            options = NewLoaderConfig(
                device="cpu",
                parallelism_config=service.parallelism_config,
                compute_dtype=torch.bfloat16,
                load_method="scratch",
                custom_weight_mappings=module.checkpoint_weight_mappings(),
            )
            owner = NewModelLoader(config, options, model_path=str(ft)).load()
            exports = owner.runtime_weight_view()
            self.assertNotIn("lm_head", exports)
            self.assertIsNone(owner.lm_head)
            for role, source in module.checkpoint_weight_mappings():
                torch.testing.assert_close(exports[role], data[source], atol=0, rtol=0)
            # A task without a declared classifier may not silently discard
            # the archive's additional tensors.
            config.task_type = TaskType.DENSE_EMBEDDING
            options = replace(options, custom_weight_mappings=())
            with self.assertRaisesRegex(ValueError, "Unbound FT tensors"):
                NewModelLoader(config, options, model_path=str(ft)).load()


if __name__ == "__main__":
    unittest.main()
