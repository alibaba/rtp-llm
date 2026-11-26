"""Release-layout Qwen1: real FP16 files/transport; not a PPU PD golden."""

import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.config.py_config_modules import ResidentWeightsConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.qwen1.language import Qwen1ForCausalLM
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.ops import RoleType, TaskType


def release():
    def matrix(rows, cols, offset):
        return (torch.arange(rows * cols).reshape(rows, cols) + offset).half() / 64

    p = "transformer.h.0."
    return {
        "transformer.wte.weight": matrix(8, 4, 1),
        "transformer.ln_f.weight": torch.arange(1, 5).half(),
        "lm_head.weight": matrix(8, 4, 3),
        p + "ln_1.weight": torch.arange(2, 6).half(),
        p + "ln_2.weight": torch.arange(3, 7).half(),
        p + "attn.c_attn.weight": matrix(12, 4, 11),
        p + "attn.c_attn.bias": torch.arange(12).half(),
        p + "attn.c_proj.weight": matrix(4, 4, 7),
        p + "mlp.w2.weight": matrix(8, 4, 13),
        p + "mlp.w1.weight": matrix(8, 4, 51),
        p + "mlp.c_proj.weight": matrix(4, 8, 19),
    }


def write(root, missing=None):
    config = dict(
        architectures=["QWenLMHeadModel"],
        hidden_size=4,
        num_attention_heads=2,
        kv_channels=2,
        num_hidden_layers=1,
        intermediate_size=16,
        vocab_size=8,
        rotary_emb_base=20000,
        use_logn_attn=True,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
    )
    (root / "config.json").write_text(json.dumps(config))
    values = release()
    if missing is not None:
        del values[missing]
    index = {}
    for slot in range(2):
        tensors = dict(list(values.items())[slot::2])
        name = f"model-{slot}.safetensors"
        save_file(tensors, str(root / name))
        index.update(dict.fromkeys(tensors, name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )


def load(root, method, role, tp, rank, device="cpu"):
    cfg = ModelFactory.get_model_cls("qwen_7b").create_config(str(root))
    cfg.model_type, cfg.data_type, cfg.task_type = (
        "qwen_7b",
        "fp16",
        TaskType.LANGUAGE_MODEL,
    )
    ns = SimpleNamespace
    pc = ns(
        tp_size=tp,
        tp_rank=rank,
        ep_size=1,
        ep_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        world_size=tp,
        world_rank=rank,
        local_rank=rank,
        role_type=role,
        get_attn_tp_size=lambda: tp,
        get_attn_tp_rank=lambda: rank,
        get_ffn_tp_size=lambda: tp,
        get_ffn_tp_rank=lambda: rank,
    )
    engine = ns(
        parallelism_config=pc,
        runtime_config=ns(
            warm_up=False, model_warm_up=False, max_generate_batch_size=2
        ),
        hw_kernel_config=ns(enable_cuda_graph=False, use_swizzleA=False),
        kv_cache_config=ns(multi_task_prompt="", multi_task_prompt_str=""),
        fmha_config=None,
        moe_config=None,
        load_config=ns(
            resident_weights_config=ResidentWeightsConfig(),
            load_method=method,
            force_cpu_load_weights=False,
            loader_recycle_handles=False,
            moe_pure_tp_preshard=False,
        ),
    )
    # Tokenizer assets are not needed to verify the checkpoint ABI.
    with patch.object(BaseModel, "load_tokenizer"), patch.object(
        BaseModel, "_get_device_str", return_value=device
    ):
        svc = ModelFactory._create_model(cfg, engine)
    assert svc.model_weights_loader.loaded_method == NewLoaderLoadMethod(method)
    return svc.py_model


def check(root, method, role, tp, rank):
    model = load(root, method, role, tp, rank)
    assert model.parallelism_config.role_type == role
    assert model.config.attn_config.use_logn_attn
    assert model.config.attn_config.rope_config.base == 20000
    assert model.config.inter_size == 8
    data, p = release(), "transformer.h.0."
    layer = model.layers[0]

    def same(actual, expected):
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)

    def shard(tensor, dim):
        return tensor.chunk(tp, dim=dim)[rank]

    for leaf in ("weight", "bias"):
        packed = data[p + "attn.c_attn." + leaf]
        expected = torch.cat([shard(part, 0) for part in packed.chunk(3, dim=0)])
        same(getattr(layer.self_attn.qkv_proj, leaf), expected)
    same(layer.self_attn.o_proj.weight, shard(data[p + "attn.c_proj.weight"], 1))
    same(
        layer.mlp.gate_up_proj.weight,
        torch.cat(
            [shard(data[p + "mlp." + part + ".weight"], 0) for part in ("w2", "w1")]
        ),
    )
    same(layer.mlp.down_proj.weight, shard(data[p + "mlp.c_proj.weight"], 1))
    same(layer.input_layernorm.weight, data[p + "ln_1.weight"])
    same(layer.post_attention_layernorm.weight, data[p + "ln_2.weight"])
    same(model.norm.weight, data["transformer.ln_f.weight"])
    same(model.embed_tokens.weight, shard(data["transformer.wte.weight"], 0))
    same(model.runtime_weight_view()["lm_head"], shard(data["lm_head.weight"], 0))
    if tp == 1:
        x = torch.tensor([[1.0, 2.0, -3.0, 0.5]], dtype=torch.float16)
        gate = F.linear(x, data[p + "mlp.w2.weight"])
        up = F.linear(x, data[p + "mlp.w1.weight"])
        expected = F.linear(
            (F.silu(gate.float()) * up.float()).half(), data[p + "mlp.c_proj.weight"]
        )
        torch.testing.assert_close(layer.mlp(x), expected)


def worker(rank, root):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(root / "store"),
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=40),
    )
    try:
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            for role in (RoleType.PREFILL, RoleType.DECODE):
                check(root, method, role, 2, rank)
    finally:
        dist.destroy_process_group()


class Qwen1CheckpointTest(unittest.TestCase):
    def test_ppu_streamed_scratch_policy(self):
        owner = SimpleNamespace(ppu_dense_streamed_scratch=True)
        method = NewLoaderLoadMethod.SCRATCH

        def selected(config, candidate=owner, reader=method):
            return NewModelLoader({}, config)._ppu_streamed_scratch(candidate, reader)

        with patch.object(
            NewModelLoader, "_accelerator_backend_name", return_value="Ppu"
        ):
            for dtype in (torch.float16, torch.bfloat16):
                config = NewLoaderConfig(
                    compute_dtype=dtype, quant_config=SimpleNamespace(quant_type="none")
                )
                self.assertTrue(selected(config))
            for config in (
                NewLoaderConfig(device="cpu"),
                NewLoaderConfig(compute_dtype=torch.float32),
                NewLoaderConfig(quant_config=SimpleNamespace(quant_type="FP8")),
                NewLoaderConfig(force_cpu_load_weights=True),
            ):
                self.assertFalse(selected(config))
            self.assertFalse(selected(NewLoaderConfig(), SimpleNamespace()))
            self.assertFalse(
                selected(NewLoaderConfig(), reader=NewLoaderLoadMethod.FASTSAFETENSORS)
            )
            isolated = SimpleNamespace(
                ppu_dense_streamed_scratch=True, fastsafetensors_load_isolated=True
            )
            self.assertFalse(selected(NewLoaderConfig(), isolated))
            with patch.object(dist, "is_initialized", return_value=True), patch.object(
                dist, "get_world_size", return_value=2
            ):
                self.assertFalse(selected(NewLoaderConfig()))
        with patch.object(
            NewModelLoader, "_accelerator_backend_name", return_value="Cuda"
        ):
            self.assertFalse(selected(NewLoaderConfig()))

    @unittest.skipUnless(os.getenv("PPU_QWEN_SCRATCH_TEST") == "1", "requires a PPU")
    def test_ppu_scratch_populates_final_parameter_storage(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(NewModelLoader._accelerator_backend_name(), "Ppu")
        device = torch.device("cuda", torch.cuda.current_device())
        original = Qwen1ForCausalLM.load_weights
        seen = {}

        def observe(owner, weights):
            for name, value in owner.named_parameters():
                self.assertEqual(value.device, device)
                seen[name] = value.data_ptr()

            def checked_weights():
                for name, value in weights:
                    source, _ = unwrap_checkpoint_tensor(value)
                    self.assertEqual(source.device.type, "cpu")
                    yield name, value

            return original(owner, checked_weights())

        with tempfile.TemporaryDirectory() as path, patch.object(
            Qwen1ForCausalLM, "load_weights", observe
        ):
            write(Path(path))
            model = load(Path(path), "scratch", RoleType.PREFILL, 1, 0, str(device))
        self.assertTrue(seen)
        self.assertEqual({n: p.data_ptr() for n, p in model.named_parameters()}, seen)
        expected = release()
        torch.testing.assert_close(
            model.embed_tokens.weight.cpu(), expected["transformer.wte.weight"]
        )
        torch.testing.assert_close(
            model.layers[0].mlp.gate_up_proj.weight.cpu(),
            torch.cat(
                [
                    expected["transformer.h.0.mlp.w2.weight"],
                    expected["transformer.h.0.mlp.w1.weight"],
                ]
            ),
        )

    def test_fp16_prefill_and_decode(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            write(root)
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                for role in (RoleType.PREFILL, RoleType.DECODE):
                    with self.subTest(method=method, role=role):
                        check(root, method, role, 1, 0)

    def test_tp2_real_transport_and_local_ownership(self):
        with tempfile.TemporaryDirectory() as path:
            write(Path(path))
            torch.multiprocessing.start_processes(
                worker, args=(path,), nprocs=2, join=True, start_method="fork"
            )

    def test_missing_gate_and_fused_bias_fail(self):
        for key in (
            "transformer.h.0.mlp.w2.weight",
            "transformer.h.0.attn.c_attn.bias",
        ):
            with tempfile.TemporaryDirectory() as path:
                write(Path(path), missing=key)
                for method in checkpoint_readers("scratch", "fastsafetensors"):
                    with self.subTest(key=key, method=method), self.assertRaisesRegex(
                        RuntimeError, "missing|required|Missing"
                    ):
                        load(Path(path), method, RoleType.DECODE, 1, 0)


if __name__ == "__main__":
    unittest.main()
