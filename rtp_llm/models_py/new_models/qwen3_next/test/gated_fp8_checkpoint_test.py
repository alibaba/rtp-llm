"""Real FP8 checkpoints: split interleaved Q/gate before slicing for TP."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3_next.test.main_checkpoint_test import (
    _config,
    _load_config,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig

_MODELS = ("qwen3_next", "qwen35_moe", "qwen35_dense")


def _codes(grid):
    return (
        grid.remainder(17)
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
        .to(torch.float8_e4m3fn)
    )


def _fixture(tag):
    cfg = _config()
    cfg.model_type = tag
    cfg.hidden_size = cfg.inter_size = 256
    cfg.attn_config.head_num = 4
    cfg.attn_config.kv_head_num = 2
    cfg.attn_config.size_per_head = 256
    if tag == "qwen35_dense":
        cfg.expert_num = cfg.moe_k = cfg.moe_inter_size = cfg.moe_style = 0
    # Qwen3.5 includes the language_model wrapper even for the same attention.
    root = "model." if tag == "qwen3_next" else "model.language_model."
    layer = root + "layers.0."
    weights = {
        root + "embed_tokens.weight": torch.ones(8, 256, dtype=torch.bfloat16),
        root + "norm.weight": torch.zeros(256),
    }
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
    ):
        weights[layer + name + ".weight"] = torch.zeros(256)
    scales = {}
    projections = (
        ("self_attn.q_proj", 2048, 256),
        ("self_attn.k_proj", 512, 256),
        ("self_attn.v_proj", 512, 256),
        ("self_attn.o_proj", 256, 1024),
        ("mlp.gate_proj", 256, 256),
        ("mlp.up_proj", 256, 256),
        ("mlp.down_proj", 256, 256),
    )
    for offset, (name, rows, cols) in enumerate(projections):
        # Distinguish every grid cell, including both output blocks per head.
        n, k = rows // 128, cols // 128
        grid = (1 + 32 * offset + torch.arange(n * k)).float().reshape(n, k)
        # The published Next FP8 grid is F32; the Qwen3.5 grid is BF16.
        scale_dtype = torch.float32 if tag == "qwen3_next" else torch.bfloat16
        weights[layer + name + ".weight_scale_inv"] = grid.to(scale_dtype)
        scales[name] = grid
        weights[layer + name + ".weight"] = _codes(grid)
    return cfg, weights, scales


def _check_rank(rank, directory):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        for tag in _MODELS:
            cfg, _, grids = _fixture(tag)
            packed = grids["self_attn.q_proj"].reshape(4, 2, 2, 2)
            heads = packed[2 * rank : 2 * rank + 2]
            kv = slice(2 * rank, 2 * rank + 2)
            expected_qkv = torch.cat(
                (
                    heads[:, 0].reshape(4, 2),
                    grids["self_attn.k_proj"][kv],
                    grids["self_attn.v_proj"][kv],
                )
            )
            for method in checkpoint_readers("scratch", "fastsafetensors"):
                opts = _load_config(
                    tp_size=2,
                    tp_rank=rank,
                    quant_config=QuantizationConfig(
                        "fp8_block",
                        source_config=SimpleNamespace(weight_block_size=[128, 128]),
                    ),
                )
                opts = replace(
                    opts,
                    compute_dtype=torch.bfloat16,
                    device=f"cuda:{rank}",
                    load_method=method,
                )
                loader = NewModelLoader(
                    cfg, opts, model_path=str(Path(directory) / tag)
                )
                model = loader.load()
                if loader.loaded_method.value != method:
                    raise AssertionError(f"{tag} rank {rank} did not use {method}")
                attn = model.layers[0].self_attn
                for projection, expected in (
                    (attn.qkv_proj, expected_qkv),
                    (attn.gate, heads[:, 1].reshape(4, 2)),
                    (
                        attn.o_proj,
                        grids["self_attn.o_proj"][:, 4 * rank : 4 * rank + 4],
                    ),
                ):
                    torch.testing.assert_close(
                        projection.weight_scale.cpu(), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        projection.weight.cpu().view(torch.uint8),
                        _codes(expected).view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
    finally:
        torch.distributed.destroy_process_group()


class GatedFp8CheckpointTest(unittest.TestCase):
    def test_q_gate_blocks_and_horizontal_output_scale_on_both_tp_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            for tag in _MODELS:
                _, weights, _ = _fixture(tag)
                root = Path(directory) / tag
                root.mkdir()
                save_file(weights, str(root / "model.safetensors"))
            torch.multiprocessing.spawn(_check_rank, args=(directory,), nprocs=2)


if __name__ == "__main__":
    unittest.main()
