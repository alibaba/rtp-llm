"""Restart the registered decoders in another process, with the HF shard removed."""

import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3_next.test.main_checkpoint_test import (
    _config,
    _dense_weights,
    _load_config,
)
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.resident_weights import ResidentWeights, store


def _weights(model):
    data = _dense_weights(include_lm_head=True)
    if model == "qwen35_dense":
        prefix = "model.language_model.layers.0.linear_attn."
        data = {
            k.replace("model.", "model.language_model.", 1): v
            for k, v in data.items()
            if ".self_attn." not in k
        }
        for name, shape in (
            ("in_proj_qkv.weight", (12, 4)),
            ("in_proj_z.weight", (4, 4)),
            ("in_proj_b.weight", (2, 4)),
            ("in_proj_a.weight", (2, 4)),
            ("conv1d.weight", (12, 1, 2)),
            ("out_proj.weight", (4, 4)),
            ("A_log", (2,)),
            ("dt_bias", (2,)),
            ("norm.weight", (2,)),
        ):
            data[prefix + name] = (
                torch.arange(torch.Size(shape).numel()).float().reshape(shape)
            )
    if model == "qwen_3_moe":
        prefix = "model.layers.0."
        data[prefix + "self_attn.q_proj.weight"] = torch.eye(4)
        for name in ("gate", "up", "down"):
            value = data.pop(prefix + "mlp." + name + "_proj.weight")
            for i in range(2):
                data[prefix + f"mlp.experts.{i}.{name}_proj.weight"] = value + i
        data[prefix + "mlp.gate.weight"] = torch.arange(8).reshape(2, 4).float()
    return data


def _worker(root, model_type, warm):
    gdn = model_type == "qwen35_dense"
    config = _config(
        HybridAttentionType.LINEAR if gdn else HybridAttentionType.NONE, tie=False
    )
    if gdn:
        config.expert_num = config.moe_k = config.moe_inter_size = config.moe_style = 0
    config.model_type = model_type
    config.logit_scale = 2.0
    config.data_type = "bf16"
    if model_type == "qwen_3_moe":
        config.moe_layer_index = [0]
        config.qk_norm = True
        config.n_shared_experts = 0
        config.scoring_func = 0
    manager = ResidentWeights("target", 0, {"owner": model_type}, False)
    record = manager.acquire_for_load("cuda:0", synchronize_ranks=False)
    assert (record is not None) == warm
    method = os.environ["RESIDENT_TEST_METHOD"]
    cfg = replace(
        _load_config(),
        load_method=method,
        device="cuda:0",
        compute_dtype=torch.bfloat16,
        parallelism_config=ParallelismConfig(),
        moe_config=MoeConfig(),
    )
    loader = NewModelLoader(config, cfg, model_path=root)
    model = loader.load(resident=manager, restored=record)
    assert loader._selected_load_method.value == method
    for parameter in (model.lm_head.weight, model.embed_tokens.weight):
        assert manager.allocation_of(parameter) is not None
    torch.testing.assert_close(
        model.lm_head.weight, torch.full_like(model.lm_head.weight, 6), rtol=0, atol=0
    )
    if gdn:
        leaf = model.layers[0].linear_attn
        assert leaf.prefill_gdn is not None and leaf.decode_gdn is not None
        assert manager.allocation_of(leaf.conv1d_w) is not None
        qkv = leaf.in_proj_qkvz
    else:
        qkv = model.layers[0].self_attn.qkv_proj
    x = torch.arange(4, device="cuda", dtype=qkv.weight.dtype).reshape(1, 4)
    output = qkv(x)
    if model_type == "qwen_3_moe":
        expert = model.layers[0].mlp.experts
        assert (
            expert.fused_moe is not None
            and manager.allocation_of(expert.w2) is not None
        )
        torch.testing.assert_close(expert.w2[1], torch.eye(4).to(expert.w2) + 1)
    path = Path(root, "cold-projection.pt")
    if warm:
        torch.testing.assert_close(
            output.cpu(), torch.load(path, weights_only=True), rtol=0, atol=0
        )
    else:
        torch.save(output.cpu(), path)
    manager.finalize()


class ResidentRestartTest(unittest.TestCase):
    def _check(self, model, method):
        with tempfile.TemporaryDirectory() as root:
            name = "rtp_decoder_restart." + uuid.uuid4().hex
            shard = Path(root, "model.safetensors")
            save_file(_weights(model), str(shard))
            env = dict(
                os.environ, RTP_LLM_RESIDENT_STORE=name, RESIDENT_TEST_METHOD=method
            )
            store.ensure_running(name, owner_pid=os.getpid())
            try:
                for phase in ("cold", "warm"):
                    if phase == "warm":
                        shard.unlink()
                    subprocess.run(
                        [sys.executable, __file__, root, model, phase],
                        env=env,
                        check=True,
                        timeout=120,
                    )
            finally:
                store.stop_store(name)

    def test_qwen3_next(self):
        self._check("qwen3_next", "scratch")

    def test_qwen3_next_fst(self):
        self._check("qwen3_next", "fastsafetensors")

    def test_gdn_projection_and_runtime_views(self):
        self._check("qwen35_dense", "fastsafetensors")

    def test_qwen3_moe(self):
        self._check("qwen_3_moe", "scratch")

    def test_qwen3_moe_fst(self):
        self._check("qwen_3_moe", "fastsafetensors")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[-1] in ("cold", "warm"):
        _worker(sys.argv[1], sys.argv[2], sys.argv[3] == "warm")
    else:
        unittest.main()
