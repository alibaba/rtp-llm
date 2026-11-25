"""Real MLA owner: warm start cannot reconstruct KC/VC from a deleted HF shard."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v3.test.test_nextn_route import (
    _Score,
    fixture,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.resident_weights import ResidentWeights, store


def _run(root, warm):
    config = _Score._create_config(root)
    config.ckpt_path, config.model_type, config.max_seq_len = root, "deepseek3", 128
    config.attn_config.tokens_per_block = config.attn_config.kernel_tokens_per_block = (
        64
    )
    config.data_type = "bf16"
    _Score._post_build_model_config(config)
    load = NewLoaderConfig(
        device="cuda:0",
        compute_dtype=torch.bfloat16,
        load_method="fastsafetensors",
        parallelism_config=ParallelismConfig(),
        moe_config=MoeConfig(),
        py_hw_kernel_config=HWKernelConfig(),
        quant_config=QuantizationConfig("none"),
    )
    manager = ResidentWeights("target", 0, {"owner": "mla"}, False)
    record = manager.acquire_for_load("cuda:0", synchronize_ranks=False)
    assert (record is not None) == warm
    model = NewModelLoader(config, load, model_path=root).load(
        resident=manager, restored=record
    )
    model._ensure_mla_kernel_layout()
    ref = Path(root, "cold.pt")
    values = {}
    for i, layer in enumerate(model.layers):
        att = layer.self_attn
        for key in ("_fused_qkv_a_w", "_kc_w", "_vc_w"):
            value = getattr(att, key)
            assert manager.allocation_of(value) is not None
            values[f"{i}.{key}"] = value.cpu()
    if warm:
        torch.testing.assert_close(
            values, torch.load(ref, weights_only=True), rtol=0, atol=0
        )
    else:
        torch.save(values, ref)
    manager.finalize()


class MlaResidentTest(unittest.TestCase):
    def test_recover_absorb_and_fused_projections(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, data, draft = fixture(True)
            cfg["num_nextn_predict_layers"] = 0
            Path(root, "config.json").write_text(json.dumps(cfg))
            shard = Path(root, "model.safetensors")
            save_file(
                {k: v for k, v in data.items() if not k.startswith(draft)}, str(shard)
            )
            name = "rtp_mla_restart." + uuid.uuid4().hex
            env = dict(os.environ, RTP_LLM_RESIDENT_STORE=name)
            store.ensure_running(name, owner_pid=os.getpid())
            try:
                for phase in ("cold", "warm"):
                    if phase == "warm":
                        shard.unlink()
                    subprocess.run(
                        [sys.executable, __file__, root, phase],
                        env=env,
                        check=True,
                        timeout=120,
                    )
            finally:
                store.stop_store(name)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] in ("cold", "warm"):
        _run(sys.argv[1], sys.argv[2] == "warm")
    else:
        unittest.main()
