"""Real Qwen3.5-35B-A3B W8A8_MXFP8 weight-level spot checks (Ascend NPU).

Branch-specific supplement to test_w8a8_mxfp8_real_ckpt.py (which is pinned
to the Qwen3-0.6B ModelSlim ckpt, as on main).  This file samples the real
Qwen3.5-35B-A3B-W8A8-MXFP8 ckpt and covers structures unique to 35B:

  * split-format 256-expert MoE down_proj (quantized, NOT anti-outlier
    smoothed -> raw dequant comparison is valid) through the real
    MoeAtomicWeight definition (qwen3_next_weight._create_moe_expert_weights)
  * full-attention ``o_proj`` / linear-attention ``out_proj``: both are
    FLOAT-marked in this ckpt -> bf16 fallback (the #55 exclude-first
    scenario at 35B scale)

The smoothed quantized dense weights (q/k/v/in_proj, expert up/gate) are NOT
raw-comparable (W' = W*diag(s), factor fused into the preceding RMSNorm);
that equivalence is already covered end-to-end by the 0.6B file.

Run inside the NPU container:
  PYTHONPATH=<repo> python -m pytest rtp_llm/test/npu/test_w8a8_mxfp8_real_ckpt_35b.py -v -s
"""

import json
import os

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from rtp_llm.config.quant_config import AscendW8A8MXFP8Config, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.weight_module import WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, stack_, transpose

from test_w8a8_common import (
    MXFP8_GROUP,
    RealTensorSource,
    cos_sim,
    e8m0_dequant,
    load_and_check,
    make_load_config,
)

CKPT = "/mnt/docker/weights/Qwen3.5-35B-A3B-W8A8-MXFP8"
BF16_DIR = "/mnt/docker/weights/Qwen3.5-35B-A3B"

# Qwen3.5-35B-A3B: hybrid layout, every 4th layer is full attention; the
# ckpt uses the "model.language_model." prefix and split (per-expert) MoE.
PREFIX = "model.language_model.layers.{i}."
LAYER_FULL_ATTN = 3  # layer_types[3] == "full_attention"
LAYER_LINEAR = 0  # layer_types[0] == "linear_attention"
HIDDEN = 2048
MOE_INTER = 512
N_EXPERTS = 256
HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256

_BF16_MAP = None


def bf16_ref(name: str) -> torch.Tensor:
    """Load a tensor from the sharded bf16 HF ckpt (CPU, bf16)."""
    global _BF16_MAP
    if _BF16_MAP is None:
        with open(os.path.join(BF16_DIR, "model.safetensors.index.json")) as f:
            _BF16_MAP = json.load(f)["weight_map"]
    fname = _BF16_MAP[name]
    with safe_open(os.path.join(BF16_DIR, fname), framework="pt") as fh:
        return fh.get_tensor(name)


# ---------------- fixtures ----------------


@pytest.fixture(scope="module")
def qc():
    return QuantizationConfig.load_from_ckpt(CKPT)


@pytest.fixture(scope="module")
def tensor_source():
    return RealTensorSource(CKPT)


@pytest.fixture(scope="module")
def load_config():
    return make_load_config(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, num_experts=N_EXPERTS)


@pytest.fixture(scope="module")
def attn_config():
    return AttnConfig(hidden_size=HIDDEN, size_per_head=HEAD_DIM,
                      head_num=HEADS, head_num_kv=KV_HEADS)


@pytest.fixture(scope="module")
def moe_config():
    return MoeConfig(expert_num=N_EXPERTS, align_size=0)


@pytest.fixture(scope="module")
def lin_config():
    cfg = LinearAttnConfig.__new__(LinearAttnConfig)
    return cfg


# ---------------- tests ----------------


def test_config_detection(qc):
    assert isinstance(qc, AscendW8A8MXFP8Config), f"got {type(qc)}"
    assert qc.is_quanted() and qc.group_size() == 32
    # FLOAT-marked 35B modules must be collected as excludes.  Note the GDN
    # (linear-attention) blocks are entirely FLOAT in this ckpt, including
    # in_proj_qkv; only routed experts and full-attn q/k/v are quantized.
    for probe in [
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.mlp.shared_expert.down_proj.weight",
        "lm_head.weight",
        "model.visual.patch_embed.proj.weight",
        "model.language_model.layers.3.input_layernorm.weight",
    ]:
        assert probe in qc.exclude_modules, f"{probe} missing from excludes"
    # quantized modules must NOT be excluded
    for quantized in [
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
    ]:
        assert quantized not in qc.exclude_modules, quantized
    print(f"[config] 35B load_from_ckpt OK (excludes={len(qc.exclude_modules)})")


def test_full_attn_o_proj_float_fallback(qc, tensor_source, load_config, attn_config):
    """L3 (full attention) o_proj is FLOAT-marked -> must load bf16, no scale."""
    weight = AttnAtomicWeight(
        W.attn_o_w,
        [CkptWeightInfo(PREFIX + "self_attn.o_proj.weight")],
        process_fun=transpose,
        config=attn_config,
    )
    qw = WeightModule.create(weight, qc)
    assert type(qw).__name__ == "AscendW8A8MXFP8Weight", type(qw)
    res = qw.load(tensor_source, layer_id=LAYER_FULL_ATTN, device="npu",
                  load_config=load_config)
    assert W.attn_o_w in res, list(res.keys())
    assert W.attn_o_s not in res, "FLOAT layer must not produce a scale tensor"
    got = res[W.attn_o_w]
    assert got.dtype == torch.bfloat16, got.dtype
    assert tuple(got.shape) == (HEADS * HEAD_DIM, HIDDEN), tuple(got.shape)
    ref = bf16_ref(f"model.language_model.layers.{LAYER_FULL_ATTN}.self_attn.o_proj.weight").float()
    cs = cos_sim(got.cpu(), ref.T)  # loaded [K, N] vs bf16 [N, K]
    print(f"[35B o_proj L{LAYER_FULL_ATTN} fallback] bf16, no scale, "
          f"cos_sim vs bf16 ckpt = {cs:.6f}")
    assert cs > 0.999, cs


def test_linear_attn_out_proj_float_fallback(qc, tensor_source, load_config, lin_config):
    """L0 (linear attention) out_proj is FLOAT-marked (#55 scenario).

    out_proj is also an UNVERIFIED layout (not in _VERIFIED_W8A8_MXFP8_WEIGHTS),
    so per the #55 exclude-first semantics ``support()`` must return False and
    WeightModule.create must NOT wrap it -- the weight loads as plain bf16
    through the untouched LinearAttnAtomicWeight (contrast with the verified
    o_proj above, which is wrapped and falls back at load time).
    """
    weight = LinearAttnAtomicWeight(
        W.linear_attn_out_w,
        [CkptWeightInfo(PREFIX + "linear_attn.out_proj.weight", identity)],
        process_fun=transpose,
        config=lin_config,
    )
    qw = WeightModule.create(weight, qc)
    assert type(qw) is LinearAttnAtomicWeight, (
        f"FLOAT + unverified weight must not be wrapped, got {type(qw)}"
    )
    res = qw.load(tensor_source, layer_id=LAYER_LINEAR, device="npu",
                  load_config=load_config)
    assert W.linear_attn_out_w in res, list(res.keys())
    assert W.linear_attn_out_s not in res, "FLOAT layer must not produce a scale"
    got = res[W.linear_attn_out_w]
    assert got.dtype == torch.bfloat16, got.dtype
    n_out = 32 * 128  # linear_num_value_heads * linear_value_head_dim
    assert tuple(got.shape) == (n_out, HIDDEN), tuple(got.shape)
    ref = bf16_ref(f"model.language_model.layers.{LAYER_LINEAR}.linear_attn.out_proj.weight").float()
    cs = cos_sim(got.cpu(), ref.T)
    print(f"[35B out_proj L{LAYER_LINEAR} fallback] not wrapped, bf16, no scale, "
          f"cos_sim vs bf16 ckpt = {cs:.6f}")
    assert cs > 0.999, cs


def test_moe_expert_down_proj_dequant(qc, tensor_source, load_config, moe_config):
    """256-expert split MoE down_proj (quantized, unsmoothed) via the real
    MoeAtomicWeight definition: full chain _load_raw_tensor -> stack_ ->
    _postprocess (3D transpose + E8M0 swizzle), dequant round-trip vs bf16."""
    weight = MoeAtomicWeight(
        W.moe_w2,
        [CkptWeightInfo(PREFIX + "mlp.experts.{expert_id}.down_proj.weight", identity)],
        process_fun=stack_,
        config=moe_config,
        enable_pure_tp_preshard=True,
    )
    res = load_and_check(weight, qc, tensor_source, layer_id=LAYER_FULL_ATTN,
                         load_config=load_config, expected={
        W.moe_w2: ((N_EXPERTS, MOE_INTER, HIDDEN), torch.float8_e4m3fn),
        W.moe_s2: ((N_EXPERTS, MOE_INTER // MXFP8_GROUP // 2, HIDDEN, 2), torch.uint8),
    })
    for e in (0, 1, N_EXPERTS - 1):
        ref_e = bf16_ref(
            f"model.language_model.layers.{LAYER_FULL_ATTN}.mlp.experts.down_proj"
        )[e].float()  # [N=hidden, K=moe_inter]
        rec = e8m0_dequant(res[W.moe_w2][e], res[W.moe_s2][e])  # [N, K]
        cs = cos_sim(rec, ref_e)
        print(f"[35B moe_w2 expert {e}] dequant cos_sim vs bf16 ckpt = {cs:.6f}")
        assert cs > 0.99, (e, cs)
