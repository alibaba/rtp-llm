"""ModelSlim (W8A8_MXFP8) weight loading path verification on Ascend NPU
(needs torch_npu).

Run inside the NPU container:
  PYTHONPATH=<repo> python -m pytest rtp_llm/test/npu/test_w8a8_mxfp8_weight_load.py -v -s

Synthetic-ckpt validation of the full static-quantization loading chain:
  AscendW8A8MXFP8Weight._load_raw_tensor (orientation normalization)
    -> _split (no-op at tp=1)
    -> PerBlockFp8Weight._postprocess (NPU branch: real transpose + swizzle)
plus numeric round-trip (E8M0 dequant vs original bf16 weights).
"""

import functools

import pytest
import torch
import torch_npu  # noqa: F401

from rtp_llm.config.quant_config import AscendW8A8MXFP8Config
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.linear_attn_weight import LinearAttnAtomicWeight, LinearAttnConfig
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    merge_qkvz_transpose_reorder,
    transpose_stack_moe_w1,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
)

from test_w8a8_common import (
    MXFP8_GROUP,
    FakeTensorSource,
    cos_sim,
    e8m0_dequant,
    load_and_check,
    make_load_config,
    mx_quantize,
)

torch.manual_seed(123)

HIDDEN = 256  # % 32 == 0
INTER = 128
N_EXPERTS = 4
PREFIX = "model.language_model."


# ---------------- fixtures ----------------


@pytest.fixture(scope="module")
def qc():
    return AscendW8A8MXFP8Config(bits=8, group_size=32, is_quanted=True)


@pytest.fixture(scope="module")
def load_config():
    # num_experts is needed by the MoE cases; harmless for the dense ones
    return make_load_config(HIDDEN, head_num=4, head_num_kv=2,
                            size_per_head=32, num_experts=N_EXPERTS)


@pytest.fixture(scope="module")
def attn_config():
    return AttnConfig(hidden_size=HIDDEN, size_per_head=32,
                      head_num=4, head_num_kv=2)


@pytest.fixture(scope="module")
def moe_config():
    return MoeConfig(expert_num=N_EXPERTS, align_size=0)


@pytest.fixture(scope="module")
def lin_config():
    cfg = LinearAttnConfig.__new__(LinearAttnConfig)
    cfg.linear_num_key_heads = 4
    cfg.linear_num_value_heads = 4
    cfg.linear_key_head_dim = 32
    cfg.linear_value_head_dim = 32
    return cfg


@pytest.fixture(scope="module")
def moe_stacked_ckpt():
    """Synthetic stacked MoE ckpt tensors + the bf16 originals for reference."""
    moe_w1_ckpt = torch.randn(N_EXPERTS, 2 * INTER, HIDDEN, dtype=torch.bfloat16)
    moe_w2_ckpt = torch.randn(N_EXPERTS, HIDDEN, INTER, dtype=torch.bfloat16)

    def quantize_3d(t):
        e, n, kk = t.shape
        kern = torch.empty(e, n, kk, dtype=torch.float8_e4m3fn)
        sc = torch.empty(e, n, kk // MXFP8_GROUP, dtype=torch.uint8)
        for i in range(e):
            kern[i], sc[i] = mx_quantize(t[i])
        return kern, sc

    w1_k, w1_s = quantize_3d(moe_w1_ckpt)
    w2_k, w2_s = quantize_3d(moe_w2_ckpt)
    tensors = {
        PREFIX + "layers.0.mlp.experts.gate_up_proj.weight": w1_k,
        PREFIX + "layers.0.mlp.experts.gate_up_proj.weight_scale": w1_s,
        PREFIX + "layers.0.mlp.experts.down_proj.weight": w2_k,
        PREFIX + "layers.0.mlp.experts.down_proj.weight_scale": w2_s,
    }
    return tensors, moe_w1_ckpt, moe_w2_ckpt


# ---------------- tests ----------------


def test_dense_attn_o_w(qc, load_config, attn_config):
    """Single-tensor dense path (process_fun=transpose)."""
    n_o, k_o = 128, HIDDEN
    w_bf16 = torch.randn(n_o, k_o, dtype=torch.bfloat16)
    kernel, scale = mx_quantize(w_bf16)
    tensors = {
        PREFIX + "layers.0.self_attn.o_proj.weight": kernel,
        PREFIX + "layers.0.self_attn.o_proj.weight_scale": scale,
    }
    weight = AttnAtomicWeight(
        W.attn_o_w,
        [CkptWeightInfo(PREFIX + "layers.{i}.self_attn.o_proj.weight")],
        process_fun=transpose,
        config=attn_config,
    )
    res = load_and_check(weight, qc, FakeTensorSource(tensors), layer_id=0,
                         load_config=load_config, expected={
        W.attn_o_w: ((k_o, n_o), torch.float8_e4m3fn),
        W.attn_o_s: ((k_o // MXFP8_GROUP // 2, n_o, 2), torch.uint8),
    })
    cs = cos_sim(e8m0_dequant(res[W.attn_o_w], res[W.attn_o_s]), w_bf16)
    print(f"[dense attn_o_w] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99


def test_dense_attn_qkv_w(qc, load_config, attn_config):
    """q/k/v merge path (process_fun=merge_qkv_hf)."""
    n_q, n_k, n_v = 128, 64, 64
    q = torch.randn(n_q, HIDDEN, dtype=torch.bfloat16)
    k = torch.randn(n_k, HIDDEN, dtype=torch.bfloat16)
    v = torch.randn(n_v, HIDDEN, dtype=torch.bfloat16)
    qk, qs = mx_quantize(q)
    kk, ks = mx_quantize(k)
    vk, vs = mx_quantize(v)
    tensors = {
        PREFIX + "layers.0.self_attn.q_proj.weight": qk,
        PREFIX + "layers.0.self_attn.q_proj.weight_scale": qs,
        PREFIX + "layers.0.self_attn.k_proj.weight": kk,
        PREFIX + "layers.0.self_attn.k_proj.weight_scale": ks,
        PREFIX + "layers.0.self_attn.v_proj.weight": vk,
        PREFIX + "layers.0.self_attn.v_proj.weight_scale": vs,
    }
    weight = AttnAtomicWeight(
        W.attn_qkv_w,
        [
            CkptWeightInfo(PREFIX + "layers.{i}.self_attn.q_proj.weight"),
            CkptWeightInfo(PREFIX + "layers.{i}.self_attn.k_proj.weight"),
            CkptWeightInfo(PREFIX + "layers.{i}.self_attn.v_proj.weight"),
        ],
        process_fun=merge_qkv_hf,
        config=attn_config,
    )
    n_total = n_q + n_k + n_v
    res = load_and_check(weight, qc, FakeTensorSource(tensors), layer_id=0,
                         load_config=load_config, expected={
        W.attn_qkv_w: ((HIDDEN, n_total), torch.float8_e4m3fn),
        W.attn_qkv_s: ((HIDDEN // MXFP8_GROUP // 2, n_total, 2), torch.uint8),
    })
    w_rec = e8m0_dequant(res[W.attn_qkv_w], res[W.attn_qkv_s])  # [N_total, K]
    ref = torch.cat([q, k, v], dim=0)  # rows: q|k|v
    cs = cos_sim(w_rec, ref)
    print(f"[dense attn_qkv_w] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99


def test_linear_attn_qkvz(qc, load_config, lin_config):
    """Linear attn in_proj_qkvz (cat dim0 + .T)."""
    n_qkv, n_z, k_qkvz = 128, 64, HIDDEN
    qkv = torch.randn(n_qkv, k_qkvz, dtype=torch.bfloat16)
    z_t = torch.randn(n_z, k_qkvz, dtype=torch.bfloat16)
    qkvv, qkvs = mx_quantize(qkv)
    zk, zs = mx_quantize(z_t)
    tensors = {
        PREFIX + "layers.0.linear_attn.in_proj_qkv.weight": qkvv,
        PREFIX + "layers.0.linear_attn.in_proj_qkv.weight_scale": qkvs,
        PREFIX + "layers.0.linear_attn.in_proj_z.weight": zk,
        PREFIX + "layers.0.linear_attn.in_proj_z.weight_scale": zs,
    }
    weight = LinearAttnAtomicWeight(
        W.linear_attn_qkvz_w,
        [
            CkptWeightInfo(PREFIX + "layers.{i}.linear_attn.in_proj_qkv.weight"),
            CkptWeightInfo(PREFIX + "layers.{i}.linear_attn.in_proj_z.weight"),
        ],
        functools.partial(merge_qkvz_transpose_reorder, linear_attention_config=None),
        lin_config,
    )
    n_qkvz = n_qkv + n_z
    res = load_and_check(weight, qc, FakeTensorSource(tensors), layer_id=0,
                         load_config=load_config, expected={
        W.linear_attn_qkvz_w: ((k_qkvz, n_qkvz), torch.float8_e4m3fn),
        W.linear_attn_qkvz_s: ((k_qkvz // MXFP8_GROUP // 2, n_qkvz, 2), torch.uint8),
    })
    cs = cos_sim(e8m0_dequant(res[W.linear_attn_qkvz_w], res[W.linear_attn_qkvz_s]),
                 torch.cat([qkv, z_t], dim=0))
    print(f"[linear_attn_qkvz] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99


def test_moe_w1_stacked(qc, load_config, moe_config, moe_stacked_ckpt):
    """Stacked MoE w1 (transpose_stack_moe_w1 swaps to up|gate)."""
    tensors, moe_w1_ckpt, _ = moe_stacked_ckpt
    weight = MoeAtomicWeight(
        W.moe_w1,
        [CkptWeightInfo(PREFIX + "layers.{i}.mlp.experts.gate_up_proj.weight")],
        process_fun=transpose_stack_moe_w1,
        config=moe_config,
        stacked_ckpt_keys=True,
    )
    res = load_and_check(weight, qc, FakeTensorSource(tensors), layer_id=0,
                         load_config=load_config, expected={
        W.moe_w1: ((N_EXPERTS, HIDDEN, 2 * INTER), torch.float8_e4m3fn),
        W.moe_s1: ((N_EXPERTS, HIDDEN // MXFP8_GROUP // 2, 2 * INTER, 2), torch.uint8),
    })
    w1_rec = torch.empty(N_EXPERTS, 2 * INTER, HIDDEN)
    for i in range(N_EXPERTS):
        w1_rec[i] = e8m0_dequant(res[W.moe_w1][i], res[W.moe_s1][i])
    # transpose_stack_moe_w1 swaps to up|gate; build the same reference
    ref_w1 = torch.cat([moe_w1_ckpt[:, INTER:, :], moe_w1_ckpt[:, :INTER, :]], dim=1)
    cs = cos_sim(w1_rec, ref_w1)
    print(f"[moe_w1 stacked] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99


def test_moe_w2_stacked(qc, load_config, moe_config, moe_stacked_ckpt):
    """Stacked MoE w2 (process_fun=stack_)."""
    tensors, _, moe_w2_ckpt = moe_stacked_ckpt
    weight = MoeAtomicWeight(
        W.moe_w2,
        [CkptWeightInfo(PREFIX + "layers.{i}.mlp.experts.down_proj.weight")],
        process_fun=stack_,
        config=moe_config,
        stacked_ckpt_keys=True,
    )
    res = load_and_check(weight, qc, FakeTensorSource(tensors), layer_id=0,
                         load_config=load_config, expected={
        W.moe_w2: ((N_EXPERTS, INTER, HIDDEN), torch.float8_e4m3fn),
        W.moe_s2: ((N_EXPERTS, INTER // MXFP8_GROUP // 2, HIDDEN, 2), torch.uint8),
    })
    w2_rec = torch.empty(N_EXPERTS, HIDDEN, INTER)
    for i in range(N_EXPERTS):
        w2_rec[i] = e8m0_dequant(res[W.moe_w2][i], res[W.moe_s2][i])
    cs = cos_sim(w2_rec, moe_w2_ckpt)
    print(f"[moe_w2 stacked] shapes OK, dequant cos_sim={cs:.6f}")
    assert cs > 0.99


def test_moe_w1_split(qc, load_config, moe_config):
    """Split (per-expert file) MoE w1 (process_fun=stack_moe_w1), shapes only."""
    tensors = {}
    for e in range(N_EXPERTS):
        for nm in ("gate", "up"):
            t = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16)
            kern, sc = mx_quantize(t)
            tensors[f"{PREFIX}layers.0.mlp.experts.{e}.{nm}_proj.weight"] = kern
            tensors[f"{PREFIX}layers.0.mlp.experts.{e}.{nm}_proj.weight_scale"] = sc
        t2 = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16)
        k2, s2 = mx_quantize(t2)
        tensors[f"{PREFIX}layers.0.mlp.experts.{e}.down_proj.weight"] = k2
        tensors[f"{PREFIX}layers.0.mlp.experts.{e}.down_proj.weight_scale"] = s2
    weight = MoeAtomicWeight(
        W.moe_w1,
        [CkptWeightInfo(PREFIX + "layers.{i}.mlp.experts.{expert_id}.up_proj.weight")]
        + [CkptWeightInfo(PREFIX + "layers.{i}.mlp.experts.{expert_id}.gate_proj.weight")],
        process_fun=stack_moe_w1,
        config=moe_config,
    )
    load_and_check(weight, qc, FakeTensorSource(tensors), layer_id=0,
                   load_config=load_config, expected={
        W.moe_w1: ((N_EXPERTS, HIDDEN, 2 * INTER), torch.float8_e4m3fn),
        W.moe_s1: ((N_EXPERTS, HIDDEN // MXFP8_GROUP // 2, 2 * INTER, 2), torch.uint8),
    })
    print("[moe_w1 split] shapes OK")
