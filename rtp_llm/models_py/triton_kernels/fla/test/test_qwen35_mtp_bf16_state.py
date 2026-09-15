"""Regression: persistent five-token GDN must match five one-token calls.

Uses production operators with synthetic inputs and the official 397B dimensions.
An exit-zero result validates operator feedback semantics, not model/smoke accuracy.
"""

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import traceback

import torch

from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_update
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter


# Optional provenance check; synthetic operator regression needs no model files.
CONFIG_PATH = os.environ.get("RTP397B_CONFIG")
T = 5
H, HV, D, CONV_WIDTH = 4, 16, 128, 4
QKV = (2 * H + HV) * D
PREFIX = 2775
# LINEAR uses the physical interval, not the FULL attention kernel page size 64.
LINEAR_INTERVAL = 2048
PHYSICAL_STRIDE_BYTES = 2097152  # m1_k0 / BF16 FULL-backed shared pool
READ_COLUMN = (PREFIX - 1) // LINEAR_INTERVAL
WRITE_COLUMN = PREFIX // LINEAR_INTERVAL
BLOCK_IDS = [3, 8, 2, 7, 4]
POOL_BLOCKS = 10
# Same strict normalized-RMS threshold as test_gdn_decode.py; no FLA_CI_ENV bypass.
ORACLE_NRMSE = 0.005
ORACLE_ABS = 1e-6


def checked_dimensions(path):
    # Verified against the official 397B config in the original GPU diagnosis.
    expected = {
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_hidden_layers": 60,
    }
    if path is None:
        return {"path": None, "sha256": None, "text_config_fields": expected,
                "attention_tp": 4, "verification": "embedded_verified_dimensions"}
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    config = data.get("text_config", data)
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise AssertionError(f"397B dimensions mismatch: {actual}; expected {expected}")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "text_config_fields": actual, "attention_tp": 4,
            "verification": "explicit_config_file"}


def digest_tensor(tensor):
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(actual, expected):
    actual, expected = actual.detach().double().cpu(), expected.detach().double().cpu()
    delta = actual - expected
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"finite": False, "max_abs": None, "rms": None,
                "reference_rms": None, "nrmse": None,
                "different_elements": int(torch.count_nonzero(delta)),
                "elements": delta.numel()}
    rms = float(delta.square().mean().sqrt())
    base = float(expected.square().mean().sqrt())
    return {"finite": finite, "max_abs": float(delta.abs().max()),
            "rms": rms, "reference_rms": base, "nrmse": rms / (base + 1e-8),
            "different_elements": int(torch.count_nonzero(delta)),
            "elements": delta.numel()}


def require_oracle(name, metric, errors):
    if not metric["finite"] or not (
        metric["max_abs"] <= ORACLE_ABS or metric["nrmse"] < ORACLE_NRMSE
    ):
        errors.append(f"{name}: independent reference mismatch: {metric}")


def make_pool(h0, conv0, state_dtype):
    raw = torch.zeros((POOL_BLOCKS, PHYSICAL_STRIDE_BYTES // 2),
                      dtype=torch.bfloat16, device="cuda")
    converter = LinearCacheConverter(
        local_num_v_heads=HV, head_v_dim=D, head_k_dim=D,
        ssm_state_dtype=state_dtype, linear_conv_kernel_dim=CONV_WIDTH,
        qkv_size=QKV, conv_state_dtype=torch.bfloat16,
    )
    ssm = converter.get_ssm_state_tensor(raw)
    conv = converter.get_conv_state_tensor(raw).transpose(1, 2)
    ssm[BLOCK_IDS[0]].copy_(h0[0].to(state_dtype))
    conv[BLOCK_IDS[0]].copy_(conv0[0])
    return raw, ssm, conv


def block_table():
    table = torch.zeros((1, WRITE_COLUMN + T), dtype=torch.int32, device="cuda")
    table[0, WRITE_COLUMN:] = torch.tensor(BLOCK_IDS, device="cuda", dtype=torch.int32)
    assert READ_COLUMN == WRITE_COLUMN == 1
    return table


def run_conv(x, weight, h0, conv0, multi):
    raw, _, conv = make_pool(h0, conv0, torch.bfloat16)
    table = block_table()
    outputs, states = [], []
    for start, width in ([(0, T)] if multi else [(i, 1) for i in range(T)]):
        length = torch.tensor([PREFIX + 1 + start], device="cuda", dtype=torch.int32)
        output = causal_conv1d_update(
            x[:, start:start + width].transpose(1, 2), conv, weight,
            bias=None, activation="silu", cache_seqlens=None, block_map=table,
            seq_size_per_block=LINEAR_INTERVAL, sequence_lengths=length,
            validate_data=True,
        )
        outputs.append(output.transpose(1, 2).clone())
        if multi:
            states = [conv[block].clone() for block in BLOCK_IDS]
        else:
            # Each normal decode overwrites the current physical state slot.
            # At prefix 2775..2779 the read/write column remains 1.
            states.append(conv[BLOCK_IDS[0]].clone())
    return torch.cat(outputs, dim=1), torch.stack(states, dim=0).unsqueeze(0)


def conv_reference(x, weight, conv0):
    # Independent CPU FP64 depthwise causal convolution + SiLU. States hold
    # the original BF16 input values, with no arithmetic in the state update.
    x, weight = x.double().cpu(), weight.double().cpu()
    history = conv0.double().cpu().clone()
    outputs, states = [], []
    for token in range(T):
        window = torch.cat((history, x[:, token].unsqueeze(-1)), dim=-1)
        z = (window * weight.unsqueeze(0)).sum(-1)
        outputs.append((z * torch.sigmoid(z)).to(torch.bfloat16))
        history = window[..., 1:].clone()
        states.append(history.to(torch.bfloat16))
    return torch.stack(outputs, dim=1), torch.stack(states, dim=1)


def split_qkv(post_conv):
    packed = post_conv.reshape(1, T, 2 * H + HV, D)
    return torch.split(packed, [H, H, HV], dim=2)


def run_gdn(q, k, v, g, beta, h0, conv0, state_dtype, multi):
    raw, ssm, _ = make_pool(h0, conv0, state_dtype)
    table = block_table()
    outputs, states = [], []
    for start, width in ([(0, T)] if multi else [(i, 1) for i in range(T)]):
        length = torch.tensor([PREFIX + 1 + start], device="cuda", dtype=torch.int32)
        output, _ = fused_recurrent_gated_delta_rule(
            q=q[:, start:start + width], k=k[:, start:start + width],
            v=v[:, start:start + width], g=g[:, start:start + width],
            beta=beta[:, start:start + width], scale=None,
            initial_state=ssm, inplace_final_state=True, block_map=table,
            sequence_lengths=length, seq_size_per_block=LINEAR_INTERVAL,
            use_qk_l2norm_in_kernel=True,
        )
        outputs.append(output.clone())
        if multi:
            states = [ssm[block].clone() for block in BLOCK_IDS]
        else:
            states.append(ssm[BLOCK_IDS[0]].clone())
    return torch.cat(outputs, dim=1), torch.stack(states, dim=0).unsqueeze(0)


def gdn_reference(q, k, v, g, beta, h0, state_dtype, round_between_tokens):
    # Independent CPU FP64 recurrence, V-first. Head grouping and epsilon
    # follow the mathematical operator contract, with no production kernel call.
    q, k, v, g, beta = [value.detach().double().cpu() for value in (q, k, v, g, beta)]
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(HV // H, dim=2) / math.sqrt(D)
    k = k.repeat_interleave(HV // H, dim=2)
    state = h0.double().cpu().clone()
    outputs, states = [], []
    for token in range(T):
        state = state * torch.exp(g[:, token])[..., None, None]
        key = k[:, token]
        value = (v[:, token] - (state * key.unsqueeze(-2)).sum(-1))
        value = value * beta[:, token].unsqueeze(-1)
        state = state + value.unsqueeze(-1) * key.unsqueeze(-2)
        output = (state * q[:, token].unsqueeze(-2)).sum(-1)
        outputs.append(output.to(torch.bfloat16))
        stored = state.to(state_dtype)
        states.append(stored.clone())
        if round_between_tokens:
            state = stored.double()
    return torch.stack(outputs, dim=1), torch.stack(states, dim=1)


def run(report):
    report["model_config"] = checked_dimensions(CONFIG_PATH)
    assert torch.cuda.is_available(), "requires an actual CUDA GPU; CPU/mock execution is unsupported"
    assert torch.cuda.get_device_capability() == (10, 0), "397B diagnosis targets actual SM100"
    report["device"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "name": torch.cuda.get_device_name(), "capability": [10, 0]}
    report["operators"] = {}
    for function in (causal_conv1d_update, fused_recurrent_gated_delta_rule,
                     fused_gdn_gating, LinearCacheConverter):
        path = Path(inspect.getfile(function)).resolve()
        report["operators"][function.__name__] = {
            "source_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report["geometry"] = {
        "batch": 1, "tokens": T, "key_heads": H, "value_heads": HV,
        "key_head_dim": D, "value_head_dim": D, "qkv_width": QKV,
        "conv_width": CONV_WIDTH, "physical_stride_bytes": PHYSICAL_STRIDE_BYTES,
        "linear_interval": LINEAR_INTERVAL, "prefix": PREFIX,
        "read_column": READ_COLUMN, "write_columns": list(range(WRITE_COLUMN, WRITE_COLUMN + T)),
        "multi_write_block_ids": BLOCK_IDS, "single_inplace_block_id": BLOCK_IDS[0],
        "ssm_layout": "V-first; BF16 with physical byte stride", "seed": 42,
    }
    torch.manual_seed(42)
    x = torch.randn((1, T, QKV), device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn((QKV, CONV_WIDTH), device="cuda") * 0.2).to(torch.bfloat16)
    conv0 = torch.randn((1, QKV, CONV_WIDTH - 1), device="cuda", dtype=torch.bfloat16)
    h0 = (torch.randn((1, HV, D, D), device="cuda") * 0.25).to(torch.bfloat16)
    a = torch.randn((T, HV), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((T, HV), device="cuda", dtype=torch.bfloat16)
    alog = torch.full((HV,), -1.5, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros((HV,), device="cuda", dtype=torch.bfloat16)
    g, beta = fused_gdn_gating(alog, a, b, dt_bias)
    g, beta = g.reshape(1, T, HV), beta.reshape(1, T, HV)
    report["input_sha256"] = {name: digest_tensor(value) for name, value in
                              {"pre_conv": x, "conv_weight": weight, "conv_state": conv0,
                               "ssm_state": h0, "g": g, "beta": beta}.items()}
    errors = report["errors"]
    cm, cs = run_conv(x, weight, h0, conv0, True), run_conv(x, weight, h0, conv0, False)
    cr = conv_reference(x, weight, conv0)
    report["conv_per_token"] = []
    for token in range(T):
        entry = {"token": token}
        for label, actual, expected in (
            ("multi_vs_single_output", cm[0][:, token], cs[0][:, token]),
            ("multi_vs_single_state", cm[1][:, token], cs[1][:, token]),
            ("multi_vs_reference_output", cm[0][:, token], cr[0][:, token]),
            ("multi_vs_reference_state", cm[1][:, token], cr[1][:, token]),
        ):
            entry[label] = metrics(actual, expected)
            if label in ("multi_vs_single_output", "multi_vs_single_state", "multi_vs_reference_state"):
                if not entry[label]["finite"] or entry[label]["different_elements"]:
                    errors.append(f"conv boundary must be exact: token {token} {label}")
            else:
                require_oracle(f"conv token {token} {label}", entry[label], errors)
        report["conv_per_token"].append(entry)
    # Both GDN modes consume this exact same post-conv tensor; conv equivalence
    # is separately asserted above. This isolates state rounding from convolution.
    q, k, v = split_qkv(cm[0])
    report["post_conv_sha256"] = digest_tensor(cm[0])
    report["gdn"] = {}
    for dtype_name, state_dtype in (("bf16", torch.bfloat16), ("fp32_control", torch.float32)):
        multi = run_gdn(q, k, v, g, beta, h0, conv0, state_dtype, True)
        single = run_gdn(q, k, v, g, beta, h0, conv0, state_dtype, False)
        repeat = run_gdn(q, k, v, g, beta, h0, conv0, state_dtype, True)
        ref_carry = gdn_reference(q, k, v, g, beta, h0, state_dtype, False)
        ref_round = gdn_reference(q, k, v, g, beta, h0, state_dtype, True)
        rows = []
        for token in range(T):
            row = {"token": token, "approx_first_round_output_index": token + 1}
            for label, actual, expected in (
                ("multi_vs_single_output", multi[0][:, token], single[0][:, token]),
                ("multi_vs_single_state", multi[1][:, token], single[1][:, token]),
                ("repeat_output", repeat[0][:, token], multi[0][:, token]),
                ("repeat_state", repeat[1][:, token], multi[1][:, token]),
                ("multi_vs_carry_reference_output", multi[0][:, token], ref_carry[0][:, token]),
                ("multi_vs_carry_reference_state", multi[1][:, token], ref_carry[1][:, token]),
                ("single_vs_rounded_reference_output", single[0][:, token], ref_round[0][:, token]),
                ("single_vs_rounded_reference_state", single[1][:, token], ref_round[1][:, token]),
                ("multi_vs_rounded_reference_output", multi[0][:, token], ref_round[0][:, token]),
                ("multi_vs_rounded_reference_state", multi[1][:, token], ref_round[1][:, token]),
                ("single_vs_carry_reference_output", single[0][:, token], ref_carry[0][:, token]),
            ):
                row[label] = metrics(actual, expected)
                if label.startswith("repeat_") or label.startswith("multi_vs_single_"):
                    if not row[label]["finite"] or row[label]["different_elements"]:
                        errors.append(f"{dtype_name} token {token} exact equivalence failed: {label}")
                elif label.startswith("multi_vs_rounded_reference") or label.startswith("single_vs_rounded_reference"):
                    # Both modes implement sequential storage-dtype feedback.
                    # Carry-reference distances remain diagnostic only.
                    require_oracle(f"{dtype_name} token {token} {label}", row[label], errors)
                elif not row[label]["finite"]:
                    errors.append(f"{dtype_name} token {token} nonfinite: {label}")
            rows.append(row)
        report["gdn"][dtype_name] = rows
    bf16 = report["gdn"]["bf16"]
    fp32 = report["gdn"]["fp32_control"]
    first_bf16_different = next((row["token"] for row in bf16
                                if row["multi_vs_single_output"]["different_elements"]), None)
    if bf16[0]["multi_vs_single_output"]["different_elements"] or bf16[0]["multi_vs_single_state"]["different_elements"]:
        errors.append("first-token equivalence failed; cannot isolate inter-token BF16 rounding")
    fp32_control_exact = all(not row[key]["different_elements"] for row in fp32
                            for key in ("multi_vs_single_output", "multi_vs_single_state"))
    report["assessment"] = {
        "first_bf16_output_difference_token": first_bf16_different,
        "fp32_control_exact": fp32_control_exact,
        "model_failure_root_cause_proven": False,
        "bf16_inter_token_difference_observed": first_bf16_different is not None and first_bf16_different > 0,
        "interpretation": "persistent multi-token output/state must exactly match sequential decode; model accuracy is unverified",
    }
    report["status"] = "STATE_FEEDBACK_REGRESSION_FAILED" if errors else "STATE_FEEDBACK_REGRESSION_PASS_NOT_MODEL_ACCURACY"


def main():
    report = {"schema": "qwen35-mtp-bf16-state-feedback-v2", "status": "INCOMPLETE", "errors": [],
              "oracle_policy": {"normalized_rms_below": ORACLE_NRMSE, "absolute_bypass_at_most": ORACLE_ABS,
                                "basis": "existing test_gdn_decode.py strict ratio 0.005; independent FP64 CPU reference",
                                "equivalence_checks": "conv boundaries, repeatability and all five GDN output/state pairs require exact equality",
                                "mode_difference": "must be exactly zero for BF16 and FP32 state; independent rounded-reference tolerances unchanged"}}
    try:
        with torch.inference_mode():
            run(report)
    except Exception as error:
        report["status"] = "DIAGNOSTIC_INCOMPLETE"
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.environ.get("TEST_TMPDIR", ".")))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mtp-bf16-state-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    # The unchanged launcher collects test.log; retain the complete report there
    # as well as the Bazel undeclared-output artifact.
    print("MTP_BF16_STATE_REPORT=" + json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    print(f"report_file={path}", flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
