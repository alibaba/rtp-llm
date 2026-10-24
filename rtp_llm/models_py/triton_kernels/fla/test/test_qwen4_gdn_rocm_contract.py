"""MI308 Qwen4 TP4 GDN contract; no model/checkpoint allocation.

Calls actual model dispatch with H4/HV12/D128/page128. Independent FP32
recurrence checks output and V-first page state. Conv and FP8 projections are
outside this test; the packed state row tail is checked for corruption.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.models_py.model_desc import qwen3_next as model
from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import Qwen4Sm120GdnPrefill
from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
    Qwen4PrefillGatedNorm,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.fla import aiter_flydsl_gdn_decode as dec
from rtp_llm.models_py.triton_kernels.fla import aiter_flydsl_gdn_prefill as pref

H, HV, D, PAGE = 4, 12, 128, 128
WIDTH = HV * D * D
TAIL = (H * 2 + HV) * D * 4
_ERRORS = []


def strict(name, got, want, ratio=0.015):
    x, y = got.float(), want.float()
    assert torch.isfinite(x).all() and torch.isfinite(y).all(), name
    rms = (x - y).square().mean().sqrt().item()
    rel = rms / max(y.square().mean().sqrt().item(), 1e-8)
    _ERRORS.append(
        {
            "name": name,
            "rms": rms,
            "relative_rms": rel,
            "max_abs": (x - y).abs().max().item(),
        }
    )
    assert rms < 1e-5 or rel < ratio, (name, rms, rel, (x - y).abs().max().item())


def reference(mixed, a, b, alog, dt, state):
    """Scalar-in-time FP32 recurrence, V-first state, BF16 normalized q/k."""
    q, k, v = mixed.reshape(-1, H * 2 + HV, D).split([H, H, HV], dim=1)
    q = (
        (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6))
        .to(q.dtype)
        .float()
        .repeat_interleave(HV // H, 1)
    )
    k = (
        (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6))
        .to(k.dtype)
        .float()
        .repeat_interleave(HV // H, 1)
    )
    g = -alog.float().exp() * F.softplus(a.float() + dt.float())
    beta = b.float().sigmoid()
    state = state.float().clone()
    outs, checkpoints = [], {}
    for t in range(mixed.shape[0]):
        state = state * g[t].exp()[:, None, None]
        residual = (v[t].float() - (state * k[t, :, None, :]).sum(-1)) * beta[
            t, :, None
        ]
        state = state + residual[:, :, None] * k[t, :, None, :]
        outs.append((state * q[t, :, None, :]).sum(-1) * (D**-0.5))
        if (t + 1) % PAGE == 0 or t == mixed.shape[0] - 1:
            checkpoints[t + 1] = state.clone()
    return torch.stack(outs), checkpoints


def pool(blocks, dtype):
    packed = torch.randn(blocks, WIDTH + TAIL, dtype=dtype, device="cuda") * 0.01
    return packed, packed[:, :WIDTH].view(blocks, HV, D, D)


def clear():
    pref._get_aiter_flydsl_gdn_prefill_ops.cache_clear()
    pref._is_aiter_flydsl_gdn_prefill_disabled.cache_clear()
    dec._get_aiter_flydsl_gdn_decode.cache_clear()
    dec._is_aiter_flydsl_gdn_decode_disabled.cache_clear()


@unittest.skipUnless(
    torch.version.hip and torch.cuda.is_available(), "ROCm GPU required"
)
class Qwen4GdnRocmContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(257)
        _ERRORS.clear()
        clear()
        self.addCleanup(clear)

    def tearDown(self):
        for name in sorted({item["name"] for item in _ERRORS}):
            values = [item for item in _ERRORS if item["name"] == name]
            print(
                json.dumps(
                    {
                        "metric": name,
                        "comparisons": len(values),
                        "max_relative_rms": max(x["relative_rms"] for x in values),
                        "max_abs": max(x["max_abs"] for x in values),
                        "relative_rms_limit": 0.015,
                    }
                ),
                flush=True,
            )

    def test_prefill_resume_checkpoint_default_and_triton(self):
        for dtype in (torch.float32, torch.bfloat16):
            for fallback in (False, True):
                with self.subTest(dtype=dtype, triton=fallback), mock.patch.dict(
                    os.environ,
                    {"DISABLE_AITER_FLYDSL_GDN_PREFILL": "1" if fallback else "0"},
                ):
                    clear()
                    lengths, prefixes = (2048, 129), (0, 128)
                    mixed = torch.randn(
                        sum(lengths),
                        (H * 2 + HV) * D,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    ba = torch.randn(
                        sum(lengths), HV * 2, dtype=torch.bfloat16, device="cuda"
                    )
                    a, b = ba.split(HV, -1)
                    alog = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
                    dt = torch.randn_like(alog)
                    packed, states = pool(23, dtype)
                    before = packed.clone()
                    table = torch.tensor(
                        [list(range(1, 18)) + [0], [18, 19, 20] + [0] * 15],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    cu = torch.tensor(
                        [0, lengths[0], sum(lengths)], dtype=torch.int32, device="cuda"
                    )
                    inputs = SimpleNamespace(
                        input_lengths=torch.tensor(lengths),
                        prefix_lengths=torch.tensor(prefixes),
                        prefix_lengths_device=torch.tensor(
                            prefixes, dtype=torch.int32, device="cuda"
                        ),
                        kv_cache_kernel_block_id_device=table,
                        cu_seqlens_device=cu,
                        is_cuda_graph=False,
                    )
                    metadata = (
                        pref.build_aiter_flydsl_gdn_prefill_metadata(lengths, cu)
                        if pref.is_aiter_flydsl_gdn_prefill_available()
                        else None
                    )
                    meta = SimpleNamespace(
                        get_aiter_gdn_prefill_metadata=lambda _: metadata
                    )
                    owner = Qwen4Sm120GdnPrefill.__new__(Qwen4Sm120GdnPrefill)
                    torch.nn.Module.__init__(owner)
                    owner.local_num_k_heads, owner.local_num_v_heads = H, HV
                    owner.head_k_dim = owner.head_v_dim = D
                    owner.ssm_state_dtype = dtype
                    owner.alog, owner.dt_bias = alog, dt
                    owner._get_ssm_states = lambda _: states
                    touched = set()
                    expected_outputs = []
                    offset = 0
                    for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
                        initial = (
                            torch.zeros(HV, D, D, device="cuda")
                            if prefix == 0
                            else states[int(table[row, (prefix - 1) // PAGE])].clone()
                        )
                        out, checks = reference(
                            mixed[offset : offset + length],
                            a[offset : offset + length],
                            b[offset : offset + length],
                            alog,
                            dt,
                            initial,
                        )
                        expected_outputs.append(out)
                        for end, h in checks.items():
                            index = int(table[row, (prefix + end - 1) // PAGE])
                            touched.add(index)
                            # Keep expected page states independent of production stores.
                            before[index, :WIDTH] = h.to(dtype).flatten()
                        offset += length
                    with mock.patch.object(
                        model,
                        "chunk_gated_delta_rule",
                        wraps=model.chunk_gated_delta_rule,
                    ) as tri, mock.patch.object(
                        model,
                        "chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states",
                        wraps=model.chunk_gated_delta_rule_aiter_flydsl_with_intermediate_states,
                    ) as ait:
                        actual = owner._fla(mixed, b, a, packed, PAGE, inputs, meta)
                    self.assertEqual(tri.call_count + ait.call_count, 1)
                    if fallback:
                        self.assertEqual(tri.call_count, 1)
                    else:
                        self.assertEqual(ait.call_count, 1)
                    strict("prefill output", actual, torch.cat(expected_outputs))
                    for index in touched:
                        strict(
                            "prefill checkpoint",
                            states[index],
                            before[index, :WIDTH].view(HV, D, D),
                        )
                    untouched = sorted(set(range(states.shape[0])) - touched)
                    torch.testing.assert_close(
                        packed[untouched], before[untouched], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        packed[:, WIDTH:], before[:, WIDTH:], rtol=0, atol=0
                    )
                    # Continue each request from the page state just produced,
                    # including the first transition into a new page.
                    step = torch.randn(
                        2, (H * 2 + HV) * D, device="cuda", dtype=torch.bfloat16
                    )
                    ab = torch.randn(2, HV * 2, device="cuda", dtype=torch.bfloat16)
                    da, db = ab.split(HV, -1)
                    prev = [p + n for p, n in zip(prefixes, lengths)]
                    di = SimpleNamespace(
                        kv_cache_kernel_block_id_device=table,
                        kv_cache_kernel_block_id=table.cpu(),
                        sequence_lengths_plus_1_device=torch.tensor(
                            [n + 1 for n in prev], device="cuda", dtype=torch.int32
                        ),
                        sequence_lengths=torch.tensor(prev, dtype=torch.int32),
                    )
                    dm = SimpleNamespace(
                        aiter_flydsl_gdn_decode_indices={},
                        aiter_flydsl_gdn_decode_unsupported=set(),
                        is_cuda_graph=False,
                    )
                    decoder = SimpleNamespace(
                        local_num_k_heads=H,
                        local_num_v_heads=HV,
                        head_k_dim=D,
                        head_v_dim=D,
                        alog=alog,
                        dt_bias=dt,
                        _is_target_verify=lambda *_: False,
                        _get_bs_from_attenion_input=lambda *_: (2, 1),
                        _get_ssm_states=lambda _: states,
                        _get_fla_block_map=lambda _: table,
                    )
                    continuation = []
                    for row, n in enumerate(prev):
                        read = int(table[row, (n - 1) // PAGE])
                        ref, h = reference(
                            step[row : row + 1],
                            da[row : row + 1],
                            db[row : row + 1],
                            alog,
                            dt,
                            states[read],
                        )
                        continuation.append((ref[0], h[1], int(table[row, n // PAGE])))
                    dout = model.Qwen3NextGatedDeltaNetDecode._fla(
                        decoder, step, db, da, packed, PAGE, di, dm
                    )
                    for row, (ref, h, write) in enumerate(continuation):
                        strict("prefill to decode output", dout[row], ref)
                        strict("prefill to decode state", states[write], h.to(dtype))
                    print(
                        json.dumps(
                            {
                                "phase": "prefill",
                                "shape": [sum(lengths), H, HV, D],
                                "page": PAGE,
                                "state": str(dtype),
                                "forced_triton": fallback,
                                "backend": "aiter" if ait.call_count else "triton",
                                "checkpoint_blocks": len(touched),
                            }
                        ),
                        flush=True,
                    )

    def test_changed_input_decode_graph_default_and_triton(self):
        for dtype in (torch.float32, torch.bfloat16):
            for fallback in (False, True):
                with self.subTest(dtype=dtype, triton=fallback), mock.patch.dict(
                    os.environ,
                    {"DISABLE_AITER_FLYDSL_GDN_DECODE": "1" if fallback else "0"},
                ):
                    clear()
                    batch = 3
                    mixed = torch.randn(
                        batch, (H * 2 + HV) * D, device="cuda", dtype=torch.bfloat16
                    )
                    ba = torch.randn(batch, HV * 2, device="cuda", dtype=torch.bfloat16)
                    a, b = ba.split(HV, -1)
                    alog = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
                    dt = torch.randn_like(alog)
                    packed, states = pool(9, dtype)
                    pristine = packed.clone()
                    table = torch.tensor(
                        [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    host_table = table.cpu()
                    lens = torch.tensor([128, 256, 0], device="cuda", dtype=torch.int32)
                    host_lens = torch.tensor([127, 255, 0], dtype=torch.int32)
                    inputs = SimpleNamespace(
                        kv_cache_kernel_block_id_device=table,
                        kv_cache_kernel_block_id=host_table,
                        sequence_lengths_plus_1_device=lens,
                        sequence_lengths=host_lens,
                    )

                    def metadata(graph=False):
                        return SimpleNamespace(
                            aiter_flydsl_gdn_decode_indices={},
                            aiter_flydsl_gdn_decode_unsupported=set(),
                            is_cuda_graph=graph,
                        )

                    owner = SimpleNamespace(
                        local_num_k_heads=H,
                        local_num_v_heads=HV,
                        head_k_dim=D,
                        head_v_dim=D,
                        alog=alog,
                        dt_bias=dt,
                        _is_target_verify=lambda *_: False,
                        _get_bs_from_attenion_input=lambda *_: (batch, 1),
                        _get_ssm_states=lambda _: states,
                        _get_fla_block_map=lambda _: table,
                    )
                    norm = Qwen4PrefillGatedNorm(
                        RmsNormGated(
                            torch.randn(D, device="cuda", dtype=torch.bfloat16),
                            group_size=D,
                            activation="sigmoid",
                        ),
                        role="PDFUSION",
                    )
                    gate = torch.randn(
                        batch, HV * D, device="cuda", dtype=torch.bfloat16
                    )

                    def run(meta):
                        out = model.Qwen3NextGatedDeltaNetDecode._fla(
                            owner, mixed, b, a, packed, PAGE, inputs, meta
                        )
                        return out, norm(out.reshape(batch, HV * D), gate)

                    with mock.patch.object(
                        model,
                        "aiter_flydsl_gdn_decode",
                        wraps=model.aiter_flydsl_gdn_decode,
                    ) as ait, mock.patch.object(
                        model,
                        "fused_recurrent_gated_delta_rule",
                        wraps=model.fused_recurrent_gated_delta_rule,
                    ) as tri:
                        run(metadata())
                    self.assertEqual(ait.call_count + tri.call_count, 1)
                    if fallback:
                        self.assertEqual(tri.call_count, 1)
                    else:
                        self.assertEqual(ait.call_count, 1)
                    torch.cuda.synchronize()
                    packed.copy_(pristine)
                    graph = torch.cuda.CUDAGraph()
                    gm = metadata(True)
                    with torch.cuda.graph(graph):
                        graph_out, graph_norm = run(gm)
                    packed.copy_(pristine)
                    # Change values, block-table contents, page-boundary lengths,
                    # and norm gate while retaining captured storage addresses.
                    mixed.mul_(0.73)
                    a.add_(0.2)
                    b.sub_(0.3)
                    gate.mul_(0.4)
                    table[0, 1] = 7
                    table[1, 2] = 8
                    host_table.copy_(table.cpu())
                    lens.copy_(
                        torch.tensor([129, 257, 0], device="cuda", dtype=torch.int32)
                    )
                    host_lens.copy_(torch.tensor([128, 256, 0], dtype=torch.int32))
                    expected = pristine.clone()
                    expected_out = torch.zeros(batch, HV, D, device="cuda")
                    for row, read, write in ((0, 1, 7), (1, 5, 8)):
                        ref, h = reference(
                            mixed[row : row + 1],
                            a[row : row + 1],
                            b[row : row + 1],
                            alog,
                            dt,
                            pristine[read, :WIDTH].view(HV, D, D),
                        )
                        expected_out[row] = ref[0]
                        expected[write, :WIDTH] = h[1].to(dtype).flatten()
                    eager_out, eager_norm = run(metadata())
                    eager_state = packed.clone()
                    packed.copy_(pristine)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(graph_out, eager_out, rtol=0, atol=0)
                    torch.testing.assert_close(graph_norm, eager_norm, rtol=0, atol=0)
                    torch.testing.assert_close(packed, eager_state, rtol=0, atol=0)
                    strict("decode output", graph_out, expected_out)
                    nx = expected_out.to(torch.bfloat16).float()
                    norm_ref = nx * torch.rsqrt(
                        nx.square().mean(-1, keepdim=True) + norm.eps
                    )
                    norm_ref *= norm.weight.float()
                    norm_ref *= gate.reshape(batch, HV, D).float().sigmoid()
                    strict("gated norm", graph_norm.reshape(batch, HV, D), norm_ref)
                    for index in (7, 8):
                        strict(
                            "decode state",
                            packed[index, :WIDTH],
                            expected[index, :WIDTH],
                        )
                    for index in (0, 1, 2, 3, 4, 5, 6):
                        torch.testing.assert_close(
                            packed[index], pristine[index], rtol=0, atol=0
                        )
                    torch.testing.assert_close(
                        packed[:, WIDTH:], pristine[:, WIDTH:], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        graph_out[2], torch.zeros_like(graph_out[2]), rtol=0, atol=0
                    )
                    print(
                        json.dumps(
                            {
                                "phase": "decode_graph_changed_input",
                                "shape": [batch, H, HV, D],
                                "page": PAGE,
                                "state": str(dtype),
                                "forced_triton": fallback,
                                "backend": "aiter" if ait.call_count else "triton",
                                "padding": 1,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
