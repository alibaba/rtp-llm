"""Shared semantic oracle; load torch-only modules without RTP native imports."""
import json
import importlib.util
from pathlib import Path
import sys

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ATTENTION = Path(__file__).resolve().parents[2]
schedule_module = load("uqi_schedule_test", ATTENTION / "bert_uqi_schedule.py")
core = load("uqi_rocm_core_test", ATTENTION / "rocm_impl/bert_uqi_two_pass_core.py")


def explicit_mask(q, k, v, segments, cu):
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    for lo, hi in zip(cu[:-1], cu[1:]):
        if lo == hi:
            continue
        a, b, c = (x[lo:hi].float() for x in (q, k, v))
        s = segments[lo:hi]
        mask = (s[:, None] == 1) | (s[None, :] == 0)
        logits = torch.einsum("qhd,khd->hqk", a, b) * q.shape[-1] ** -0.5
        probs = logits.masked_fill(~mask[None], -torch.inf).softmax(-1)
        out[lo:hi] = torch.einsum("hqk,khd->qhd", probs, c)
    return out


def cpu_varlen(q, k, v, cq, ck, mq, mk, dropout_p, causal):
    assert not causal and dropout_p == 0
    out = torch.empty_like(q)
    for i in range(len(cq) - 1):
        a, b = int(cq[i]), int(cq[i + 1])
        c, d = int(ck[i]), int(ck[i + 1])
        assert b > a and d > c  # Exercise zero-segment compaction contract.
        logits = torch.einsum("qhd,khd->hqk", q[a:b], k[c:d]) * q.shape[-1] ** -0.5
        out[a:b] = torch.einsum("hqk,khd->qhd", logits.softmax(-1), v[c:d])
    return out


def check_cases(test, device, dtype, attention):
    torch.manual_seed(714)
    for bs in (1, 32):
        for mode in ("none", "middle", "all_b", "mixed", "empty"):
            with test.subTest(batch=bs, mode=mode, dtype=dtype):
                lengths, starts, blens, seg, cu = [], [], [], [], [0]
                for i in range(bs):
                    n = 300 if i == bs - 1 else 7 + i % 17
                    m = ("none", "middle", "all_b", "empty")[i % 4] if mode == "mixed" else mode
                    if m == "empty": n = 0
                    start, blen = (-1, 0) if m in ("none", "empty") else ((0, n) if m == "all_b" else (2, 3))
                    lengths.append(n); starts.append(start); blens.append(blen)
                    seg.extend([int(start <= j < start + blen) if start >= 0 else 0 for j in range(n)])
                    cu.append(cu[-1] + n)
                cp = torch.tensor(cu, dtype=torch.int32)
                segments = torch.tensor(seg, device=device, dtype=torch.int32)
                sched = schedule_module.build_bert_uqi_two_pass_schedule_from_bounds(
                    torch.tensor(starts), torch.tensor(blens), cp, torch.device(device))
                derived = schedule_module.build_bert_uqi_two_pass_schedule(segments, cp.to(device), cp)
                test.assertEqual(sched.has_b, derived.has_b)
                if sched.has_b:
                    torch.testing.assert_close(sched.perm, derived.perm)
                plan = core.prepare_two_pass(sched, torch.device(device))
                tensors = [torch.randn(cu[-1], 4, 128, device=device, dtype=dtype)[..., ::2] for _ in range(3)]
                def run(q, k, v):
                    if sched.has_b: q, k, v = (x.index_select(0, sched.perm) for x in (q, k, v))
                    # Exercise noncontiguous last dimension even after permutation.
                    views = []
                    for x in (q, k, v):
                        buf = torch.empty((*x.shape[:-1], 2 * x.shape[-1]), device=device, dtype=dtype)
                        buf[..., ::2] = x
                        views.append(buf[..., ::2])
                    if mode == "mixed":
                        # Projection-style packed QKV: last stride 1, token stride
                        # includes all three matrices. Exercise the no-copy path.
                        packed = torch.cat([q, k, v], dim=1)
                        views = list(packed.split(q.shape[1], dim=1))
                    out = core.run_two_pass(plan, *views, attention=attention)
                    return out.index_select(0, sched.inv_perm) if sched.has_b else out
                q, k, v = tensors
                out = run(q, k, v)
                ref = explicit_mask(q, k, v, segments, cu)
                error = (out.float()-ref).abs()
                print("CASE", json.dumps(dict(batch=bs, mode=mode, dtype=str(dtype), tokens=cu[-1], max_abs=float(error.max()) if error.numel() else 0.0, mean_abs=float(error.mean()) if error.numel() else 0.0)), flush=True)
                tol = 0.025 if dtype == torch.bfloat16 else (0.003 if dtype == torch.float16 else 2e-6)
                torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)
                # Profile K/V must NEVER affect a QI output; same plan reused.
                kb, vb = k.clone(), v.clone()
                kb[segments == 1] += 9
                vb[segments == 1] -= 7
                changed = run(q, kb, vb)
                torch.testing.assert_close(out[segments == 0], changed[segments == 0], atol=0, rtol=0)
                torch.testing.assert_close(changed.float(), explicit_mask(q, kb, vb, segments, cu), atol=tol, rtol=tol)
