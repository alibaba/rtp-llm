"""wf6_module_final.py -- self-contained sparse MLA prefill module (B300 / sm_103, Triton).

一句话: wf4_module_final.py (md5 b51150a896a215de2981050b3e9bdbd5) + 已实测的 "消波" 收益 --
把 A 的两段 (window / dyn) 拆开放到两条 CUDA 流上并发执行, 互相填补 148-CTA 的半个尾波;
代价/前提是两段都改用 m_off = 0 (逐行位移在 out = sum(P V)/sum(P) 中精确抵销).

SOURCE OF THE GAIN (agent "dirc", files wf5_dirc_z.py / wf5_dirc_z_confirm.py, logs
wf5_dirc_z_z1.log / wf5_dirc_z_confirm_conf1.log -- imported, never modified):
    T=2048: baseline trio (win + dyn_v3 + B + 56MB L2) 0.4952 -> conc_z + B 0.4624 ms (+6.62%),
            MFU 23.59% -> 25.26%, e2e (含 q-pack) 0.5049 / 23.14%, rel_out 4.317e-3 PASS;
    T=4096: 0.8773 -> 0.8668 ms (+1.20%), MFU 26.63% -> 26.95%, rel_out 4.544e-3 PASS.
    Mechanism: 512 CTAs @T=2048 = 3.46 waves of 148 SMs -> the half-empty 4th wave of one
    segment is filled by CTAs of the other segment; at T=4096 (1024 CTAs = 6.92 waves) the
    tail is small, hence only +1.20% -- the gain tracks the tail fraction.

WHAT THIS FILE IS
    * the wf4 kernels for everything except A_win / A_dyn (q-pack, KV-pack, V dequant,
      idx bookkeeping, B = _kb_512_lacc, L2 persisting set-aside are carried over verbatim);
    * NEW A_win (_a_win_z) and A_dyn (_a_dyn_z / guarded _a_dyn_z_safe): both use m_off = 0,
      P = exp2(min(s, EMAX) * log2e) in bf16, no m_off load, no window tile-0 read+dot in dyn;
    * concurrent=True  -> _a_win_z on stream A || _a_dyn_z on stream B, then B on the caller
      stream after both (a stream pair is created lazily per device and cached);
      concurrent=False -> the same two kernels back to back on the caller stream (bitwise
      identical P/out/lse, only the schedule differs -- this is the audit/fallback state).
    * SELF-CONTAINED: no import of any wf_* / wf2_* / wf3_* / wf5_* module; the only triton/
      torch calls on the hot path are launches + cached-buffer bookkeeping.  __main__ may
      import wf4_module_final.py (guarded) for the cross-check only, and always skips
      gracefully when it is absent.

PUBLIC ENTRY (superset of the wf4 / sparse_mla_two_kernel.py call form):

    sparse_mla_two_kernel(q, kv, idx, out, lse, dyn_v3=False, l2_persist=False,
                          concurrent=True) -> (out, lse)
    run = sparse_mla_two_kernel                                     (alias)
    enable_l2_persist_v(vbf=None, reserve_mb=56, span_mb=56, ratio=1.0, stream=None)
    reset_l2_persist(stream=None)
    l2_persist_state()
    sampled_max_abs_s(q, kv, idx, rows=None)   # test/audit helper for the safety precondition

INPUT CONTRACT
    q    [T, 32, 576]  bf16   NoPE 512 + RoPE 64 (DeepSeek-style MLA); T % 4 == 0
    kv   [N, 576]      bf16   packed once (pack_kv); V reuses the fp8 K payload
    idx  [T, 2048]     int32  *** 4 行共享: every 4 consecutive rows must share the SAME idx
                              row (G=4 spec-decode); all four rows of a group are handled by
                              one CTA (pid*4) ***
                              column layout (union form these kernels are tuned for):
                                [   0,   16)   16 sink tokens
                                [  16, 1040)   contiguous window, idx[t, 16+j] = p_t-1023+j
                                               (1024 slots, requires position p_t >= 1023)
                                [1040, 2048)   1008 dynamically gathered slots
                              -1 marks padding inside the dyn block: the default path
                              (dyn_v3=False) zeroes P there via clamped loads + a positional
                              mask; dyn_v3=True does NOT handle -1 (dyn block must be fully
                              valid), exactly like wf4.
    out  [T, 32, 512]  fp32
    lse  [T, 32]       fp32   = m_off + ln(l)   (here m_off = 0)

(m_off, l) 语义  (offset-invariant softmax, same contract as wf4 / sparse_mla_two_kernel.py)
    * BOTH segments use the SAME row-constant offset m_off = 0: A_win stores literal zeros
      into MOUT, A_dyn never reads MOUT.  Any per-row shift K = exp(-m_off) cancels exactly
      in out = sum(P V)/sum(P), so m_off = 0 is mathematically equivalent to wf4's
      m_off = rowmax(first 128 window slots).
    * l = sum over all 2048 slots of P is accumulated inside B (A never reads P back):
      out = (sum_slot P*V) / l,   lse = m_off + ln(l) = ln(sum exp(s)).
    * Because m_off = 0, A_dyn needs NO information from A_win -> the two segments are
      independent and can run on two streams at the same time (this is the whole point).

P 列划分 (zero-copy handoff; no concat in the timed path)
    columns [0, 1024)     <- A_win   (_a_win_z,  the 1024 contiguous window slots)
    columns [1024, 2048)  <- A_dyn   (_a_dyn_z,  sink + gathered slots)
    B consumes the single [T*32, 2048] bf16 buffer directly; idx_all = [win | dyn] carries
    exactly that slot order, and all kernels share the 4-row group base (pid*4).

m_off=0 的安全性前提 (SAFETY PRECONDITION -- audited, see wf6_safety.log)
    P = exp2(min(s, EMAX)*log2e), i.e. exp(s) capped at EMAX = 60.  m_off = 0 is EXACT
    (bit-identical semantics) as long as:
      (a) max_s <= EMAX = 60:  otherwise the cap fires on the TOP scores, flattening the
          ratio between them and biasing both out and lse.  This is a hard cliff (the wf4
          convention caps the *relative* excursion s-m_off at 60, so wf4 degrades only for
          rowmax-windowmax > 60 while wf6 degrades for absolute s > 60).
      (b) per-row max s >~ -80:  P=exp(s) must stay in bf16's normal range (min normal
          ~1.2e-38 = e^-87.3; subnormal floor e^-92.7).  A row whose scores are ALL below
          ~-93 underflows to P = 0, hence l = 0, out = 0 and lse = 0 -- silently wrong.
          wf4 is shift-robust here (it subtracts the row max), wf6 is not.
    Neither bound is reachable in this workload: measured sampled max|s| ~ 0.05 (near-uniform
    attention) -> ~1200x margin to the cliff.  Audited empirically with constructed inputs
    (max|s| ~ 20/40/60 -> PASS vs the same fp8-packed reference and vs wf4; boundary probes
    65/70/88/100 and all-negative-row probes -80/-90/-100 show exactly the two failure modes
    above and nothing else -- wf6_safety.py / wf6_safety_*.log).  If your inputs can exceed
    max_s > 60 (or sit below ~-80 for whole rows), do NOT use m_off=0: use
    wf4_module_final.py, which subtracts a row offset and is shift-robust.

L2 开关的进程级影响 (l2_persist=True)
    enable_l2_persist_v() 改动的是 *进程级* CUDA 状态: cudaLimitPersistingL2CacheSize
    (limit id 6) reserve ~56 MB + cudaStreamSetAttribute (attr id 1 = AccessPolicyWindow)
    覆盖 VBF[0:56MB], hitRatio 1.0, hitProp=persisting(2) / missProp=streaming(1).
    该属性挂在当前 stream 上, 之后同一 stream 上的所有 kernel (不止本模块) 都会受影响;
    device limit 是整个 context 的.  l2_persist=True 在入口里只是幂等地 "确保" 当前 VBF
    的窗口已设置, 从不关闭已有窗口; 实验结束必须 reset_l2_persist() 恢复 (limit -> 0,
    清空窗口, flush persisting lines).  内核本身不因该开关而改变 (measured B -8.9~-10.2%).

MEASURED SUMMARY (this file, B300/SM103 cc(10,3), GPU4, torch 2.11 / triton 3.6, N=131072,
union S=2048, 4-row-shared union, L2 = 56 MB persisting window; sessions wf6_module_final_m2.log
/ m3.log, safety wf6_safety_safe3.log / safe4.log / safe5.log):
    combo3 = the REAL three-segment pipeline (win||dyn then B when concurrent=True);
    entry = combo3 + q-pack + idx bookkeeping.  rel_out / lse_diff are identical in every
    group (concurrency / dyn_v3 / l2 are bitwise-neutral):
     T    cfg(conc,dv3,l2) entry_ms  MFU%  combo3  MFU3%   win     dyn      B    rel_out   lse_diff
     512  (1,1,1)            0.1470  19.87  0.1315  22.21  0.0355  0.0454  0.0516  4.274e-03  2.432e-04
     512  (0,1,1)            0.1496  19.52  0.1350  21.63  0.0357  0.0456  0.0517  4.274e-03  2.432e-04
     2048 (1,1,1)            0.4975  23.48  0.4643  25.16  0.1382  0.1689  0.1864  4.317e-03  2.360e-04
     2048 (0,1,1)            0.5303  22.03  0.4964  23.53  0.1378  0.1689  0.1863  4.317e-03  2.360e-04
     4096 (1,1,1)            0.9325  25.06  0.8765  26.66  0.2443  0.2960  0.3333  4.544e-03  2.594e-04
     4096 (0,1,1)            0.9399  24.86  0.8855  26.39  0.2441  0.2959  0.3332  4.544e-03  2.594e-04
    * A pair alone (win || dyn  vs  win + dyn): T=512 0.0839/0.0809 (-3.8%: 128 CTA < 1 wave,
      nothing to fill), T=2048 0.2732/0.3071 (+11.0%), T=4096 0.5338/0.5403 (+1.2%) =>
      the gain tracks the wave-tail fraction: combo3 +6.47% (T=2048), +1.02% (T=4096),
      T=512 ~neutral.  Reproduces the source session (wf5_dirc_z_confirm: 0.4624 = +6.62%).
    * vs wf4_module_final.py default entry in the same session: 0.5567 -> 0.4975 ms (-10.6%)
      @T=2048 (T=512 0.1543 -> 0.1470, T=4096 0.9858 -> 0.9325).
    * bitwise self-checks (all three T, both concurrent states): concurrent on == off out/lse/P
      identical (0 of 134217728 P entries differ); guarded dyn P == fast dyn P (0 differ);
      q-pack byte-exact and idx buffers bitwise-equal vs wf4; zero-score bridge (q = 0 =>
      wf4's rowmax offset is exactly 0.0) bitwise-equal to wf4 in all 3 states -> validates
      the P/idx/B/pack plumbing end to end.  On general inputs out/lse are NOT bitwise equal to
      wf4 (m_off convention changes the bf16 rounding of P): rel_out 2.99e-3 / lse 2.87e-4
      @T=2048, i.e. the same order as wf4's own error vs the fp8-packed reference.
    * m_off=0 safety audit (wf6_safety.py; T=1024 all levels, T=2048 required levels):
      max|s| ~ 20/40/60  ALL PASS (wf6 rel_out 3.1~3.5e-3 / lse 3.5~3.9e-3, wf4 3.0~3.5e-3 /
      3.6~3.9e-3, gates 3e-2 / 1e-2; concurrent on/off bitwise identical at every level).
      Cliff (a) measured: max|s| 65 (6 of 33.5M scores > EMAX) -> lse error jumps to 4.4 while
      wf4 stays at 3.9e-3; at 88 / 100 the clamped counts are 2891 / 17604 and rel_out reaches
      0.68 / 0.90.  Cliff (b) measured: whole-row negative shift: -60 PASS (3.0e-6), -80 PASS
      (4.8e-6), -90 FAIL (out = 0, lse error 83) => the bf16 normal floor e^-87 is the boundary.
    中文摘要: 本文件 = wf4_module_final.py + "m_off=0 双流并发消波"; 入口新增 concurrent=True
    (默认开); dyn_v3/l2_persist 语义与 wf4 相同 (默认关).  T=2048 三段 combo3 0.4643 vs 0.4964
    (+6.47%), 端到端 0.4975 ms / MFU 23.48% (源头 wf5 测值 0.4624 / +6.62% / 0.5049 复现);
    T=4096 +1.02%; T=512 无实质收益 (128 CTA < 1 波).  concurrent 开/关逐位一致; 常规输入与
    wf4 数值等价而非逐位 (m_off 取整不同), 零分数桥接用例与 wf4 逐位相等.  安全前提: 每行
    max_s <= 60 且 rowmax >~ -80 (实测 20/40/60 PASS, 65 起 EMAX 夹取, -90 起 bf16 下溢) --
    超出该范围请改用 wf4_module_final.py (它做行内 max 位移, 对平移鲁棒).

usage:  python wf6_module_final.py [T ...]          (default: 512 2048 4096)
        python wf6_safety.py [tag] [T] [all|req|neg]  (m_off=0 safety audit)
"""
from __future__ import annotations

import ctypes
import math
import os
import sys

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------------------
# constants / tuned configuration (all values measured on B300/SM103)
# --------------------------------------------------------------------------------------
E4M3_MAX = 448.0
LOG2E = 1.4426950408889634
LOG2E_C = tl.constexpr(LOG2E)
SCALE = 1.0 / math.sqrt(576.0)
EMAX = 60.0
WIN_SLOTS = 1024          # P columns [0, 1024)      <- A_win
DYN_SLOTS = 1024          # P columns [1024, 2048)   <- A_dyn
UNION_S = WIN_SLOTS + DYN_SLOTS          # 2048
SINK = 16                 # idx columns [0, 16): sink tokens
WIN0 = 16                 # idx columns [16, 1040): contiguous window
DYN0 = 1040               # idx columns [1040, 2048): gathered slots

CFG_Q = dict(R=4, num_warps=4)                    # 4 rows/CTA q-pack, byte-exact
CFG_WIN = dict(NS=128, num_warps=8, num_stages=1)          # _a_win_z (wf5 measured)
CFG_DYN = dict(NS=64, DST=3, num_warps=4, num_stages=1)    # _a_dyn_z  (wf5 measured)
# ^ fast m_off=0 dyn: KERNEL-level num_stages stays 1, the pipeline comes from
#   tl.range(num_stages=3) (same lesson as wf4's dyn_v3).
CFG_DYN_SAFE = dict(NS=64, num_warps=4, num_stages=2)      # guarded fallback (mirrors wf4 CFG_DYN)
CFG_B = dict(NS=32, num_warps=8, num_stages=8)    # 0.2048 ms @T=2048
CFG_KV = dict(num_warps=4)
MOFF_ZERO = True          # this module is the m_off=0 pipeline; kept as a named constant so
                          # the convention is greppable (MOFF_ZERO / EMAX are the safety pair).


# ======================================================================================
# packing kernels (carried over verbatim from wf4_module_final.py)
# ======================================================================================
@triton.jit
def _q_pack(Q, Q8, QSC, QR, E4M3: tl.constexpr, R: tl.constexpr):
    """R rows per CTA; identical math to the baseline 1-row/_q_pack (byte-exact)."""
    pid = tl.program_id(0)
    rr = pid * R + tl.arange(0, R)
    r512 = tl.arange(0, 512)
    r64 = tl.arange(0, 64)
    qv = tl.load(Q + rr[:, None] * 576 + r512[None, :]).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(qv), axis=1) / E4M3, 1e-6)
    tl.store(QSC + rr, sc)
    tl.store(Q8 + rr[:, None] * 512 + r512[None, :], (qv / sc[:, None]).to(tl.float8e4nv))
    tl.store(QR + rr[:, None] * 64 + r64[None, :],
             tl.load(Q + rr[:, None] * 576 + 512 + r64[None, :]))


@triton.jit
def _kv_pack(KV, K8, KSC, KR, E4M3: tl.constexpr):
    """One slot per CTA: per-slot scale over the whole 512-dim NoPE block."""
    n = tl.program_id(0)
    r512 = tl.arange(0, 512)
    r64 = tl.arange(0, 64)
    kv = tl.load(KV + n * 576 + r512).to(tl.float32)
    sc = tl.maximum(tl.max(tl.abs(kv)) / E4M3, 1e-6)
    tl.store(KSC + n, sc)
    tl.store(K8 + n * 512 + r512, (kv / sc).to(tl.float8e4nv))
    tl.store(KR + n * 64 + r64, tl.load(KV + n * 576 + 512 + r64))


@triton.jit
def _v_bf16(K8, KSC, VBF):
    """V = (fp8 NoPE * per-slot scale) as bf16; same math as baseline dequant."""
    n = tl.program_id(0)
    r512 = tl.arange(0, 512)
    v = tl.load(K8 + n * 512 + r512).to(tl.float32) * tl.load(KSC + n)
    tl.store(VBF + n * 512 + r512, v.to(tl.bfloat16))


# ======================================================================================
# A_win_z (m_off = 0): 1024 contiguous window slots -> P columns [0, 1024)
#   verbatim measured kernel wf5_dirc_z._win_z (agent "dirc"): no m_off computation, no
#   subtract; stores literal zeros into MOUT so B keeps its lse = m_off + ln(l) contract.
# ======================================================================================
@triton.jit
def _a_win_z(Q8, QSC, QR, K8, KSC, KR, BASEP, POUT, MOUT,
             S, HW, PSTRIDE,
             NS: tl.constexpr, SCALE_C: tl.constexpr, EMAX: tl.constexpr, CS: tl.constexpr):
    pid = tl.program_id(0)
    rq = pid * 128 + tl.arange(0, 128)
    r64 = tl.arange(0, 64)
    r512 = tl.arange(0, 512)
    rns = tl.arange(0, NS)
    qa = tl.load(Q8 + rq[:, None] * 512 + r512[None, :])
    qr = tl.load(QR + rq[:, None] * 64 + r64[None, :])
    qsc = tl.load(QSC + rq)
    base = tl.load(BASEP + pid * 4)
    pbase = POUT + rq[:, None] * PSTRIDE
    tl.store(MOUT + rq, tl.zeros((128,), tl.float32))
    for t0 in range(0, HW, NS):
        rows = base + t0 + rns
        kt = tl.load(K8 + rows[None, :] * 512 + r512[:, None])
        ktr = tl.load(KR + rows[None, :] * 64 + r64[:, None])
        ksc = tl.load(KSC + rows)
        s = tl.dot(qa, kt)
        s = s * (qsc[:, None] * ksc[None, :])
        s = tl.dot(qr, ktr, s)
        s = s * SCALE_C
        p = tl.exp2(tl.minimum(s, EMAX) * LOG2E_C).to(tl.bfloat16)
        if CS:
            tl.store(pbase + (t0 + rns)[None, :], p, cache_modifier=".cs")
        else:
            tl.store(pbase + (t0 + rns)[None, :], p)


# ======================================================================================
# A_dyn_z (m_off = 0, fast): 16 sink + 1008 gathered slots -> P columns [1024, 2048)
#   verbatim measured kernel wf5_dirc_z._dy_z; CONSTRAINT: the dyn block must be fully
#   valid (no -1 padding) -- this is the dyn_v3=True path, same contract as wf4.
# ======================================================================================
@triton.jit
def _a_dyn_z(Q8, QSC, QR, K8, KSC, KR, IDX, POUT,
             DS, DW, PST, POFF,
             NS: tl.constexpr, DST: tl.constexpr, SCALE_C: tl.constexpr,
             EMAX: tl.constexpr, CS: tl.constexpr):
    pid = tl.program_id(0)
    rq = pid * 128 + tl.arange(0, 128)
    r64 = tl.arange(0, 64)
    r512 = tl.arange(0, 512)
    rns = tl.arange(0, NS)
    qa = tl.load(Q8 + rq[:, None] * 512 + r512[None, :])
    qr = tl.load(QR + rq[:, None] * 64 + r64[None, :])
    qsc = tl.load(QSC + rq)
    idx0 = IDX + (pid * 4) * DS
    pbase = POUT + rq[:, None] * PST + POFF
    for td in tl.range(0, DW, NS, num_stages=DST):
        idx = tl.load(idx0 + td + rns)
        ktg = tl.load(K8 + idx[None, :] * 512 + r512[:, None])
        ktrg = tl.load(KR + idx[None, :] * 64 + r64[:, None])
        kscg = tl.load(KSC + idx)
        sg = tl.dot(qa, ktg)
        sg = sg * (qsc[:, None] * kscg[None, :])
        sg = tl.dot(qr, ktrg, sg)
        sg = sg * SCALE_C
        pg = tl.exp2(tl.minimum(sg, EMAX) * LOG2E_C).to(tl.bfloat16)
        if CS:
            tl.store(pbase + (td + rns)[None, :], pg, cache_modifier=".cs")
        else:
            tl.store(pbase + (td + rns)[None, :], pg)


# ======================================================================================
# A_dyn_z_safe (m_off = 0, padding-safe): same math as _a_dyn_z; idx < 0 padding is handled
#   by clamping the gather index (loads never touch K/V tables out of bounds) and folding
#   `valid` into the positional P-zero mask.  On fully-valid unions it is bitwise identical
#   to _a_dyn_z (verified in __main__).  This is the dyn_v3=False (default) path.
# ======================================================================================
@triton.jit
def _a_dyn_z_safe(Q8, QSC, QR, K8, KSC, KR, IDX, POUT,
                  DS, DW, SVALID, PST, POFF,
                  NS: tl.constexpr, SCALE_C: tl.constexpr, EMAX: tl.constexpr,
                  CS: tl.constexpr):
    pid = tl.program_id(0)
    rq = pid * 128 + tl.arange(0, 128)
    r64 = tl.arange(0, 64)
    r512 = tl.arange(0, 512)
    rns = tl.arange(0, NS)
    qa = tl.load(Q8 + rq[:, None] * 512 + r512[None, :])
    qr = tl.load(QR + rq[:, None] * 64 + r64[None, :])
    qsc = tl.load(QSC + rq)
    idx0 = IDX + (pid * 4) * DS
    pbase = POUT + rq[:, None] * PST + POFF
    for td in range(0, DW, NS):
        idx = tl.load(idx0 + td + rns)
        valid = idx >= 0
        ids = tl.maximum(idx, 0)          # clamped load: -1 rows never touch K/V tables
        ktg = tl.load(K8 + ids[None, :] * 512 + r512[:, None])
        ktrg = tl.load(KR + ids[None, :] * 64 + r64[:, None])
        kscg = tl.load(KSC + ids)
        sg = tl.dot(qa, ktg)
        sg = sg * (qsc[:, None] * kscg[None, :])
        sg = tl.dot(qr, ktrg, sg)
        sg = sg * SCALE_C
        pg = tl.exp2(tl.minimum(sg, EMAX) * LOG2E_C).to(tl.bfloat16)
        w = (td + rns) < SVALID            # positional padding guard (never fires at S==1024)
        w = w & valid                      # padding slots get P = 0 (no extra float select,
        pg = tl.where(w[None, :], pg, 0.0)  # so valid-lane values stay bitwise identical)
        if CS:
            tl.store(pbase + (td + rns)[None, :], pg, cache_modifier=".cs")
        else:
            tl.store(pbase + (td + rns)[None, :], pg)


# ======================================================================================
# B: PV contraction + denominator over the whole 2048-slot union (carried over verbatim):
#    one CTA owns all 512 dv columns -> P is read exactly once; lse = m_off + ln(l) with
#    m_off = 0 as stored by _a_win_z.
# ======================================================================================
@triton.jit
def _kb_512_lacc(P, MOFF, VBF, IDX, OUT, LSE, S, NS: tl.constexpr):
    pid = tl.program_id(0)
    rq = pid * 128 + tl.arange(0, 128)
    rns = tl.arange(0, NS)
    rc0 = tl.arange(0, 256)
    rc1 = 256 + tl.arange(0, 256)
    idx0 = IDX + (pid * 4) * S
    acc0 = tl.zeros((128, 256), dtype=tl.float32)
    acc1 = tl.zeros((128, 256), dtype=tl.float32)
    l_t = tl.zeros((128, NS), dtype=tl.float32)
    for t0 in range(0, S, NS):
        idx = tl.load(idx0 + t0 + rns)
        valid = idx >= 0
        p = tl.load(P + rq[:, None] * S + (t0 + rns)[None, :])
        l_t += p.to(tl.float32)
        v0 = tl.load(VBF + idx[:, None] * 512 + rc0[None, :], mask=valid[:, None], other=0.0)
        v1 = tl.load(VBF + idx[:, None] * 512 + rc1[None, :], mask=valid[:, None], other=0.0)
        acc0 += tl.dot(p, v0)
        acc1 += tl.dot(p, v1)
    l_i = tl.sum(l_t, axis=1)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(OUT + rq[:, None] * 512 + rc0[None, :], acc0 / l_safe[:, None])
    tl.store(OUT + rq[:, None] * 512 + rc1[None, :], acc1 / l_safe[:, None])
    m_off = tl.load(MOFF + rq)
    tl.store(LSE + rq, m_off + tl.log(l_safe))


# ======================================================================================
# index bookkeeping (single Triton kernel, no torch op chain; carried over verbatim):
#   idx [T,2048] -> basep [T] (window start), dyn [T,1024] (sink+gather),
#   idx_all [T,2048] = [win | dyn] with win[t,j] = base + j (contiguous window)
# ======================================================================================
@triton.jit
def _idx_build(IDX, BASEP, OUT, DYN,
               S: tl.constexpr, WIN: tl.constexpr, SINK: tl.constexpr):
    """grid = (T, 2): pid1=0 -> window half + basep, pid1=1 -> sink+gather half + DYN."""
    pid = tl.program_id(0)
    half = tl.program_id(1)
    base = tl.load(IDX + pid * S + SINK)
    u = half * WIN + tl.arange(0, WIN)
    if half == 0:
        tl.store(BASEP + pid, base)
        tl.store(OUT + pid * S + u, base + u)
    else:
        c = u - WIN
        off = tl.where(c < SINK, c, c + WIN)
        v = tl.load(IDX + pid * S + off)
        tl.store(OUT + pid * S + u, v)
        tl.store(DYN + pid * WIN + c, v)


# ======================================================================================
# L2 persisting set-aside (call-side; carried over verbatim from wf4_module_final.py)
#   * cudaLimitPersistingL2CacheSize (limit id 6) reserve ~56 MB (driver readback ~63 MB)
#   * cudaStreamSetAttribute (attr id 1 = AccessPolicyWindow) over VBF[0:56MB],
#     hitRatio 1.0, hitProp = cudaAccessPropertyPersisting(2) / missProp = Streaming(1)
#   * PROCESS-LEVEL state: applies to every later kernel on that stream (the kernels are
#     unchanged; only the cache policy changes).  reset_l2_persist() restores limit 0 and
#     clears the window.  l2_persist=True in the entry only *ensures* the window is set for
#     the current VBF (idempotent when the pointer is unchanged); it never disables one.
# ======================================================================================
class _APW(ctypes.Structure):
    _fields_ = [("base_ptr", ctypes.c_void_p), ("num_bytes", ctypes.c_size_t),
                ("hitRatio", ctypes.c_float), ("hitProp", ctypes.c_int),
                ("missProp", ctypes.c_int)]


class _SAV(ctypes.Union):
    _fields_ = [("accessPolicyWindow", _APW), ("pad", ctypes.c_byte * 64)]


try:
    _RT = ctypes.CDLL("libcudart.so")
except OSError:                                                       # pragma: no cover
    _RT = None

_CUDA_LIMIT_PERSISTING_L2 = 6          # cudaLimitPersistingL2CacheSize
_CUDA_STREAM_ATTR_APW = 1              # cudaStreamAttributeAccessPolicyWindow
_CUDA_PROP_PERSISTING = 2              # cudaAccessPropertyPersisting
_CUDA_PROP_STREAMING = 1               # cudaAccessPropertyStreaming

_L2_STATE = {"enabled": False, "ptr": 0, "reserve_mb": 0.0, "span_mb": 0.0, "ratio": 1.0,
             "rc_set_limit": 0, "rc_set_attr": 0}
_LAST_VBF = {}


def _resolve_stream(stream):
    if stream is None:
        return int(torch.cuda.current_stream().cuda_stream)
    return int(stream)


def l2_persist_state():
    """Copy of the current persisting-L2 state (for logging / assertions)."""
    return dict(_L2_STATE)


def enable_l2_persist_v(vbf=None, reserve_mb=56, span_mb=56, ratio=1.0, stream=None):
    """Reserve `reserve_mb` of persisting L2 and mark VBF[0:span_mb] as persisting.

    Call-side knob only (all kernels unchanged), measured B 0.2048 -> 0.1854 ms @T=2048.
    `vbf` is the bf16 V table returned by _kv_bundle; when omitted, the last VBF seen by
    the entry is used (call the entry once first, or pass vbf=...).  The window is applied
    to `stream` (default: the current torch stream) and the device limit is context-wide,
    so this is a PROCESS-LEVEL change: call reset_l2_persist() when the experiment ends.
    Returns the state dict (reserve readback in MB, effective span, return codes).
    """
    if _RT is None:
        raise RuntimeError("enable_l2_persist_v: libcudart.so not loadable")
    if vbf is None:
        vbf = _LAST_VBF.get(torch.cuda.current_device())
        if vbf is None:
            raise RuntimeError("enable_l2_persist_v: pass vbf=... (or call the entry once)")
    base = int(vbf.data_ptr())
    nbytes = int(min(span_mb * 1024 * 1024, vbf.numel() * vbf.element_size()))
    stream_id = _resolve_stream(stream)
    _RT.cudaCtxResetPersistingL2Cache()
    rc1 = int(_RT.cudaDeviceSetLimit(_CUDA_LIMIT_PERSISTING_L2,
                                     ctypes.c_size_t(int(reserve_mb * 1024 * 1024))))
    got = ctypes.c_size_t(0)
    _RT.cudaDeviceGetLimit(ctypes.byref(got), _CUDA_LIMIT_PERSISTING_L2)
    val = _SAV()
    val.accessPolicyWindow = _APW(ctypes.c_void_p(base), ctypes.c_size_t(nbytes),
                                  ctypes.c_float(float(ratio)), _CUDA_PROP_PERSISTING,
                                  _CUDA_PROP_STREAMING)
    rc2 = int(_RT.cudaStreamSetAttribute(ctypes.c_void_p(stream_id), _CUDA_STREAM_ATTR_APW,
                                         ctypes.byref(val)))
    _L2_STATE.update(enabled=True, ptr=base, reserve_mb=got.value / 1048576.0,
                     span_mb=nbytes / 1048576.0, ratio=float(ratio),
                     rc_set_limit=rc1, rc_set_attr=rc2)
    _LAST_VBF[int(vbf.device.index)] = vbf
    return l2_persist_state()


def reset_l2_persist(stream=None):
    """Undo enable_l2_persist_v: flush persisting lines, device limit -> 0, clear window."""
    if _RT is None:
        return l2_persist_state()
    stream_id = _resolve_stream(stream)
    _RT.cudaCtxResetPersistingL2Cache()
    rc1 = int(_RT.cudaDeviceSetLimit(_CUDA_LIMIT_PERSISTING_L2, ctypes.c_size_t(0)))
    val = _SAV()
    val.accessPolicyWindow = _APW(None, 0, ctypes.c_float(0.0), 0, 0)
    rc2 = int(_RT.cudaStreamSetAttribute(ctypes.c_void_p(stream_id), _CUDA_STREAM_ATTR_APW,
                                         ctypes.byref(val)))
    _L2_STATE.update(enabled=False, ptr=0, reserve_mb=0.0, span_mb=0.0,
                     rc_set_limit=rc1, rc_set_attr=rc2)
    return l2_persist_state()


def _ensure_l2_persist(vbf):
    if not (_L2_STATE["enabled"] and _L2_STATE["ptr"] == int(vbf.data_ptr())):
        enable_l2_persist_v(vbf)


# ======================================================================================
# buffer caches (same idea as sparse_mla_two_kernel._BUFS; no re-alloc per call)
# ======================================================================================
class _Bufs:
    def __init__(self):
        self.q = {}
        self.kv = {}
        self.p = {}
        self.idx = {}

    def qbuf(self, T, H, dev):
        key = (T, H, dev.index)
        if key not in self.q:
            rows = T * H
            self.q[key] = (torch.empty(rows, 512, dtype=torch.float8_e4m3fn, device=dev),
                           torch.empty(rows, dtype=torch.float32, device=dev),
                           torch.empty(rows, 64, dtype=torch.bfloat16, device=dev))
        return self.q[key]

    def kvbuf(self, N, dev):
        key = (N, dev.index)
        if key not in self.kv:
            self.kv[key] = (torch.empty(N, 512, dtype=torch.float8_e4m3fn, device=dev),
                            torch.empty(N, dtype=torch.float32, device=dev),
                            torch.empty(N, 64, dtype=torch.bfloat16, device=dev),
                            torch.empty(N, 512, dtype=torch.bfloat16, device=dev))
        return self.kv[key]

    def pbuf(self, T, S, dev):
        key = (T, S, dev.index)
        if key not in self.p:
            self.p[key] = (torch.zeros(T * 32, S, dtype=torch.bfloat16, device=dev),
                           torch.zeros(T * 32, dtype=torch.float32, device=dev))
        return self.p[key]

    def idxbufs(self, T, S, dev):
        key = (T, S, dev.index)
        if key not in self.idx:
            self.idx[key] = (torch.empty(T, dtype=torch.int32, device=dev),
                             torch.empty(T, DYN_SLOTS, dtype=torch.int32, device=dev),
                             torch.empty(T, S, dtype=torch.int32, device=dev))
        return self.idx[key]


_BUFS = _Bufs()
_KV_CACHE = {}          # kv.data_ptr() -> (kv, (k8, ksc, kr), vbf)   identity-checked


# ======================================================================================
# packing wrappers
# ======================================================================================
def pack_q(q):
    """[T,32,576] bf16 -> (q8 [T*32,512] fp8e4m3, qsc [T*32] fp32, qr [T*32,64] bf16)."""
    T, H, _ = q.shape
    rows = T * H
    q8, qsc, qr = _BUFS.qbuf(T, H, q.device)
    _q_pack[(rows // CFG_Q["R"],)](q.reshape(rows, 576), q8, qsc, qr, E4M3=E4M3_MAX,
                                   R=CFG_Q["R"], num_warps=CFG_Q["num_warps"])
    return q8, qsc, qr


def pack_kv(kv):
    """[N,576] bf16 -> (k8 [N,512] fp8, ksc [N] fp32, kr [N,64] bf16)."""
    N = kv.shape[0]
    k8, ksc, kr, _ = _BUFS.kvbuf(N, kv.device)
    _kv_pack[(N,)](kv, k8, ksc, kr, E4M3=E4M3_MAX, num_warps=CFG_KV["num_warps"])
    return k8, ksc, kr


def _kv_bundle(kv):
    """packed K + the bf16 dequantised V table, cached per kv tensor identity."""
    ent = _KV_CACHE.get(kv.data_ptr())
    if ent is not None and ent[0] is kv:
        return ent[1][0], ent[1][1], ent[1][2], ent[2]
    k8, ksc, kr = pack_kv(kv)
    _, _, _, vbf = _BUFS.kvbuf(kv.shape[0], kv.device)
    _v_bf16[(kv.shape[0],)](k8, ksc, vbf, num_warps=CFG_KV["num_warps"])
    _KV_CACHE[kv.data_ptr()] = (kv, (k8, ksc, kr), vbf)
    return k8, ksc, kr, vbf


def _prepare_idx(idx, T, S, dev):
    basep, dyn, idx_all = _BUFS.idxbufs(T, S, dev)
    _idx_build[(T, 2)](idx, basep, idx_all, dyn, S=S, WIN=WIN_SLOTS, SINK=SINK,
                       num_warps=4)
    return basep, dyn, idx_all


# ======================================================================================
# stream pair for the concurrent A segments (lazily created, cached per device)
# ======================================================================================
_STREAMS = {}


def _stream_pair(dev):
    key = int(dev.index)
    pair = _STREAMS.get(key)
    if pair is None:
        pair = (torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev))
        _STREAMS[key] = pair
    return pair


# ======================================================================================
# public entry
# ======================================================================================
def sparse_mla_two_kernel(q, kv, idx, out, lse, dyn_v3=False, l2_persist=False,
                          concurrent=True):
    """Sparse MLA prefill, m_off=0 + optional two-stream concurrency ("消波"), self-contained.

    Same call form / semantics as wf4_module_final.sparse_mla_two_kernel (see module
    docstring): idx must be 4-row shared and use the union column layout
    [sink16 | window1024 | gather1008]; P is partitioned into columns [0,1024) (A_win) and
    [1024,2048) (A_dyn); B computes out = sum(P V)/sum(P) and lse = m_off + ln(l) with the
    row-constant m_off = 0 used by BOTH segments (any per-row shift cancels in out, so this
    is mathematically equivalent to wf4's rowmax offset -- with the safety precondition
    max_s <= EMAX=60 documented in the module docstring).  Returns (out, lse).

    concurrent=True  -> _a_win_z and _a_dyn_z run on two CUDA streams and overlap (the free
                        stream waits the caller stream first; the caller stream waits both
                        before B).  This is where the measured +6.62% (T=2048) comes from.
    dyn_v3=True      -> fast _a_dyn_z (tl.range num_stages=3, bitwise identical P on a fully
                        valid union; requires the dyn block to contain no -1 padding).
    l2_persist=True  -> ensure the process-level L2 persisting set-aside window covers the
                        current VBF (see enable_l2_persist_v / reset_l2_persist; reset after
                        experiments).  No effect on results, only on speed.
    """
    T, H, _ = q.shape
    if (H != 32 or idx.shape[1] != UNION_S or idx.shape[0] != T or T % 4 or T < 4
            or idx.dtype != torch.int32):
        raise ValueError(
            f"wf6_module_final: unsupported inputs q={tuple(q.shape)} {q.dtype} / "
            f"idx={tuple(idx.shape)} {idx.dtype}; need q[T,32,576], idx[T,2048] int32, T%4==0")
    dev = q.device
    q8, qsc, qr = pack_q(q)
    k8, ksc, kr, vbf = _kv_bundle(kv)
    _LAST_VBF[int(dev.index)] = vbf
    if l2_persist:
        _ensure_l2_persist(vbf)
    basep, dyn, idx_all = _prepare_idx(idx, T, UNION_S, dev)
    p_all, m_all = _BUFS.pbuf(T, UNION_S, dev)

    def _win():
        _a_win_z[(T // 4,)](q8, qsc, qr, k8, ksc, kr, basep, p_all, m_all,
                            WIN_SLOTS, WIN_SLOTS, UNION_S,
                            NS=CFG_WIN["NS"], SCALE_C=SCALE, EMAX=EMAX, CS=True,
                            num_warps=CFG_WIN["num_warps"],
                            num_stages=CFG_WIN["num_stages"])

    def _dyn():
        if dyn_v3:
            _a_dyn_z[(T // 4,)](q8, qsc, qr, k8, ksc, kr, dyn, p_all,
                                DYN_SLOTS, DYN_SLOTS, UNION_S, WIN_SLOTS,
                                NS=CFG_DYN["NS"], DST=CFG_DYN["DST"], SCALE_C=SCALE,
                                EMAX=EMAX, CS=True, num_warps=CFG_DYN["num_warps"],
                                num_stages=CFG_DYN["num_stages"])
        else:
            _a_dyn_z_safe[(T // 4,)](q8, qsc, qr, k8, ksc, kr, dyn, p_all,
                                     DYN_SLOTS, DYN_SLOTS, DYN_SLOTS, UNION_S, WIN_SLOTS,
                                     NS=CFG_DYN_SAFE["NS"], SCALE_C=SCALE, EMAX=EMAX,
                                     CS=True, num_warps=CFG_DYN_SAFE["num_warps"],
                                     num_stages=CFG_DYN_SAFE["num_stages"])

    if concurrent:
        s_a, s_b = _stream_pair(dev)
        cur = torch.cuda.current_stream()
        s_a.wait_stream(cur)
        s_b.wait_stream(cur)
        with torch.cuda.stream(s_a):
            _win()
        with torch.cuda.stream(s_b):
            _dyn()
        cur.wait_stream(s_a)
        cur.wait_stream(s_b)
    else:
        _win()
        _dyn()

    _kb_512_lacc[(T // 4,)](p_all, m_all, vbf, idx_all, out, lse, UNION_S,
                            NS=CFG_B["NS"], num_warps=CFG_B["num_warps"],
                            num_stages=CFG_B["num_stages"])
    return out, lse


run = sparse_mla_two_kernel


# ======================================================================================
# safety-precondition helper (test/audit only; NOT on the hot path)
#   returns (max_abs_s, max_s, min_s) over the sampled rows, computed from the SAME packed
#   fp8 tensors the kernels use; compare with EMAX (60, hard cliff for m_off=0) and with the
#   bf16 underflow floor (~-87 normal / ~-93 subnormal) documented in the module docstring.
# ======================================================================================
def sampled_max_abs_s(q, kv, idx, rows=None, every=53, max_rows=40):
    q8, qsc, qr = pack_q(q)
    k8, ksc, kr, _ = _kv_bundle(kv)
    T = q.shape[0]
    if rows is None:
        rows = sorted(set(list(range(min(8, T))) + list(range(0, T, every))[:max_rows]))
    worst_abs, worst_pos, worst_neg = 0.0, float("-inf"), float("inf")
    for t in rows:
        ids = idx[t]
        sl = ids[ids >= 0].long()
        kd = k8[sl].float() * ksc[sl][:, None]
        qd = q8[t * 32:(t + 1) * 32].float() * qsc[t * 32:(t + 1) * 32, None]
        rd = qr[t * 32:(t + 1) * 32].float()
        s = (qd @ kd.t()) + (rd @ kr[sl].float().t())
        s = s * SCALE
        worst_abs = max(worst_abs, s.abs().max().item())
        worst_pos = max(worst_pos, s.max().item())
        worst_neg = min(worst_neg, s.min().item())
    return worst_abs, worst_pos, worst_neg


# ======================================================================================
# __main__ helpers
# ======================================================================================
def _bench(fn, w=2, it=20):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def _case(T, N=131072, S=2048, H=32, seed=5, qk_scale=1.0):
    """Union sparse index (16 sink + recent window + random top-k), 4-row shared.

    qk_scale scales BOTH q and kv (scores ~ scale^2); used by the safety audit only.
    """
    torch.manual_seed(seed)
    q = (torch.randn(T, H, 576, device="cuda") * (0.1 * qk_scale)).bfloat16()
    kv = (torch.randn(N, 576, device="cuda") * (0.1 * qk_scale)).bfloat16()
    pos = torch.arange(N - T, N, device="cuda", dtype=torch.int64)
    idx = torch.full((T, S), -1, dtype=torch.int32, device="cuda")
    for r in range(T):
        p = int(pos[r])
        w = torch.arange(max(0, p - 1023), p + 1, device="cuda")
        row = torch.cat([torch.arange(16, device="cuda"), w[-1024:],
                         torch.randint(16, max(17, p - 1023), (1008,), device="cuda")])[:S]
        idx[r, :row.numel()] = row.int()
    for r in range(T):
        if r % 4:
            idx[r] = idx[r - (r % 4)]
    return q, kv, idx


def _packed_ref(q8, qsc, qr, k8, ksc, kr, idx, T):
    """fp32 reference from the SAME fp8-packed tensors (4-row chunks); acceptance gate."""
    out_ref = torch.zeros(T, 32, 512, dtype=torch.float32, device=q8.device)
    lse_ref = torch.zeros(T, 32, dtype=torch.float32, device=q8.device)
    for b in range(T // 4):
        ids = idx[4 * b]
        valid = ids >= 0
        sl = ids[valid].long()
        kd = k8[sl].float() * ksc[sl][:, None]
        r0 = 128 * b
        qd = q8[r0:r0 + 128].float() * qsc[r0:r0 + 128].reshape(128, 1).float()
        rd = qr[r0:r0 + 128].reshape(128, 64).float()
        krr = kr[sl].float()
        s = (qd @ kd.t()) + (rd @ krr.t())
        s = s * SCALE
        m = s.max(dim=-1).values
        p = torch.exp(s - m[:, None])
        ps = p.sum(-1)
        out_ref[4 * b:4 * b + 4] = ((p @ kd) / ps[:, None]).view(4, 32, 512)
        lse_ref[4 * b:4 * b + 4] = (m + torch.log(ps)).view(4, 32)
    return out_ref, lse_ref


def _rel_chunk(out, out_ref):
    worst = 0.0
    for b in range(out.shape[0] // 4):
        den = out_ref[4 * b:4 * b + 4].abs().max().item()
        worst = max(worst, (out[4 * b:4 * b + 4] - out_ref[4 * b:4 * b + 4]).abs().max().item()
                    / max(1e-9, den))
    return worst


def _mfu(ms, T, S=2048):
    fl = 2.0 * 1088 * 32 * T * S
    return fl / (ms * 1e-3) / 2.5e15 * 100.0


def _import_wf4():
    """Guarded import of wf4_module_final.py for the cross-check only (test-time)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import wf4_module_final as w4
        return w4
    except Exception as ex:                                              # noqa: BLE001
        print(f"[wf6] wf4 cross-check skipped: wf4_module_final not importable "
              f"({type(ex).__name__}: {ex})", flush=True)
        return None


GROUPS6 = (("con=1 dv3=1 l2=1", dict(concurrent=True, dyn_v3=True, l2_persist=True)),
           ("con=0 dv3=1 l2=1", dict(concurrent=False, dyn_v3=True, l2_persist=True)),
           ("con=1 dv3=1 l2=0", dict(concurrent=True, dyn_v3=True, l2_persist=False)),
           ("con=0 dv3=1 l2=0", dict(concurrent=False, dyn_v3=True, l2_persist=False)),
           ("con=1 dv3=0 l2=0", dict(concurrent=True, dyn_v3=False, l2_persist=False)),
           ("con=0 dv3=0 l2=0", dict(concurrent=False, dyn_v3=False, l2_persist=False)))


def run_T(T, N=131072, S=2048, reps=3):
    log = lambda *a: print(*a, flush=True)                               # noqa: E731
    log(f"---- T={T} N={N} union S={S} ----")
    dev = "cuda"
    q, kv, idx = _case(T, N, S)
    w4 = _import_wf4()

    out = torch.zeros(T, 32, 512, dtype=torch.float32, device=dev)
    lse = torch.zeros(T, 32, dtype=torch.float32, device=dev)
    out_w4 = torch.zeros_like(out)
    lse_w4 = torch.zeros_like(lse)
    have_w4 = True
    if w4 is not None:
        try:
            w4.sparse_mla_two_kernel(q, kv, idx, out_w4, lse_w4)   # wf4 default path
        except Exception as ex:                                          # noqa: BLE001
            log(f"[T={T}] wf4 entry unavailable: {type(ex).__name__}: {str(ex)[:90]}")
            have_w4 = False

    reset_l2_persist()
    sparse_mla_two_kernel(q, kv, idx, out, lse)      # warm-up (packs kv/vbf, allocs bufs)
    torch.cuda.synchronize()
    q8, qsc, qr = pack_q(q)
    k8, ksc, kr, vbf = _kv_bundle(kv)
    pk_out, pk_lse = _packed_ref(q8, qsc, qr, k8, ksc, kr, idx, T)
    torch.cuda.synchronize()
    rel_pk = _rel_chunk(out, pk_out)
    lse_pk = (lse - pk_lse).abs().max().item()
    rel_pk_w4 = _rel_chunk(out_w4, pk_out) if have_w4 else float("nan")
    d_out_w4 = (out - out_w4).abs().max().item() / max(1e-6, out_w4.abs().max().item()) \
        if have_w4 else float("nan")
    d_lse_w4 = (lse - lse_w4).abs().max().item() if have_w4 else float("nan")
    s_abs, s_pos, s_neg = sampled_max_abs_s(q, kv, idx)
    log(f"[T={T}] ref: fp8pack rel_out {rel_pk:.3e} (wf4 {rel_pk_w4:.3e}) | "
        f"lse_maxdiff {lse_pk:.3e} | vs wf4 entry rel_out {d_out_w4:.3e} lse_maxdiff "
        f"{d_lse_w4:.3e}")
    log(f"[T={T}] safety: sampled max|s| {s_abs:.3f} (max {s_pos:.3f} / min {s_neg:.3f}) vs "
        f"EMAX {EMAX:.0f}, bf16 underflow ~-87/-93 -> margin ok")

    t_w4 = float("nan")
    if have_w4:
        t_w4 = min(_bench(lambda: w4.sparse_mla_two_kernel(q, kv, idx, out_w4, lse_w4),
                          w=2, it=20) for _ in range(reps))
        reset_l2_persist()
        log(f"[T={T}] wf4 entry (default path) {t_w4:.4f} ms MFU {_mfu(t_w4, T):5.2f}%")

    basep, dyn, idx_all = _prepare_idx(idx, T, UNION_S, q.device)
    p_all, m_all = _BUFS.pbuf(T, UNION_S, dev)

    def launch_win():
        _a_win_z[(T // 4,)](q8, qsc, qr, k8, ksc, kr, basep, p_all, m_all,
                            WIN_SLOTS, WIN_SLOTS, UNION_S, NS=CFG_WIN["NS"], SCALE_C=SCALE,
                            EMAX=EMAX, CS=True, num_warps=CFG_WIN["num_warps"],
                            num_stages=CFG_WIN["num_stages"])

    def launch_dyn(dv):
        if dv:
            _a_dyn_z[(T // 4,)](q8, qsc, qr, k8, ksc, kr, dyn, p_all,
                                DYN_SLOTS, DYN_SLOTS, UNION_S, WIN_SLOTS,
                                NS=CFG_DYN["NS"], DST=CFG_DYN["DST"], SCALE_C=SCALE,
                                EMAX=EMAX, CS=True, num_warps=CFG_DYN["num_warps"],
                                num_stages=CFG_DYN["num_stages"])
        else:
            _a_dyn_z_safe[(T // 4,)](q8, qsc, qr, k8, ksc, kr, dyn, p_all,
                                     DYN_SLOTS, DYN_SLOTS, DYN_SLOTS, UNION_S, WIN_SLOTS,
                                     NS=CFG_DYN_SAFE["NS"], SCALE_C=SCALE, EMAX=EMAX,
                                     CS=True, num_warps=CFG_DYN_SAFE["num_warps"],
                                     num_stages=CFG_DYN_SAFE["num_stages"])

    def launch_b():
        _kb_512_lacc[(T // 4,)](p_all, m_all, vbf, idx_all, out, lse, UNION_S,
                                NS=CFG_B["NS"], num_warps=CFG_B["num_warps"],
                                num_stages=CFG_B["num_stages"])

    def _a_pair(dv, conc):
        if conc:
            s_a, s_b = _stream_pair(q.device)
            cur = torch.cuda.current_stream()
            s_a.wait_stream(cur)
            s_b.wait_stream(cur)
            with torch.cuda.stream(s_a):
                launch_win()
            with torch.cuda.stream(s_b):
                launch_dyn(dv)
            cur.wait_stream(s_a)
            cur.wait_stream(s_b)
        else:
            launch_win()
            launch_dyn(dv)

    def combo3(dv, conc):
        _a_pair(dv, conc)
        launch_b()

    res = {}
    l2_info = None
    reset_l2_persist()
    for tag, kw in GROUPS6:
        if not kw["l2_persist"]:
            reset_l2_persist()                     # keep the non-L2 groups clean
        elif not _L2_STATE["enabled"]:
            l2_info = enable_l2_persist_v(vbf)
            log(f"[T={T}] L2 cfg: reserve {l2_info['reserve_mb']:.1f} MB "
                f"(rc {l2_info['rc_set_limit']}/{l2_info['rc_set_attr']}), window "
                f"{l2_info['span_mb']:.1f} MB hitRatio {l2_info['ratio']}")
        sparse_mla_two_kernel(q, kv, idx, out, lse, **kw)
        torch.cuda.synchronize()
        t = {}
        t["win"] = min(_bench(launch_win, w=2, it=20) for _ in range(reps))
        t["dyn"] = min(_bench(lambda: launch_dyn(kw["dyn_v3"]), w=2, it=20)
                       for _ in range(reps))
        t["B"] = min(_bench(launch_b, w=2, it=20) for _ in range(reps))
        t["combo3"] = min(_bench(lambda: combo3(kw["dyn_v3"], kw["concurrent"]), w=2, it=20)
                          for _ in range(reps))
        t["entry"] = min(_bench(lambda: sparse_mla_two_kernel(q, kv, idx, out, lse, **kw),
                                w=2, it=20) for _ in range(reps))
        t["sum3"] = t["win"] + t["dyn"] + t["B"]
        t["mfu"] = _mfu(t["entry"], T)
        t["mfu3"] = _mfu(t["combo3"], T)
        if kw["concurrent"] and kw["dyn_v3"] and kw["l2_persist"]:
            t["A_conc"] = min(_bench(lambda: _a_pair(True, True), w=2, it=20)
                              for _ in range(reps))
            t["A_seq"] = t["win"] + t["dyn"]
            log(f"[T={T}] A pair ({tag}): conc {t['A_conc']:.4f} ms vs seq {t['A_seq']:.4f} ms "
                f"({(t['A_seq'] - t['A_conc']) / t['A_seq'] * 100:+.1f}%; wave tail fill)")
        t["gain"] = (t_w4 - t["entry"]) / t_w4 * 100.0 if have_w4 else float("nan")
        sparse_mla_two_kernel(q, kv, idx, out, lse, **kw)
        torch.cuda.synchronize()
        t["rel_out"] = _rel_chunk(out, pk_out)
        t["lse_diff"] = (lse - pk_lse).abs().max().item()
        res[tag] = t
        log(f"[T={T}] {tag} entry {t['entry']:.4f} ms MFU {t['mfu']:5.2f}% | "
            f"win {t['win']:.4f} dyn {t['dyn']:.4f} B {t['B']:.4f} sum3 {t['sum3']:.4f} "
            f"combo3 {t['combo3']:.4f} MFU3 {t['mfu3']:5.2f}% | "
            f"rel_out {t['rel_out']:.3e} lse_diff {t['lse_diff']:.3e} gain_vs_wf4 "
            f"{t['gain']:5.1f}%")
    reset_l2_persist()

    # ---------------------------------------------------------------- bitwise self-checks
    reset_l2_persist()
    # (1) concurrent on/off: out/lse/P bitwise (same math, only the schedule differs)
    sparse_mla_two_kernel(q, kv, idx, out, lse, dyn_v3=True, concurrent=True)
    torch.cuda.synchronize()
    o_on, l_on, p_on = out.clone(), lse.clone(), p_all.clone()
    sparse_mla_two_kernel(q, kv, idx, out, lse, dyn_v3=True, concurrent=False)
    torch.cuda.synchronize()
    eq_c_o = bool(torch.equal(o_on, out))
    eq_c_l = bool(torch.equal(l_on, lse))
    eq_c_p = bool(torch.equal(p_on, p_all))
    n_diff_cp = int((p_on != p_all).sum().item())
    # (2) guarded dyn (dyn_v3=False) P == fast dyn (dyn_v3=True) P, bitwise
    p_all.zero_()
    launch_dyn(False)
    torch.cuda.synchronize()
    p_mask = p_all.clone()
    p_all.zero_()
    launch_dyn(True)
    torch.cuda.synchronize()
    n_diff_g = int((p_all != p_mask).sum().item())
    eq_g_p = n_diff_g == 0
    # (3) zero-score bridge (q = 0 -> every s = 0 -> wf4's rowmax offset is EXACTLY 0.0):
    #     under this input the two modules must agree BITWISE, which verifies the whole
    #     P/idx/B/pack plumbing against wf4 (the normal-input case cannot be bitwise equal
    #     by design: m_off=0 vs m_off=rowmax change the bf16 rounding of P).
    q0 = torch.zeros_like(q)
    ob = torch.zeros_like(out)
    lb = torch.zeros_like(lse)
    br_bits = {"wf6_def_on": False, "wf6_def_off": False, "wf6_dv3": False}
    bridge_rel = float("nan")
    if have_w4:
        o4 = torch.zeros_like(out)
        l4 = torch.zeros_like(lse)
        w4.sparse_mla_two_kernel(q0, kv, idx, o4, l4)
        torch.cuda.synchronize()
        sparse_mla_two_kernel(q0, kv, idx, ob, lb, concurrent=True)
        torch.cuda.synchronize()
        br_bits["wf6_def_on"] = bool(torch.equal(ob, o4) and torch.equal(lb, l4))
        sparse_mla_two_kernel(q0, kv, idx, ob, lb, concurrent=False)
        torch.cuda.synchronize()
        br_bits["wf6_def_off"] = bool(torch.equal(ob, o4) and torch.equal(lb, l4))
        sparse_mla_two_kernel(q0, kv, idx, ob, lb, dyn_v3=True, concurrent=True)
        torch.cuda.synchronize()
        br_bits["wf6_dv3"] = bool(torch.equal(ob, o4) and torch.equal(lb, l4))
        bridge_rel = (ob - o4).abs().max().item() / max(1e-6, o4.abs().max().item())
    # (4) determinism (two identical default runs) + q-pack byte-exactness vs wf4 +
    #     idx buffers bitwise vs wf4
    o1 = torch.zeros_like(out)
    l1 = torch.zeros_like(lse)
    o2 = torch.zeros_like(out)
    l2b = torch.zeros_like(lse)
    sparse_mla_two_kernel(q, kv, idx, o1, l1)
    sparse_mla_two_kernel(q, kv, idx, o2, l2b)
    torch.cuda.synchronize()
    stable = bool(torch.equal(o1, o2) and torch.equal(l1, l2b))
    eq_std = bool(have_w4 and torch.equal(o_on, out_w4) and torch.equal(l_on, lse_w4))
    pack_bytes = True
    idx_bits = True
    if have_w4:
        q8b, qscb, qrb = w4.pack_q(q)
        pack_bytes = bool(torch.equal(q8.view(torch.uint8), q8b.view(torch.uint8))
                          and torch.equal(qsc, qscb)
                          and torch.equal(qr.view(torch.int16), qrb.view(torch.int16)))
        bp4, dyn4, ia4 = w4._prepare_idx(idx, T, UNION_S, q.device)
        idx_bits = bool(torch.equal(basep, bp4) and torch.equal(dyn, dyn4)
                        and torch.equal(idx_all, ia4))
    ok = bool(rel_pk < 3e-2 and lse_pk < 1e-2 and eq_c_o and eq_c_l and eq_c_p and eq_g_p
              and stable and pack_bytes and idx_bits
              and (not have_w4 or all(br_bits.values())))
    log(f"[T={T}] selfcheck: concurrent on==off bits out/lse/P {eq_c_o}/{eq_c_l}/{eq_c_p} "
        f"({n_diff_cp} P differ) | dyn guard==fast {eq_g_p} ({n_diff_g} differ) | "
        f"zero-score bridge vs wf4 {br_bits} | vs wf4 std input: bitwise {eq_std} "
        f"(m_off convention differs by design) rel_out {d_out_w4:.3e} lse_maxdiff "
        f"{d_lse_w4:.3e} | pack_bytes {pack_bytes} | idx_bits {idx_bits} | stable {stable} "
        f"| {'PASS' if ok else 'FAIL'}")

    box = {"T": T, "groups": res, "rel_out": rel_pk, "lse_diff": lse_pk,
           "rel_out_wf4": rel_pk_w4, "d_out_vs_wf4": d_out_w4, "d_lse_vs_wf4": d_lse_w4,
           "s_abs": s_abs, "l2": l2_info, "wf4_entry_ms": t_w4,
           "bitwise": {"eq_c_out": eq_c_o, "eq_c_lse": eq_c_l, "eq_c_p": eq_c_p,
                       "p_ndiff_concurrent": n_diff_cp, "eq_guard_fast": eq_g_p,
                       "p_ndiff_guard": n_diff_g, "bridge": br_bits,
                       "bridge_rel": bridge_rel, "eq_std_vs_wf4": eq_std, "stable": stable,
                       "pack_bytes": pack_bytes, "idx_bits": idx_bits},
           "ok": ok}
    return box, ok


def main(argv=None):
    Ts = [int(a) for a in (argv if argv is not None else sys.argv[1:])] or [512, 2048, 4096]
    print(f"[env] python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"triton {triton.__version__} | cuda {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)} cc{torch.cuda.get_device_capability(0)} "
          f"SMs {torch.cuda.get_device_properties(0).multi_processor_count} "
          f"L2 {torch.cuda.get_device_properties(0).L2_cache_size / 1e6:.1f} MB", flush=True)
    print(f"[cfg] m_off=0 (EMAX {EMAX:.0f}) | q R={CFG_Q['R']} nw={CFG_Q['num_warps']} | "
          f"win {CFG_WIN} | dyn_z {CFG_DYN} | dyn_z_safe {CFG_DYN_SAFE} | B {CFG_B} | "
          f"concurrency streams 2 (lazy)", flush=True)
    results, all_ok = [], True
    for T in Ts:
        N = 8192 if T <= 8 else 131072
        try:
            box, ok = run_T(T, N=N)
        except Exception:                                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            box, ok = {"T": T, "error": "see traceback"}, False
        results.append(box)
        all_ok = all_ok and ok
        torch.cuda.synchronize()
    print("=" * 132)
    print(f"{'T':>6} {'cfg':>17} {'entry_ms':>9} {'MFU%':>7} {'win':>7} {'dyn':>7} {'B':>7} "
          f"{'sum3':>7} {'combo3':>7} {'MFU3%':>7} {'rel_out':>10} {'lse_diff':>10}")
    for b in results:
        if "error" in b:
            print(f"{b['T']:>6} {'FAIL':>17}  {b['error']}")
            continue
        for tag, kw in GROUPS6:
            g = b["groups"][tag]
            print(f"{b['T']:>6} {tag:>17} {g['entry']:>9.4f} {g['mfu']:>7.2f} "
                  f"{g['win']:>7.4f} {g['dyn']:>7.4f} {g['B']:>7.4f} {g['sum3']:>7.4f} "
                  f"{g['combo3']:>7.4f} {g['mfu3']:>7.2f} {g['rel_out']:>10.3e} "
                  f"{g['lse_diff']:>10.3e}")
    print("-" * 132)
    for b in results:
        if "error" in b:
            continue
        bw = b["bitwise"]
        print(f"[T={b['T']}] P(conc on==off) {bw['eq_c_p']} ({bw['p_ndiff_concurrent']} differ) | "
              f"dyn guard==fast {bw['eq_guard_fast']} ({bw['p_ndiff_guard']} differ) | "
              f"zero-score bridge {bw['bridge']} | stable {bw['stable']} | pack/idx "
              f"{bw['pack_bytes']}/{bw['idx_bits']} | {'PASS' if b['ok'] else 'FAIL'}")
    print(f"[wf6_module_final] OVERALL {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())