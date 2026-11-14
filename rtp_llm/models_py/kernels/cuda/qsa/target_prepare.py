"""Device kernels; model state and execution policy belong to the caller."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from rtp_llm.models_py.kernels.cuda.qsa.target_writer import (
    _snapshot_target_window,
    _write_target_window,
)


@triton.jit
def _prepare_target_window(
    X,
    QG,
    QP,
    KP,
    FREQ,
    ROTATED,
    COS,
    SIN,
    PREFIX,
    KV,
    KT,
    STATE,
    ST,
    SS,
    OS,
    KS,
    OK,
    Q: tl.constexpr,
    H: tl.constexpr,
    XS: tl.constexpr,
    PS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    EPS: tl.constexpr,
    KTW: tl.constexpr,
    STW: tl.constexpr,
    KB: tl.constexpr,
    SB: tl.constexpr,
    BQ: tl.constexpr,
):
    request = tl.program_id(0)
    local = tl.program_id(1)
    step = local // H
    head = local % H
    row = request * Q + step
    if local == 0:
        _snapshot_target_window(
            PREFIX,
            KV,
            KT,
            STATE,
            ST,
            SS,
            OS,
            KS,
            OK,
            Q=Q,
            KTW=KTW,
            STW=STW,
            KB=KB,
            SB=SB,
            BQ=BQ,
        )
    lane = tl.arange(0, 32)
    col = lane * 4
    offset = row * XS + head * 128
    a = tl.load(X + offset + col).to(tl.float32)
    b = tl.load(X + offset + col + 1).to(tl.float32)
    c = tl.load(X + offset + col + 2).to(tl.float32)
    d = tl.load(X + offset + col + 3).to(tl.float32)
    inv = tl.rsqrt(tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128.0 + EPS)
    position = tl.load(QP + row * PS).to(tl.float32)
    frequency = tl.load(FREQ + lane)
    angle = (position * INV_SCALE) * frequency
    cosine = libdevice.cos(angle).to(tl.bfloat16).to(tl.float32)
    sine = libdevice.sin(angle).to(tl.bfloat16).to(tl.float32)
    for j in tl.static_range(4):
        i = col + j
        partner = tl.where(i < 32, i + 32, i - 32)
        partner = tl.where(i < 64, partner, i)
        gain = 1.0 + tl.load(QG + i).to(tl.float32)
        other_gain = 1.0 + tl.load(QG + partner).to(tl.float32)
        value = tl.load(X + offset + i).to(tl.float32)
        other = tl.load(X + offset + partner).to(tl.float32)
        normal = ((value * inv) * gain).to(tl.bfloat16).to(tl.float32)
        rotated = ((other * inv) * other_gain).to(tl.bfloat16).to(tl.float32)
        fi = i % 32
        cv = tl.gather(cosine, fi, 0)
        sv = tl.gather(sine, fi, 0)
        left = (normal * cv).to(tl.bfloat16).to(tl.float32)
        right = (
            (rotated * tl.where(i < 32, -1.0, 1.0) * sv).to(tl.bfloat16).to(tl.float32)
        )
        result = tl.where(
            i < 64, (left + right).to(tl.bfloat16), normal.to(tl.bfloat16)
        )
        tl.store(ROTATED + (row * H + head) * 128 + i, result)
    if head == 0:
        kp = tl.load(KP + row * PS).to(tl.float32)
        ka = (kp * INV_SCALE) * frequency
        kc = libdevice.cos(ka)
        ks = libdevice.sin(ka)
        tl.store(COS + row * 64 + lane, kc)
        tl.store(COS + row * 64 + lane + 32, kc)
        tl.store(SIN + row * 64 + lane, ks)
        tl.store(SIN + row * 64 + lane + 32, ks)
