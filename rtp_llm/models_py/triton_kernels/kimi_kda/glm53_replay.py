"""Replay an accepted KDA chain into RTP's FP32 K-major cache.

Verify retains BF16 convolution inputs/outputs and raw gates, rather than a
matrix state per candidate. Stable descriptors batch all layers into one commit.
"""

import torch
import triton
import triton.language as tl

MAX_VERIFY_TOKENS = 8


class KDAReplayWorkspace:
    def __init__(self, batch, heads, dim, device, conv, state):
        self.batch = batch
        self.heads = heads
        self.dim = dim
        self.qkv = torch.empty(
            (3, batch, MAX_VERIFY_TOKENS, heads * dim),
            device=device,
            dtype=torch.bfloat16,
        )
        self.gate = torch.empty(
            (batch, MAX_VERIFY_TOKENS, heads * dim), device=device, dtype=torch.bfloat16
        )
        self.beta = torch.empty(
            (batch, MAX_VERIFY_TOKENS, heads), device=device, dtype=torch.bfloat16
        )
        self.raw = torch.empty(
            (batch, MAX_VERIFY_TOKENS, 3 * heads * dim),
            device=device,
            dtype=torch.bfloat16,
        )
        self.history = torch.empty(
            (batch, 3, 3 * heads * dim), device=device, dtype=conv.dtype
        )
        # [sequence_length_plus_one, source, current destination, next destination, width]
        self.metadata = torch.zeros((batch, 5), device=device, dtype=torch.int64)
        self.conv = conv
        self.state = state

    def descriptor(self):
        return [
            t.data_ptr()
            for t in (
                self.qkv,
                self.gate,
                self.beta,
                self.raw,
                self.history,
                self.metadata,
                self.conv,
                self.state,
            )
        ]

    @property
    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.qkv,
                self.gate,
                self.beta,
                self.raw,
                self.history,
                self.metadata,
            )
        )

    def capture_gates(self, gate, beta, batch, tokens):
        _capture_replay_gates[(batch, triton.cdiv(self.heads * self.dim, 256))](
            gate,
            beta,
            self.gate,
            self.beta,
            tokens,
            self.heads,
            self.dim,
            MAX_VERIFY_TOKENS,
            256,
        )


@triton.jit
def _capture_replay_gates(
    g,
    b,
    go,
    bo,
    T: tl.int32,
    H: tl.constexpr,
    D: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    for t in range(T):
        tl.store(
            go + (n * CAP + t) * H * D + d,
            tl.load(g + (n * T + t) * H * D + d, d < H * D, 0),
            d < H * D,
        )
        tl.store(
            bo + (n * CAP + t) * H + d,
            tl.load(b + (n * T + t) * H + d, d < H, 0),
            d < H,
        )


@triton.jit
def _glm53_kda_replay_commit_kernel(
    descriptors,
    accepted,
    B: tl.int32,
    MAX_B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    CONV_STRIDE: tl.constexpr,
    LOWER: tl.constexpr,
    BV: tl.constexpr,
):
    layer = tl.program_id(0)
    n = tl.program_id(1)
    hv = tl.program_id(2) // tl.cdiv(D, BV)
    iv = tl.program_id(2) % tl.cdiv(D, BV)
    desc = descriptors + layer * 10
    qkv = tl.load(desc).to(tl.pointer_type(tl.bfloat16))
    gate = tl.load(desc + 1).to(tl.pointer_type(tl.bfloat16))
    beta = tl.load(desc + 2).to(tl.pointer_type(tl.bfloat16))
    raw = tl.load(desc + 3).to(tl.pointer_type(tl.bfloat16))
    history = tl.load(desc + 4).to(tl.pointer_type(tl.bfloat16))
    meta = tl.load(desc + 5).to(tl.pointer_type(tl.int64))
    conv = tl.load(desc + 6).to(tl.pointer_type(tl.bfloat16))
    state = tl.load(desc + 7).to(tl.pointer_type(tl.float32))
    a = tl.load(desc + 8).to(tl.pointer_type(tl.float32))
    dt = tl.load(desc + 9).to(tl.pointer_type(tl.float32))
    seq = tl.load(meta + n * 5)
    src = tl.load(meta + n * 5 + 1)
    dst0 = tl.load(meta + n * 5 + 2)
    dst1 = tl.load(meta + n * 5 + 3)
    width = tl.load(meta + n * 5 + 4)
    steps = tl.minimum(tl.load(accepted + n), width)
    if src <= 0 or steps <= 0:
        return
    kk = tl.arange(0, D)
    vv = iv * BV + tl.arange(0, BV)
    sm = vv[:, None] < D
    state_offset = hv * D * D + kk[None, :] * D + vv[:, None]
    s = tl.load(state + src * STATE_STRIDE + state_offset, sm, 0).to(tl.float32)
    c = H * D
    hd = hv * D + kk
    # The seed was captured before verify, so in-place writes cannot race
    # another value tile's reads of convolution history.
    q0 = tl.load(history + n * 9 * c + hd).to(tl.float32)
    q1 = tl.load(history + n * 9 * c + 3 * c + hd).to(tl.float32)
    q2 = tl.load(history + n * 9 * c + 6 * c + hd).to(tl.float32)
    k0 = tl.load(history + n * 9 * c + c + hd).to(tl.float32)
    k1 = tl.load(history + n * 9 * c + 4 * c + hd).to(tl.float32)
    k2 = tl.load(history + n * 9 * c + 7 * c + hd).to(tl.float32)
    v0 = tl.load(history + n * 9 * c + 2 * c + hd).to(tl.float32)
    v1 = tl.load(history + n * 9 * c + 5 * c + hd).to(tl.float32)
    v2 = tl.load(history + n * 9 * c + 8 * c + hd).to(tl.float32)
    av = tl.exp(tl.load(a + hv).to(tl.float32))
    db = tl.load(dt + hd).to(tl.float32)
    for t in range(steps):
        row = (n * CAP + t) * c + hd
        q = tl.load(qkv + row).to(tl.float32)
        k = tl.load(qkv + MAX_B * CAP * c + row).to(tl.float32)
        v = tl.load(
            qkv + 2 * MAX_B * CAP * c + (n * CAP + t) * c + hv * D + vv, vv < D, 0
        ).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6) * (D**-0.5)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        g = LOWER * tl.sigmoid(av * (tl.load(gate + row).to(tl.float32) + db))
        s *= tl.exp(g[None, :])
        v -= tl.sum(s * k[None, :], 1)
        v *= tl.sigmoid(tl.load(beta + (n * CAP + t) * H + hv).to(tl.float32))
        s += v[:, None] * k[None, :]
        q0, q1, q2 = q1, q2, tl.load(raw + (n * CAP + t) * 3 * c + hd).to(tl.float32)
        k0, k1, k2 = (
            k1,
            k2,
            tl.load(raw + (n * CAP + t) * 3 * c + c + hd).to(tl.float32),
        )
        v0, v1, v2 = (
            v1,
            v2,
            tl.load(raw + (n * CAP + t) * 3 * c + 2 * c + hd).to(tl.float32),
        )
        # Preserve an accepted page boundary, then publish the final state.
        # These are ordinary sequence pages; no per-position block swaps follow.
        if (seq + t) % PAGE == 0 or t == steps - 1:
            dest = tl.where((seq + t - 1) // PAGE == (seq - 1) // PAGE, dst0, dst1)
            tl.store(state + dest * STATE_STRIDE + state_offset, s, sm & (dest > 0))
            if iv == 0:
                cp = conv + dest * CONV_STRIDE
                tl.store(cp + hd, q0, dest > 0)
                tl.store(cp + c + hd, k0, dest > 0)
                tl.store(cp + 2 * c + hd, v0, dest > 0)
                tl.store(cp + 3 * c + hd, q1, dest > 0)
                tl.store(cp + 4 * c + hd, k1, dest > 0)
                tl.store(cp + 5 * c + hd, v1, dest > 0)
                tl.store(cp + 6 * c + hd, q2, dest > 0)
                tl.store(cp + 7 * c + hd, k2, dest > 0)
                tl.store(cp + 8 * c + hd, v2, dest > 0)


def commit_kda_replay(descriptors, workspaces, accepted, page, lower_bound):
    first = workspaces[0]
    if (
        accepted.ndim != 1
        or accepted.dtype != torch.int32
        or not accepted.is_cuda
        or accepted.device != first.qkv.device
        or accepted.numel() > first.batch
        or descriptors.shape != (len(workspaces), 10)
        or descriptors.dtype != torch.uint64
        or descriptors.device != accepted.device
        or not descriptors.is_contiguous()
    ):
        raise ValueError(
            "KDA replay requires CUDA int32 accepted lengths within its batch capacity"
        )
    if accepted.numel() == 0:
        return
    _glm53_kda_replay_commit_kernel[
        (len(workspaces), accepted.numel(), first.heads * triton.cdiv(first.dim, 32))
    ](
        descriptors,
        accepted,
        accepted.numel(),
        first.batch,
        first.heads,
        first.dim,
        MAX_VERIFY_TOKENS,
        page,
        first.state.stride(0),
        first.conv.stride(0),
        lower_bound,
        32,
        num_warps=4,
    )
