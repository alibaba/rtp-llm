# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Numerical tests for the Mamba-2 SSD kernels ported from vLLM.
#
# The oracle is a naive per-token SSM recurrence in float32 (the mathematical
# definition of the scan); the chunked kernels must reproduce it up to
# floating point reassociation and their internal low-precision casts.
#
# Shapes mirror the NemotronH Mamba-2 layer: nheads=64, headdim=64,
# dstate=128, ngroups=8, chunk_size=128. x / dt / B / C / z use the varlen
# flattened [total_T, ...] layout required by mamba_chunk_scan_combined_varlen.

import logging
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.mamba2 import mamba_chunk_scan_combined_varlen

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def make_varlen_chunk_meta(cu_seqlens, chunk_size, device):
    """Build the varlen chunk metadata from per-sequence token lengths.

    Sequences are chunked independently (pseudo chunks): each sequence starts
    at a fresh chunk boundary, exactly like vLLM's continuous-batching mixer.

    Returns:
        cu_chunk_seqlens: (nchunks + 1,) int32, token offsets of each chunk.
        seq_idx: (nchunks,) int32, owning sequence of each chunk.
        last_chunk_indices: (batch,) int32, global index of each sequence's
            last chunk.
    """
    batch = len(cu_seqlens) - 1
    cu_chunk_seqlens = [0]
    seq_idx = []
    last_chunk_indices = []
    for b in range(batch):
        start, end = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        length = end - start
        assert length >= 1, "empty sequences are not supported by the kernels"
        nchunks = (length + chunk_size - 1) // chunk_size
        for j in range(nchunks):
            cu_chunk_seqlens.append(min(start + (j + 1) * chunk_size, end))
            seq_idx.append(b)
        last_chunk_indices.append(len(cu_chunk_seqlens) - 2)
    return (
        torch.tensor(cu_chunk_seqlens, dtype=torch.int32, device=device),
        torch.tensor(seq_idx, dtype=torch.int32, device=device),
        torch.tensor(last_chunk_indices, dtype=torch.int32, device=device),
    )


def mamba_chunk_scan_varlen_ref(
    x,
    dt,
    A,
    B,
    C,
    cu_seqlens,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    """Naive per-token SSM recurrence oracle (everything in float32).

    out[t] = (state_t * C_t).sum(-1) (+ D * x_t) (gated by z)
    state_{t+1} = state_t * exp(A * dt_t) + dt_t * x_t (x) B_t
    """
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    ratio = nheads // ngroups
    device = x.device

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        # F.softplus uses the same 20.0 cutoff as the kernel's softplus.
        dt_f = F.softplus(dt_f)
    dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])

    A_f = A.float()
    # Head h reads group h // (nheads // ngroups), matching the kernels.
    B_f = B.float().repeat_interleave(ratio, dim=1)
    C_f = C.float().repeat_interleave(ratio, dim=1)
    x_f = x.float()
    z_f = z.float() if z is not None else None
    D_f = D.float() if D is not None else None

    batch = len(cu_seqlens) - 1
    out = torch.zeros_like(x_f)
    states_out = torch.zeros(
        batch, nheads, headdim, dstate, device=device, dtype=torch.float32
    )
    for b in range(batch):
        start, end = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        if initial_states is not None:
            state = initial_states[b].float().clone()
        else:
            state = torch.zeros(
                nheads, headdim, dstate, device=device, dtype=torch.float32
            )
        for t in range(start, end):
            dt_t = dt_f[t]  # (nheads,)
            dA = torch.exp(A_f * dt_t)  # (nheads,)
            state = (
                state * dA[:, None, None]
                + dt_t[:, None, None] * x_f[t][:, :, None] * B_f[t][:, None, :]
            )
            o = torch.einsum("hpn,hn->hp", state, C_f[t])
            if D_f is not None:
                if D_f.dim() == 1:
                    o = o + x_f[t] * D_f[:, None]
                else:
                    o = o + x_f[t] * D_f
            if z_f is not None:
                z_t = z_f[t]
                o = o * (z_t * torch.sigmoid(z_t))
            out[t] = o
        states_out[b] = state
    return out, states_out


class TestMamba2ChunkScan(unittest.TestCase):
    # NemotronH Mamba-2 layer shapes.
    NHEADS = 64
    HEADDIM = 64
    DSTATE = 128
    NGROUPS = 8
    CHUNK_SIZE = 128

    def _run_case(
        self,
        seqs,
        nheads,
        headdim,
        dstate,
        ngroups,
        chunk_size,
        dtype,
        dt_softplus,
        has_initial_states,
        has_z,
        d_has_hdim,
        dt_limit=(0.0, float("inf")),
        state_dtype=None,
        seed=0,
    ):
        device = "cuda"
        torch.manual_seed(seed)
        total = sum(seqs)
        cu = [0]
        for seq_len in seqs:
            cu.append(cu[-1] + seq_len)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
        cu_chunk_seqlens, seq_idx, last_chunk_indices = make_varlen_chunk_meta(
            cu_seqlens, chunk_size, device
        )

        # Data recipe keeps intermediate magnitudes mamba-like: dt_bias ~ -2
        # gives softplus(dt) ~ 0.1-0.2 and A in [-4, -1] gives fast decay.
        x = (torch.randn(total, nheads, headdim, device=device) * 0.5).to(dtype)
        B = (torch.randn(total, ngroups, dstate, device=device) * 0.5).to(dtype)
        C = (torch.randn(total, ngroups, dstate, device=device) * 0.5).to(dtype)
        dt = (torch.randn(total, nheads, device=device) * 0.5).to(dtype)
        A = -(1.0 + 3.0 * torch.rand(nheads, device=device))
        dt_bias = -2.0 - 0.5 * torch.rand(nheads, device=device)
        z = (torch.randn(total, nheads, headdim, device=device) * 0.5).to(dtype)
        if not has_z:
            z = None
        if d_has_hdim:
            D = torch.randn(nheads, headdim, device=device) * 0.1
        else:
            D = torch.randn(nheads, device=device) * 0.1
        initial_states = None
        if has_initial_states:
            # The kernel requires initial states in the SSM state dtype (the
            # cache stores them in state_dtype); bf16 activations with fp32
            # states must pass fp32 initial tensors.
            initial_states = (
                torch.randn(len(seqs), nheads, headdim, dstate, device=device).to(
                    state_dtype or dtype
                )
                * 0.1
            )

        out = torch.empty_like(x)
        varlen_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            state_dtype=state_dtype,
        )

        out_ref, states_ref = mamba_chunk_scan_varlen_ref(
            x,
            dt,
            A,
            B,
            C,
            cu_seqlens,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )

        # vLLM's own SSD kernel test uses 8e-3/5e-3 for fp32 and 5e-2 for
        # bf16: the chunked quadratic form reassociates the recurrence, so
        # tail-of-sequence drift above naive-loop fp32 rounding is expected.
        if dtype == torch.float32:
            rtol, atol = 5e-3, 8e-3
        else:
            rtol, atol = 5e-2, 5e-2
        torch.testing.assert_close(out.float(), out_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            varlen_states.float(), states_ref, rtol=rtol, atol=atol
        )
        logging.info(
            "chunk_scan ok: seqs=%s nheads=%d headdim=%d dstate=%d ngroups=%d "
            "chunk=%d dtype=%s softplus=%s init_states=%s z=%s D_hdim=%s "
            "state_dtype=%s",
            seqs,
            nheads,
            headdim,
            dstate,
            ngroups,
            chunk_size,
            dtype,
            dt_softplus,
            has_initial_states,
            has_z,
            d_has_hdim,
            state_dtype,
        )

    def test_nemotronh_shapes_fp32(self):
        # Multi-batch varlen, chunk remainder + exact multiple, no initial
        # states: first chunk of every sequence must start from zero state.
        self._run_case(
            seqs=[347, 128, 65],
            nheads=self.NHEADS,
            headdim=self.HEADDIM,
            dstate=self.DSTATE,
            ngroups=self.NGROUPS,
            chunk_size=self.CHUNK_SIZE,
            dtype=torch.float32,
            dt_softplus=True,
            has_initial_states=False,
            has_z=True,
            d_has_hdim=False,
        )

    def test_nemotronh_shapes_bf16_initial_states(self):
        # bf16 activations with initial states and sequence starts mid-batch.
        self._run_case(
            seqs=[300, 129, 7, 256],
            nheads=self.NHEADS,
            headdim=self.HEADDIM,
            dstate=self.DSTATE,
            ngroups=self.NGROUPS,
            chunk_size=self.CHUNK_SIZE,
            dtype=torch.bfloat16,
            dt_softplus=True,
            has_initial_states=True,
            has_z=True,
            d_has_hdim=False,
        )

    def test_small_dims_variants_fp32(self):
        # ngroups != nheads and != 1, 2-D D, no z gate, no softplus with an
        # explicit dt_limit (exercises the clamp path in _chunk_cumsum_fwd).
        self._run_case(
            seqs=[200, 33],
            nheads=4,
            headdim=32,
            dstate=64,
            ngroups=2,
            chunk_size=128,
            dtype=torch.float32,
            dt_softplus=False,
            has_initial_states=True,
            has_z=False,
            d_has_hdim=True,
            dt_limit=(0.05, 0.4),
        )

    def test_ngroups1_state_dtype_fp32(self):
        # ngroups=1 (every head reads group 0) with bf16 inputs but float32
        # ssm states (state_dtype mechanism).
        self._run_case(
            seqs=[259, 128],
            nheads=8,
            headdim=64,
            dstate=128,
            ngroups=1,
            chunk_size=128,
            dtype=torch.bfloat16,
            dt_softplus=True,
            has_initial_states=True,
            has_z=True,
            d_has_hdim=False,
            state_dtype=torch.float32,
        )


if __name__ == "__main__":
    unittest.main()
