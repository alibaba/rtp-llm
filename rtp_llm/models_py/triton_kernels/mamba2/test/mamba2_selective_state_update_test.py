# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Numerical tests for the Mamba-2 selective_state_update (decode) kernel
# ported from vLLM.
#
# The oracle is a naive per-token SSM update in float32 that mirrors the
# kernel's cache-slot redirection contract:
# - state_batch_indices: which cache slot each (batch, token) reads from,
#   with null_block_id marking padded rows that start from a zero state and
#   are never written back.
# - dst_state_batch_indices: which cache slot receives the state (defaults to
#   in-place updates through state_batch_indices).
# - num_accepted_tokens + per-token dst slots: the speculative-decoding path,
#   where the state after every token is snapshotted into its own slot and
#   the initial state comes from the slot of the last previously accepted
#   token.
#
# Shapes mirror the NemotronH Mamba-2 layer: nheads=64, headdim=64,
# dstate=128, ngroups=8, TIE_HDIM (dt is [T, nheads], A is [nheads]).

import logging
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.mamba2 import (
    NULL_BLOCK_ID,
    selective_state_update,
)

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def selective_state_update_ref(
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    z=None,
    dt_softplus=False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    null_block_id=NULL_BLOCK_ID,
    num_accepted_tokens=None,
    cu_seqlens=None,
):
    """Naive per-token oracle for _selective_scan_update_kernel (float32).

    Tied inputs (stride-0 expanded dt / A / dt_bias) are handled by plain
    broadcasting, so the same code covers TIE_HDIM and the full-matrix path.

    Returns:
        out_ref: (total, nheads, dim) float32.
        state_after: copy of the state cache after the kernel-equivalent
            writes (stored in the cache dtype).
    """
    cache_dtype = state.dtype
    state = state.clone()
    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    dt_bias_f = dt_bias.float() if dt_bias is not None else None
    D_f = D.float() if D is not None else None
    z_f = z.float() if z is not None else None

    nheads = x_f.shape[1]
    ngroups = B_f.shape[1]
    ratio = nheads // ngroups
    B_h = B_f.repeat_interleave(ratio, dim=1)  # (total, nheads, dstate)
    C_h = C_f.repeat_interleave(ratio, dim=1)

    if cu_seqlens is not None:
        n_seq = len(cu_seqlens) - 1
    else:
        n_seq = x_f.shape[0]
    has_indices = state_batch_indices is not None
    is_spec = num_accepted_tokens is not None
    dst_indices = (
        dst_state_batch_indices
        if dst_state_batch_indices is not None
        else state_batch_indices
    )

    out = torch.zeros_like(x_f)
    for pid_b in range(n_seq):
        if cu_seqlens is not None:
            bos, eos = int(cu_seqlens[pid_b]), int(cu_seqlens[pid_b + 1])
            seq_len = eos - bos
            if seq_len == 0:
                continue
        else:
            bos, seq_len = pid_b, 1

        if has_indices:
            init_t = max(int(num_accepted_tokens[pid_b]) - 1, 0) if is_spec else 0
            read_slot = int(state_batch_indices[pid_b, init_t])
            valid = read_slot != null_block_id
        else:
            read_slot = pid_b
            valid = True

        if valid:
            s = state[read_slot].float().clone()
        else:
            # Padded row: start from a zero state and never write back.
            s = torch.zeros_like(state[0].float())

        for i_t in range(seq_len):
            t = bos + i_t
            x_t = x_f[t]  # (nheads, dim)
            dt_t = dt_f[t]  # (nheads, dim) (stride-0 when tied)
            if dt_bias_f is not None:
                dt_t = dt_t + dt_bias_f
            if dt_softplus:
                dt_t = F.softplus(dt_t)
            dA = torch.exp(A_f * dt_t[:, :, None])  # (nheads, dim, dstate)
            dB = B_h[t][:, None, :] * dt_t[:, :, None]
            s = s * dA + dB * x_t[:, :, None]

            if is_spec:
                dst_slot = int(dst_indices[pid_b, i_t])
                if valid and dst_slot != null_block_id:
                    state[dst_slot] = s.to(cache_dtype)
            out[t] = (s * C_h[t][:, None, :]).sum(-1)
            if D_f is not None:
                out[t] = out[t] + x_t * D_f
            if z_f is not None:
                z_t = z_f[t]
                out[t] = out[t] * (z_t * torch.sigmoid(z_t))

        if not is_spec:
            if has_indices:
                write_slot = int(dst_indices[pid_b, 0])
            else:
                write_slot = pid_b
            if valid:
                state[write_slot] = s.to(cache_dtype)
    return out, state


class TestMamba2SelectiveStateUpdate(unittest.TestCase):
    # NemotronH Mamba-2 layer shapes.
    NHEADS = 64
    HEADDIM = 64
    DSTATE = 128
    NGROUPS = 8

    def _make_inputs(
        self,
        total,
        nheads,
        dim,
        dstate,
        ngroups,
        dtype,
        tie_hdim=True,
        has_z=True,
        seed=0,
    ):
        device = "cuda"
        torch.manual_seed(seed)
        x = (torch.randn(total, nheads, dim, device=device) * 0.5).to(dtype)
        B = (torch.randn(total, ngroups, dstate, device=device) * 0.5).to(dtype)
        C = (torch.randn(total, ngroups, dstate, device=device) * 0.5).to(dtype)
        z = (torch.randn(total, nheads, dim, device=device) * 0.5).to(dtype)
        if not has_z:
            z = None
        A = -(1.0 + 3.0 * torch.rand(nheads, device=device))
        dt_bias = -2.0 - 0.5 * torch.rand(nheads, device=device)
        D = torch.randn(nheads, device=device) * 0.1
        if tie_hdim:
            # dt is [T, nheads] and A is [nheads]; expand with stride-0 so the
            # kernel detects TIE_HDIM exactly like the NemotronH mixer does.
            dt = torch.randn(total, nheads, device=device) * 0.5
            dt = dt.unsqueeze(-1).expand(total, nheads, dim)
            A = A[:, None, None].expand(nheads, dim, dstate)
            dt_bias = dt_bias.unsqueeze(-1).expand(nheads, dim)
        else:
            dt = (torch.randn(total, nheads, dim, device=device) * 0.5).to(dtype)
            A = -(1.0 + 3.0 * torch.rand(nheads, dim, dstate, device=device))
            dt_bias = -(2.0 + 0.5 * torch.rand(nheads, dim, device=device))
        D = D.unsqueeze(-1).expand(nheads, dim)
        return x, dt, A, B, C, D, dt_bias, z

    def _run_kernel(
        self,
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z,
        state_batch_indices=None,
        dst_state_batch_indices=None,
        num_accepted_tokens=None,
        cu_seqlens=None,
        dt_softplus=True,
    ):
        out = torch.empty_like(x)
        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z=z,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            null_block_id=NULL_BLOCK_ID,
            out=out,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
        )
        return out

    def _asserts(self, out, out_ref, state, state_ref):
        # fp32 states match the oracle tightly (rtol 1e-3); bf16 states only
        # match up to the store-time cast (rtol 1e-2). Output tolerance always
        # follows the activation dtype.
        if state.dtype == torch.float32:
            state_rtol, state_atol = 1e-3, 1e-3
        else:
            state_rtol, state_atol = 1e-2, 1e-2
        if out.dtype == torch.float32:
            out_rtol, out_atol = 1e-3, 1e-3
        else:
            out_rtol, out_atol = 1e-2, 1e-2
        torch.testing.assert_close(out.float(), out_ref, rtol=out_rtol, atol=out_atol)
        torch.testing.assert_close(state, state_ref, rtol=state_rtol, atol=state_atol)

    def test_decode_tie_hdim_inplace_fp32_state(self):
        # Plain single-token decode, TIE_HDIM (NemotronH), in-place fp32 state.
        batch = 8
        nheads, dim, dstate, ngroups = (
            self.NHEADS,
            self.HEADDIM,
            self.DSTATE,
            self.NGROUPS,
        )
        x, dt, A, B, C, D, dt_bias, z = self._make_inputs(
            batch, nheads, dim, dstate, ngroups, torch.bfloat16, tie_hdim=True
        )
        state = torch.randn(batch, nheads, dim, dstate, device="cuda") * 0.1
        state_ref = state.clone()

        out = self._run_kernel(state, x, dt, A, B, C, D, dt_bias, z)
        out_ref, state_after = selective_state_update_ref(
            state_ref, x, dt, A, B, C, D, dt_bias, z=z, dt_softplus=True
        )
        self._asserts(out, out_ref, state, state_after)
        logging.info("selective_state_update ok: tie_hdim in-place decode")

    def test_decode_state_batch_indices_redirect(self):
        # Cache-slot redirection with padded (null) rows and a dst mapping
        # that differs from the read mapping.
        batch, nslots = 8, 16
        nheads, dim, dstate, ngroups = (
            self.NHEADS,
            self.HEADDIM,
            self.DSTATE,
            self.NGROUPS,
        )
        x, dt, A, B, C, D, dt_bias, z = self._make_inputs(
            batch, nheads, dim, dstate, ngroups, torch.bfloat16, tie_hdim=True, seed=1
        )
        # Distinct sentinel values let us verify untouched slots exactly.
        state = torch.randn(nslots, nheads, dim, dstate, device="cuda")
        state_ref = state.clone()

        # Kernel contract (same as vLLM): every cache slot is owned by at
        # most one batch row per launch. A dst slot may alias its OWN row's
        # read slot (sequential read-then-write inside one program), but it
        # must never alias another row's read slot — rows run as concurrent
        # GPU programs, so cross-row aliasing would be a data race.
        state_batch_indices = torch.tensor(
            [[3], [7], [NULL_BLOCK_ID], [0], [15], [NULL_BLOCK_ID], [2], [9]],
            dtype=torch.int32,
            device="cuda",
        )
        dst_state_batch_indices = torch.tensor(
            [[10], [11], [14], [12], [15], [NULL_BLOCK_ID], [2], [13]],
            dtype=torch.int32,
            device="cuda",
        )

        out = self._run_kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
        )
        out_ref, state_after = selective_state_update_ref(
            state_ref,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z=z,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
        )
        self._asserts(out, out_ref, state, state_after)
        # Padded rows never touch the cache, even where a dst slot is valid.
        for b in [2, 5]:
            self.assertEqual(int(state_batch_indices[b, 0]), NULL_BLOCK_ID)
            dst = int(dst_state_batch_indices[b, 0])
            if dst != NULL_BLOCK_ID:
                self.assertTrue(
                    torch.equal(state[dst], state_ref[dst]),
                    f"padded row {b} must not write dst slot {dst}",
                )
        logging.info("selective_state_update ok: state_batch_indices redirect")

    def test_decode_varlen_multi_token(self):
        # Varlen multi-token decode (cu_seqlens): the fp32 working state
        # persists across tokens and only the final state is written back.
        seqs = [1, 3, 1]
        total = sum(seqs)
        nheads, dim, dstate, ngroups = (
            self.NHEADS,
            self.HEADDIM,
            self.DSTATE,
            self.NGROUPS,
        )
        x, dt, A, B, C, D, dt_bias, z = self._make_inputs(
            total, nheads, dim, dstate, ngroups, torch.bfloat16, tie_hdim=True, seed=2
        )
        cu = [0]
        for seq_len in seqs:
            cu.append(cu[-1] + seq_len)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")
        nslots = 4
        state = torch.randn(nslots, nheads, dim, dstate, device="cuda") * 0.1
        state_ref = state.clone()
        state_batch_indices = torch.tensor(
            [[0], [2], [3]], dtype=torch.int32, device="cuda"
        )

        out = self._run_kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
        )
        out_ref, state_after = selective_state_update_ref(
            state_ref,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z=z,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
        )
        self._asserts(out, out_ref, state, state_after)
        logging.info("selective_state_update ok: varlen multi-token decode")

    def test_decode_spec_decoding(self):
        # Speculative decoding: initial state comes from the slot of the last
        # accepted token, and the state after each token is snapshotted into
        # its own dst slot (no final write-back).
        seqs = [2, 3]
        total = sum(seqs)
        nheads, dim, dstate, ngroups = (
            self.NHEADS,
            self.HEADDIM,
            self.DSTATE,
            self.NGROUPS,
        )
        x, dt, A, B, C, D, dt_bias, z = self._make_inputs(
            total, nheads, dim, dstate, ngroups, torch.bfloat16, tie_hdim=True, seed=3
        )
        cu = [0]
        for seq_len in seqs:
            cu.append(cu[-1] + seq_len)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")
        nslots = 12
        state = torch.randn(nslots, nheads, dim, dstate, device="cuda") * 0.1
        state_ref = state.clone()
        state_batch_indices = torch.tensor(
            [[5, 6, 7], [8, 9, 10]], dtype=torch.int32, device="cuda"
        )
        dst_state_batch_indices = torch.tensor(
            [[5, 6, NULL_BLOCK_ID], [NULL_BLOCK_ID, 9, 10]],
            dtype=torch.int32,
            device="cuda",
        )
        num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32, device="cuda")

        out = self._run_kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
        )
        out_ref, state_after = selective_state_update_ref(
            state_ref,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            z=z,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
        )
        self._asserts(out, out_ref, state, state_after)
        # Slots never addressed by the dst map keep their original values.
        untouched = [0, 1, 2, 3, 4, 7, 8, 11]
        for slot in untouched:
            self.assertTrue(
                torch.equal(state[slot], state_ref[slot]),
                f"slot {slot} must stay untouched in spec decoding",
            )
        logging.info("selective_state_update ok: speculative decoding snapshots")

    def test_decode_non_tie_hdim(self):
        # Full A matrix + per-(head, dim) dt and dt_bias (no TIE_HDIM).
        batch = 4
        nheads, dim, dstate, ngroups = 4, 32, 64, 2
        x, dt, A, B, C, D, dt_bias, z = self._make_inputs(
            batch, nheads, dim, dstate, ngroups, torch.bfloat16, tie_hdim=False, seed=4
        )
        state = torch.randn(batch, nheads, dim, dstate, device="cuda") * 0.1
        state_ref = state.clone()

        out = self._run_kernel(state, x, dt, A, B, C, D, dt_bias, z)
        out_ref, state_after = selective_state_update_ref(
            state_ref, x, dt, A, B, C, D, dt_bias, z=z, dt_softplus=True
        )
        self._asserts(out, out_ref, state, state_after)
        logging.info("selective_state_update ok: non-tied A/dt")

    def test_decode_bf16_state_cache(self):
        # bf16 state cache: the kernel computes in fp32 and casts on store.
        batch = 8
        nheads, dim, dstate, ngroups = (
            self.NHEADS,
            self.HEADDIM,
            self.DSTATE,
            self.NGROUPS,
        )
        x, dt, A, B, C, D, dt_bias, z = self._make_inputs(
            batch, nheads, dim, dstate, ngroups, torch.bfloat16, tie_hdim=True, seed=5
        )
        state = (torch.randn(batch, nheads, dim, dstate, device="cuda") * 0.1).to(
            torch.bfloat16
        )
        state_ref = state.clone()

        out = self._run_kernel(state, x, dt, A, B, C, D, dt_bias, z)
        out_ref, state_after = selective_state_update_ref(
            state_ref, x, dt, A, B, C, D, dt_bias, z=z, dt_softplus=True
        )
        self._asserts(out, out_ref, state, state_after)
        logging.info("selective_state_update ok: bf16 state cache")


if __name__ == "__main__":
    unittest.main()
