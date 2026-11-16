"""Multi-token (MTP target-verify) GDN device-path regression.

CPU half: index semantics of ``speculative_state_indices`` /
``decode_state_indices`` / ``seed_state_segment`` (cross-page placement,
out-of-range -> null block, segment isolation) — runs on any CI.

NPU half: the T > 1 branches of ``causal_conv1d_update`` and
``fused_recurrent_gated_delta_rule`` against the CUDA page contract
(write entry = ``cal_block_idx(length) + t``):

* cross-page seeding and conv/SSM snapshot placement (device path
  bit-matches the host adapter; conv also matches sequential single-token
  steps, recurrent matches them within bf16 rounding);
* partial rejection: accepted-prefix outputs and snapshots are
  T-independent, and a follow-up step keyed at the accepted snapshot
  (engine bookkeeping simulated) reproduces the k-token history bit-exactly;
* padding rows (zeroed block map, page 0 = null slot) leave valid rows
  untouched;
* aclgraph capture/replay with refreshed lengths, including a
  partial-rejection schedule between replays;
* outputs and snapshots match a pure-torch FP32 reference (bf16 tolerance).

The NPU half is skipped automatically when no NPU or no fla_npu wheel is
present.
"""

import unittest

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except Exception:
    _HAS_NPU = False

try:
    from fla_npu.ops import ascendc as _ascendc  # noqa: F401

    _HAS_FLA_NPU = True
except Exception:
    _HAS_FLA_NPU = False


class SpeculativeStateIndicesTest(unittest.TestCase):
    """CPU-only index-semantics tests (run on any CI, no NPU needed).

    Pins the page contract of ``speculative_state_indices`` /
    ``decode_state_indices`` / ``seed_state_segment`` that the NPU-gated
    tests below exercise end to end.
    """

    def setUp(self):
        from rtp_llm.models_py.kernels.ascend.paged_state import (
            decode_state_indices,
            seed_state_segment,
            speculative_state_indices,
        )

        self.decode_state_indices = decode_state_indices
        self.seed_state_segment = seed_state_segment
        self.speculative_state_indices = speculative_state_indices

    def test_cross_page_writes_follow_cuda_contract(self):
        block_map = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32)
        # canonical cross-page case: page size 2, length 3, T = 3
        # -> read page 10, one write page per token (11 / 12 / 13)
        lengths = torch.tensor([3], dtype=torch.int32)
        read_idx, write_idxs = self.speculative_state_indices(
            block_map, lengths, 2, 3)
        self.assertEqual(read_idx.tolist(), [10])
        self.assertEqual([w.tolist() for w in write_idxs],
                         [[11], [12], [13]])
        # a step starting mid-block keeps token 0 on the read page
        lengths = torch.tensor([4], dtype=torch.int32)
        read_idx, write_idxs = self.speculative_state_indices(
            block_map, lengths, 2, 3)
        self.assertEqual(read_idx.tolist(), [11])
        self.assertEqual([w.tolist() for w in write_idxs],
                         [[11], [12], [13]])

    def test_out_of_range_writes_land_on_null_block(self):
        block_map = torch.tensor([[10, 11]], dtype=torch.int32)
        lengths = torch.tensor([3], dtype=torch.int32)
        read_idx, write_idxs = self.speculative_state_indices(
            block_map, lengths, 2, 3)
        self.assertEqual(read_idx.tolist(), [10])
        self.assertEqual([w.tolist() for w in write_idxs],
                         [[11], [0], [0]])
        # over-long sequences clamp the read column to the last entry
        lengths = torch.tensor([10], dtype=torch.int32)
        read_idx, _ = self.speculative_state_indices(
            block_map, lengths, 2, 1)
        self.assertEqual(read_idx.tolist(), [11])

    def test_decode_indices_cross_page_semantics(self):
        block_map = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        lengths = torch.tensor([4, 5], dtype=torch.int32)
        read_idx, write_idx = self.decode_state_indices(block_map, lengths, 4)
        # read = (L - 2) // page, write = (L - 1) // page: row 0 stays
        # in-page, row 1 crosses a boundary
        self.assertEqual(read_idx.tolist(), [10, 20])
        self.assertEqual(write_idx.tolist(), [10, 21])

    def test_seed_state_segment_copies_only_its_segment(self):
        ssm_elems, conv_elems = 8, 6
        stride = ssm_elems + conv_elems
        base = torch.arange(4 * stride, dtype=torch.float32).reshape(4, stride)
        ssm, conv = base[:, :ssm_elems], base[:, ssm_elems:]
        read_idx = torch.tensor([0, 2])
        write_idx = torch.tensor([1, 3])
        self.seed_state_segment(ssm, read_idx, write_idx)
        self.assertTrue(torch.equal(ssm[1], base[0, :ssm_elems]))
        self.assertTrue(torch.equal(ssm[3], base[2, :ssm_elems]))
        # the conv sibling segment of the destination pages is untouched
        self.assertTrue(torch.equal(conv[1], base[1, ssm_elems:]))
        self.assertTrue(torch.equal(conv[3], base[3, ssm_elems:]))
        # same-page rows are no-ops
        before = base.clone()
        self.seed_state_segment(
            conv, torch.tensor([1, 3]), torch.tensor([1, 3]))
        self.assertTrue(torch.equal(base, before))


def _build_pool(pages, hv, dv, dk, state_len, dim, dtype=torch.bfloat16,
                seed=0):
    """Paged pool mimicking typed_storage_view: ssm then conv segment."""

    ssm_elems = hv * dv * dk
    conv_elems = state_len * dim
    stride = ssm_elems + conv_elems
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(
        pages, stride, dtype=torch.float32, generator=gen) * 0.05
    base = base.to(dtype).to("npu")
    ssm = torch.empty(0, dtype=dtype, device="npu")
    ssm.set_(base.untyped_storage(), 0, (pages, hv, dv, dk),
              (stride, dv * dk, dk, 1))
    conv = torch.empty(0, dtype=dtype, device="npu")
    conv.set_(base.untyped_storage(), ssm_elems, (pages, state_len, dim),
              (stride, dim, 1))
    return base, ssm, conv


@unittest.skipUnless(_HAS_NPU and _HAS_FLA_NPU,
                     "requires Ascend NPU + fla_npu")
class MtpTokensDevicePathTest(unittest.TestCase):
    """Regression for the multi-token (target-verify) device paths.

    Batch layout: rows 0-2 are valid requests covering every cross-page
    seeding variant (crossing at token 0 / mid-step / coinciding entries),
    row ``PAD_ROW`` simulates a graph-bucket padding request in the
    padding-specific test.  Pages are exclusive per row (duplicate cache
    indices across rows are impossible in production).
    """

    B, T, K_ACCEPT = 4, 4, 2
    # the AscendC operator requires Dk == Dv == 128 exactly
    NK, HV, DK, DV = 2, 4, 128, 128
    DIM = (2 * NK + HV) * DK
    STATE_LEN, PAGES, PAGE = 3, 32, 4
    MAX_BLOCKS, PAD_ROW = 8, 3
    VALID = slice(0, 3)

    def setUp(self):
        from rtp_llm.models_py.kernels.ascend.causal_conv1d import (
            causal_conv1d_update,
        )
        from rtp_llm.models_py.kernels.ascend.common import fused_gdn_gating
        from rtp_llm.models_py.kernels.ascend.recurrent import (
            fused_recurrent_gated_delta_rule,
        )

        torch.manual_seed(1234)
        self._conv_update = causal_conv1d_update
        self._recurrent = fused_recurrent_gated_delta_rule
        self.scale = self.DK ** -0.5

        # lengths_plus_1: committed length before the step is L - 1.
        # row0: L=5 (C=4, C%4=0) -> read entry 0, write entries 1..4 (cross
        #       at token 0); row1: L=7 (C=6, C%4=2) -> token 0 stays
        #       in-page, token 1 crosses; row2: L=8 (C=7, C%4=3) -> crossing
        #       at token 1; row3: padding row (L=10).
        self.lengths = torch.tensor([5, 7, 8, 10], dtype=torch.int32,
                                    device="npu")
        self.block_map = torch.zeros(self.B, self.MAX_BLOCKS,
                                    dtype=torch.int32, device="npu")
        for i in range(self.B):
            for c in range(self.MAX_BLOCKS):
                self.block_map[i, c] = 8 * i + 1 + c

        self.alog = (torch.randn(self.HV) * 0.1).to(torch.bfloat16).to("npu")
        self.dt_bias = (torch.randn(self.HV) * 0.05)
        self.dt_bias = self.dt_bias.to(torch.bfloat16).to("npu")
        self.weight = (torch.randn(self.DIM, self.STATE_LEN + 1) * 0.2)
        self.weight = self.weight.to(torch.bfloat16).to("npu")
        inputs = self._make_inputs(self.T, seed=77)
        self.q, self.k, self.v = inputs["q"], inputs["k"], inputs["v"]
        self.g, self.beta = inputs["g"], inputs["beta"]
        self.x_conv = inputs["x"]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_inputs(self, token_count, seed):
        """Random (q, k, v, g, beta, x) for a ``token_count``-token step."""
        from rtp_llm.models_py.kernels.ascend.common import fused_gdn_gating

        torch.manual_seed(seed)
        q = (torch.randn(self.B, token_count, self.NK, self.DK) * 0.3)
        k = (torch.randn(self.B, token_count, self.NK, self.DK) * 0.3)
        v = (torch.randn(self.B, token_count, self.HV, self.DV) * 0.3)
        a = (torch.randn(self.B, token_count, self.HV) * 0.3)
        b = (torch.randn(self.B, token_count, self.HV) * 0.3)
        x = (torch.randn(self.B, self.DIM, token_count) * 0.3)
        q = q.to(torch.bfloat16).to("npu")
        k = k.to(torch.bfloat16).to("npu")
        v = v.to(torch.bfloat16).to("npu")
        a, b = a.to(torch.bfloat16).to("npu"), b.to(torch.bfloat16).to("npu")
        x = x.to(torch.bfloat16).to("npu")
        g, beta = fused_gdn_gating(
            self.alog, a.reshape(-1, self.HV), b.reshape(-1, self.HV),
            self.dt_bias,
        )
        g = g.view(self.B, token_count, self.HV)
        beta = beta.view(self.B, token_count, self.HV)
        return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "x": x}

    def _slice_inputs(self, inputs, token_count):
        return {
            "q": inputs["q"][:, :token_count].contiguous(),
            "k": inputs["k"][:, :token_count].contiguous(),
            "v": inputs["v"][:, :token_count].contiguous(),
            "g": inputs["g"][:, :token_count].contiguous(),
            "beta": inputs["beta"][:, :token_count].contiguous(),
            "x": inputs["x"][:, :, :token_count].contiguous(),
        }

    def _token_inputs(self, inputs, t):
        """One-token slice of ``inputs`` for a single-token decode step."""
        return {
            "q": inputs["q"][:, t:t + 1].contiguous(),
            "k": inputs["k"][:, t:t + 1].contiguous(),
            "v": inputs["v"][:, t:t + 1].contiguous(),
            "g": inputs["g"][:, t:t + 1].contiguous(),
            "beta": inputs["beta"][:, t:t + 1].contiguous(),
            "x": inputs["x"][:, :, t:t + 1].contiguous(),
        }

    def _run_step(self, inputs, conv, ssm, block_map=None, lengths=None):
        block_map = self.block_map if block_map is None else block_map
        lengths = self.lengths if lengths is None else lengths
        out_c = self._conv_update(
            inputs["x"], conv, self.weight, bias=None, activation="silu",
            block_map=block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=lengths,
        )
        out_r, _ = self._recurrent(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
            scale=self.scale, initial_state=ssm, inplace_final_state=True,
            block_map=block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=lengths, use_qk_l2norm_in_kernel=True,
        )
        return out_c, out_r

    def _pages(self, block_map, lengths, token_count):
        """Host replica of the CUDA page contract (read + per-token writes)."""
        mapping = block_map.detach().cpu().tolist()
        lens = lengths.detach().cpu().tolist()
        reads, writes = [], []
        for bi, length in enumerate(lens):
            reads.append(mapping[bi][max(length - 2, 0) // self.PAGE])
            start = (length - 1) // self.PAGE
            writes.append([mapping[bi][start + t] for t in range(token_count)])
        return reads, writes

    def _relocate_accepted_snapshot(self, base, block_map, lengths, accepted):
        """Engine post-accept bookkeeping: move the last accepted token's
        snapshot (speculative entry) to the natural read entry of the next
        step so the ordinary read formula ``(length + accepted - 2) // page``
        lands on the accepted state."""
        mapping = block_map.detach().cpu().tolist()
        lens = lengths.detach().cpu().tolist()
        for bi, length in enumerate(lens):
            natural = mapping[bi][(length + accepted - 2) // self.PAGE]
            snapshot = mapping[bi][(length - 1) // self.PAGE + accepted - 1]
            if natural != snapshot:
                base[natural].copy_(base[snapshot])

    def _touched_pages(self, block_map, lengths, token_count):
        reads, writes = self._pages(block_map, lengths, token_count)
        pages = set()
        for bi in range(lengths.shape[0]):
            pages.add(reads[bi])
            pages.update(writes[bi])
        return sorted(pages)

    # ------------------------------------------------------------------
    # 1. device path vs host-metadata adapter (wiring + cross-page seeding)
    # ------------------------------------------------------------------

    def _host_adapter(self, base, ssm, conv, inputs, block_map, lengths):
        """Replica of the pre-plan-C eager host-metadata adapters, T tokens."""
        from fla_npu.ops import ascendc
        from fla_npu.ops.ascendc import npu_causal_conv1d
        from rtp_llm.models_py.kernels.ascend.linear_attention import (
            l2norm_fwd,
        )

        batch, token_count = inputs["q"].shape[:2]
        reads, writes = self._pages(block_map, lengths, token_count)
        for bi in range(batch):
            if reads[bi] != writes[bi][0]:
                ssm[writes[bi][0]].copy_(ssm[reads[bi]])
                conv[writes[bi][0]].copy_(conv[reads[bi]])

        out_conv_tokens = []
        for t in range(token_count):
            for bi in range(batch):
                if t > 0 and writes[bi][t] != writes[bi][t - 1]:
                    conv[writes[bi][t]].copy_(conv[writes[bi][t - 1]])
            out_conv_tokens.append(npu_causal_conv1d(
                x=inputs["x"][:, :, t].contiguous(),
                weight=self.weight.t().contiguous(), bias=None,
                conv_states=conv,
                cache_indices=[row[t] for row in writes],
                activation_mode=1, pad_slot_id=-1, run_mode=1, head_num=0,
            ))
        out_c = torch.stack(out_conv_tokens, dim=-1)

        qn = l2norm_fwd(inputs["q"]).reshape(-1, self.NK, self.DK)
        kn = l2norm_fwd(inputs["k"]).reshape(-1, self.NK, self.DK)
        asl = torch.tensor([0] + [token_count] * batch, dtype=torch.int32,
                           device="npu")
        ssi = torch.tensor([p for row in writes for p in row],
                           dtype=torch.int32, device="npu")
        res = ascendc.npu_recurrent_gated_delta_rule(
            qn.to(torch.bfloat16), kn.to(torch.bfloat16),
            inputs["v"].reshape(-1, self.HV, self.DV),
            ssm,
            beta=inputs["beta"].reshape(-1, self.HV).to(torch.bfloat16),
            scale=self.scale,
            actual_seq_lengths=asl,
            ssm_state_indices=ssi,
            g=inputs["g"].reshape(-1, self.HV).float(),
        )
        out_r = res[0] if isinstance(res, (tuple, list)) else res
        return out_c, out_r.reshape(batch, token_count, *out_r.shape[1:])

    def test_tokens_step_matches_host_adapter(self):
        inputs = {"q": self.q, "k": self.k, "v": self.v,
                  "g": self.g, "beta": self.beta, "x": self.x_conv}
        base_h, ssm_h, conv_h = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=11)
        out_c_h, out_r_h = self._host_adapter(
            base_h, ssm_h, conv_h, inputs, self.block_map, self.lengths)

        base_d, ssm_d, conv_d = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=11)
        out_c_d, out_r_d = self._run_step(inputs, conv_d, ssm_d)

        self.assertTrue(torch.equal(out_c_h, out_c_d))
        self.assertTrue(torch.equal(out_r_h, out_r_d))
        self.assertTrue(torch.equal(base_h, base_d))

    # ------------------------------------------------------------------
    # 2. partial rejection: accepted-prefix snapshots are T-independent
    # ------------------------------------------------------------------

    def test_partial_rejection_prefix_is_token_count_independent(self):
        inputs = {"q": self.q, "k": self.k, "v": self.v,
                  "g": self.g, "beta": self.beta, "x": self.x_conv}
        base_t, ssm_t, conv_t = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=31)
        out_c_t, out_r_t = self._run_step(inputs, conv_t, ssm_t)

        # accepted = 1: only position 0 survives (reject 1..T-1);
        # accepted = 2: mid-step rejection.  Both must see the same prefix.
        for accepted in (1, self.K_ACCEPT):
            with self.subTest(accepted=accepted):
                base_k, ssm_k, conv_k = _build_pool(
                    self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN,
                    self.DIM, seed=31)
                out_c_k, out_r_k = self._run_step(
                    self._slice_inputs(inputs, accepted), conv_k, ssm_k)

                # accepted-prefix outputs are bit-identical regardless of T
                self.assertTrue(
                    torch.equal(out_c_t[self.VALID, :, :accepted],
                                out_c_k[self.VALID]))
                self.assertTrue(
                    torch.equal(out_r_t[self.VALID, :accepted],
                                out_r_k[self.VALID]))
                # so are the accepted tokens' conv/SSM snapshot pages
                _, writes = self._pages(self.block_map, self.lengths,
                                        accepted)
                for bi in range(self.B - 1):
                    for t in range(accepted):
                        page = writes[bi][t]
                        self.assertTrue(
                            torch.equal(base_t[page], base_k[page]),
                            f"row {bi} token {t} snapshot page {page}"
                            " diverged",
                        )
                # the rejected tail pages exist only in the T-step pool:
                # the k-step pool keeps them pristine
                _, writes_t = self._pages(self.block_map, self.lengths,
                                          self.T)
                base_pristine, _, _ = _build_pool(
                    self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN,
                    self.DIM, seed=31)
                for bi in range(self.B - 1):
                    for t in range(accepted, self.T):
                        page = writes_t[bi][t]
                        self.assertTrue(
                            torch.equal(base_k[page], base_pristine[page]))

    # ------------------------------------------------------------------
    # 3. continuation after partial rejection (state recovery)
    # ------------------------------------------------------------------

    def test_continuation_after_partial_rejection(self):
        inputs = {"q": self.q, "k": self.k, "v": self.v,
                  "g": self.g, "beta": self.beta, "x": self.x_conv}
        for accepted in (1, self.K_ACCEPT):
            with self.subTest(accepted=accepted):
                # rejected run: T-token verify, only `accepted` tokens kept
                base_t, ssm_t, conv_t = _build_pool(
                    self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN,
                    self.DIM, seed=41)
                self._run_step(inputs, conv_t, ssm_t)
                self._relocate_accepted_snapshot(
                    base_t, self.block_map, self.lengths, accepted)

                # reference run: only the accepted tokens were processed
                base_k, ssm_k, conv_k = _build_pool(
                    self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN,
                    self.DIM, seed=41)
                self._run_step(
                    self._slice_inputs(inputs, accepted), conv_k, ssm_k)
                self._relocate_accepted_snapshot(
                    base_k, self.block_map, self.lengths, accepted)

                # next round continues from the accepted state on both pools
                cont = self._make_inputs(token_count=2, seed=88)
                lengths2 = self.lengths + accepted
                out_c_t, out_r_t = self._run_step(
                    cont, conv_t, ssm_t, lengths=lengths2)
                out_c_k, out_r_k = self._run_step(
                    cont, conv_k, ssm_k, lengths=lengths2)

                self.assertTrue(torch.equal(out_c_t, out_c_k))
                self.assertTrue(torch.equal(out_r_t, out_r_k))
                for page in self._touched_pages(self.block_map, lengths2, 2):
                    self.assertTrue(
                        torch.equal(base_t[page], base_k[page]),
                        f"page {page} diverged after continuation")

    # ------------------------------------------------------------------
    # 4. padding rows are isolated
    # ------------------------------------------------------------------

    def test_padding_row_is_isolated(self):
        """Graph-bucket padding rows (block map zeroed -> page 0) must not
        disturb the valid rows: page 0 is the null slot, outputs of null rows
        are undefined and discarded in production."""
        inputs = {"q": self.q, "k": self.k, "v": self.v,
                  "g": self.g, "beta": self.beta, "x": self.x_conv}
        base_n, ssm_n, conv_n = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=21)
        out_c_n, out_r_n = self._run_step(inputs, conv_n, ssm_n)

        base_p, ssm_p, conv_p = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=21)
        block_map = self.block_map.clone()
        block_map[self.PAD_ROW].zero_()
        out_c_p, out_r_p = self._run_step(inputs, conv_p, ssm_p,
                                          block_map=block_map)

        self.assertTrue(torch.equal(out_c_n[self.VALID], out_c_p[self.VALID]))
        self.assertTrue(torch.equal(out_r_n[self.VALID], out_r_p[self.VALID]))
        _, writes = self._pages(self.block_map, self.lengths, self.T)
        for bi in range(self.B - 1):
            for page in writes[bi]:
                self.assertTrue(
                    torch.equal(base_n[page], base_p[page]),
                    f"page {page} of valid row {bi} diverged",
                )
        # the padding row's would-be write pages stay pristine in the
        # padding run (conv skips the null slot; recurrent writes page 0)
        base_pristine, _, _ = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=21)
        for page in writes[self.PAD_ROW]:
            self.assertTrue(torch.equal(base_p[page], base_pristine[page]))

    # ------------------------------------------------------------------
    # 5. aclgraph capture / replay with a partial-rejection schedule
    # ------------------------------------------------------------------

    def test_graph_capture_replay_matches_eager(self):
        inputs = {"q": self.q, "k": self.k, "v": self.v,
                  "g": self.g, "beta": self.beta, "x": self.x_conv}
        base_g, ssm_g, conv_g = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=51)
        base_e = base_g.clone()
        ssm_e = torch.empty(0, dtype=base_e.dtype, device="npu")
        ssm_e.set_(base_e.untyped_storage(), 0, ssm_g.shape, ssm_g.stride())
        conv_e = torch.empty(0, dtype=base_e.dtype, device="npu")
        conv_e.set_(base_e.untyped_storage(), conv_g.storage_offset(),
                    conv_g.shape, conv_g.stride())

        # eager reference: verify at L0, accept K_ACCEPT of T, continue at
        # L0 + K_ACCEPT (relocation between steps, engine-style)
        len_e = self.lengths.clone()
        out_c_e1, out_r_e1 = self._run_step(
            inputs, conv_e, ssm_e, lengths=len_e)
        self._relocate_accepted_snapshot(
            base_e, self.block_map, len_e, self.K_ACCEPT)
        len_e = len_e + self.K_ACCEPT
        out_c_e2, out_r_e2 = self._run_step(
            inputs, conv_e, ssm_e, lengths=len_e)

        # capture one multi-token step, replay twice with refreshed lengths
        saved = base_g.clone()
        s = torch.npu.Stream()
        s.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(s):
            for _ in range(2):
                self._run_step(inputs, conv_g, ssm_g)
        torch.npu.current_stream().wait_stream(s)
        graph = torch.npu.NPUGraph()
        len_g = self.lengths.clone()
        with torch.npu.graph(graph, stream=s):
            out_c_g, out_r_g = self._run_step(
                inputs, conv_g, ssm_g, lengths=len_g)
        torch.npu.synchronize()
        base_g.copy_(saved)

        len_g.copy_(self.lengths)
        graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.equal(out_c_g, out_c_e1))
        self.assertTrue(torch.equal(out_r_g, out_r_e1))

        self._relocate_accepted_snapshot(
            base_g, self.block_map, self.lengths, self.K_ACCEPT)
        len_g.copy_(self.lengths + self.K_ACCEPT)
        graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.equal(out_c_g, out_c_e2))
        self.assertTrue(torch.equal(out_r_g, out_r_e2))
        self.assertTrue(torch.equal(base_g, base_e))

    # ------------------------------------------------------------------
    # 6. pure-torch FP32 reference (outputs and conv/SSM snapshots)
    # ------------------------------------------------------------------

    def test_multi_token_matches_torch_reference(self):
        inputs = {"q": self.q, "k": self.k, "v": self.v,
                  "g": self.g, "beta": self.beta, "x": self.x_conv}
        base, ssm, conv = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=61)
        pre_base = base.clone()
        pre_ssm = torch.empty(0, dtype=pre_base.dtype, device="npu")
        pre_ssm.set_(pre_base.untyped_storage(), 0, ssm.shape, ssm.stride())
        pre_conv = torch.empty(0, dtype=pre_base.dtype, device="npu")
        pre_conv.set_(pre_base.untyped_storage(), conv.storage_offset(),
                      conv.shape, conv.stride())
        out_c, out_r = self._run_step(inputs, conv, ssm)
        reads, writes = self._pages(self.block_map, self.lengths, self.T)

        qn = F.normalize(inputs["q"].float(), dim=-1)          # (B,T,NK,DK)
        kn = F.normalize(inputs["k"].float(), dim=-1)
        vf = inputs["v"].float()                                # (B,T,HV,DV)
        gf = inputs["g"].float()                                # (B,T,HV)
        bf = inputs["beta"].float()
        weight_t = self.weight.t().float()                      # (width, DIM)

        out_r_ref = torch.zeros(self.B, self.T, self.HV, self.DV)
        ssm_snaps = torch.zeros(self.B, self.T, self.HV, self.DV, self.DK)
        out_c_ref = torch.zeros(self.B, self.DIM, self.T)
        conv_snaps = torch.zeros(self.B, self.T, self.STATE_LEN, self.DIM)
        head_group = self.HV // self.NK
        for bi in range(self.B):
            h = pre_ssm[reads[bi]].float().clone()             # (HV,DV,DK)
            st = pre_conv[reads[bi]].float().clone()          # (state_len,DIM)
            for t in range(self.T):
                for vh in range(self.HV):
                    kh = vh // head_group
                    h[vh] = h[vh] * torch.exp(gf[bi, t, vh])
                    kv = kn[bi, t, kh]
                    delta = (vf[bi, t, vh] - h[vh] @ kv) * bf[bi, t, vh]
                    h[vh] = h[vh] + torch.outer(delta, kv)
                    out_r_ref[bi, t, vh] = h[vh] @ (qn[bi, t, kh] * self.scale)
                    ssm_snaps[bi, t, vh] = h[vh]
                x_t = inputs["x"][bi, :, t].float()
                window = torch.cat([st, x_t[None]], 0)
                out_c_ref[bi, :, t] = F.silu((window * weight_t).sum(0))
                st = window[1:]
                conv_snaps[bi, t] = st

        def rel(name, actual, ref):
            actual = actual.float().cpu()
            diff = (actual - ref).abs().max().item()
            denom = max(ref.abs().max().item(), 1e-6)
            error = diff / denom
            self.assertLess(error, 0.03, f"{name} rel error {error:.4g}")

        rel("recurrent output", out_r, out_r_ref)
        rel("conv output", out_c, out_c_ref)
        for bi in range(self.B):
            for t in range(self.T):
                rel(f"ssm snapshot row {bi} token {t}",
                    ssm[writes[bi][t]], ssm_snaps[bi, t])
                rel(f"conv snapshot row {bi} token {t}",
                    conv[writes[bi][t]], conv_snaps[bi, t])

    # ------------------------------------------------------------------
    # 7. multi-token vs sequential single-token steps (T==1 reference)
    # ------------------------------------------------------------------

    def test_tokens_step_matches_sequential_single_token_steps(self):
        """The T-token step must agree with the production single-token
        path run token by token: conv bit-exact (identical per-token
        operator chain), recurrent within bf16 state-reload rounding."""
        inputs = {"q": self.q, "k": self.k, "v": self.v,
                  "g": self.g, "beta": self.beta, "x": self.x_conv}
        base_m, ssm_m, conv_m = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=71)
        out_c_m, out_r_m = self._run_step(inputs, conv_m, ssm_m)

        base_s, ssm_s, conv_s = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM,
            seed=71)
        outs_c, outs_r = [], []
        for t in range(self.T):
            oc, orr = self._run_step(
                self._token_inputs(inputs, t), conv_s, ssm_s,
                lengths=self.lengths + t)
            outs_c.append(oc)
            outs_r.append(orr)

        for t in range(self.T):
            self.assertTrue(torch.equal(out_c_m[:, :, t], outs_c[t][:, :, 0]),
                            f"conv output token {t}")
        rel = 0.0
        for t in range(self.T):
            diff = (out_r_m[:, t].float()
                    - outs_r[t][:, 0].float()).abs().max().item()
            denom = max(outs_r[t][:, 0].float().abs().max().item(), 1e-6)
            rel = max(rel, diff / denom)
        self.assertLess(rel, 0.03, f"recurrent output rel error {rel:.4g}")

        # final state after the last token: multi-token snapshot at the
        # speculative entry vs the single-token loop's natural entry
        _, writes_m = self._pages(self.block_map, self.lengths, self.T)
        lens = self.lengths.detach().cpu().tolist()
        mapping = self.block_map.detach().cpu().tolist()
        for bi in range(self.B):
            final_m = writes_m[bi][self.T - 1]
            final_s = mapping[bi][(lens[bi] + self.T - 2) // self.PAGE]
            self.assertTrue(torch.equal(conv_m[final_m], conv_s[final_s]),
                            f"conv state row {bi}")
            diff = (ssm_m[final_m].float()
                    - ssm_s[final_s].float()).abs().max().item()
            denom = max(ssm_s[final_s].float().abs().max().item(), 1e-6)
            self.assertLess(
                diff / denom, 0.03,
                f"ssm state row {bi} rel error {diff / denom:.4g}")


if __name__ == "__main__":
    unittest.main()
