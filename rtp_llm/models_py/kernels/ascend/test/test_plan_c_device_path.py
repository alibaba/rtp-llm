"""Plan-C single-consumer GDN decode tests (real NPU required).

Verifies that the device-metadata AscendC paths inside
``causal_conv1d_update`` / ``fused_recurrent_gated_delta_rule``:

* match the host-metadata adapter (the pre-plan-C eager logic) bit-exact on a
  strided typed_storage_view-style pool, including cross-block seeding and
  padding rows (block-map row zeroed, page 0 = null slot);
* are aclgraph-capturable and replayable with refreshed block_map /
  sequence_lengths contents, reproducing the eager multi-step evolution
  bit-exactly;
* stay close to a pure-torch GDN/conv reference (bf16 tolerance).

Skipped automatically when no NPU or no fla_npu wheel is present.
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


def _build_pool(pages, hv, dv, dk, state_len, dim, dtype=torch.bfloat16, seed=0):
    """Paged pool mimicking typed_storage_view: ssm segment then conv segment."""

    ssm_elems = hv * dv * dk
    conv_elems = state_len * dim
    stride = ssm_elems + conv_elems
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = (torch.randn(pages, stride, dtype=torch.float32, generator=gen) * 0.05)
    base = base.to(dtype).to("npu")
    ssm = torch.empty(0, dtype=dtype, device="npu")
    ssm.set_(base.untyped_storage(), 0, (pages, hv, dv, dk), (stride, dv * dk, dk, 1))
    conv = torch.empty(0, dtype=dtype, device="npu")
    conv.set_(base.untyped_storage(), ssm_elems, (pages, state_len, dim),
              (stride, dim, 1))
    return base, ssm, conv


@unittest.skipUnless(_HAS_NPU and _HAS_FLA_NPU, "requires Ascend NPU + fla_npu")
class PlanCDevicePathTest(unittest.TestCase):
    B, NK, HV, DK, DV = 5, 4, 8, 128, 128
    DIM = (2 * NK + HV) * DK
    STATE_LEN, PAGES, PAGE = 3, 20, 8
    PAD_ROW = 4  # simulated graph-bucket padding row (block map zeroed)

    def setUp(self):
        from rtp_llm.models_py.kernels.ascend.causal_conv1d import (
            causal_conv1d_update,
        )
        from rtp_llm.models_py.kernels.ascend.recurrent import (
            fused_recurrent_gated_delta_rule,
        )

        torch.manual_seed(1234)
        self._conv_update = causal_conv1d_update
        self._recurrent = fused_recurrent_gated_delta_rule
        self.scale = self.DK ** -0.5

        self.q = (torch.randn(self.B, 1, self.NK, self.DK) * 0.3).to(torch.bfloat16).to("npu")
        self.k = (torch.randn(self.B, 1, self.NK, self.DK) * 0.3).to(torch.bfloat16).to("npu")
        self.v = (torch.randn(self.B, 1, self.HV, self.DV) * 0.3).to(torch.bfloat16).to("npu")
        self.a = (torch.randn(self.B, 1, self.HV) * 0.3).to(torch.bfloat16).to("npu")
        self.b = (torch.randn(self.B, 1, self.HV) * 0.3).to(torch.bfloat16).to("npu")
        self.alog = (torch.randn(self.HV) * 0.1).to(torch.bfloat16).to("npu")
        self.dt_bias = (torch.randn(self.HV) * 0.05).to(torch.bfloat16).to("npu")
        self.weight = (torch.randn(self.DIM, 4) * 0.2).to(torch.bfloat16).to("npu")
        self.xq = (torch.randn(self.B, self.DIM) * 0.3).to(torch.bfloat16).to("npu")
        # (batch, dim, 1) as the model passes it
        self.x_conv = self.xq.unsqueeze(-1)

        # lengths chosen so row0 stays in-block, row1 crosses a block edge,
        # row2 uses its second page for both read and write; row4 (PAD_ROW)
        # gets a zeroed block map in the padding-specific test only — rows
        # addressed to page 0 (null_block_id) have an undefined *output* by
        # design (padding rows are discarded), so the bit-exact tests use
        # valid pages everywhere.  Pages are exclusive per row: duplicate
        # cache indices across rows are impossible in production and their
        # final write ordering differs between the host and device metadata
        # channels.
        self.lengths = torch.tensor([7, 9, 18, 8, 10], dtype=torch.int32, device="npu")
        self.block_map = torch.zeros(self.B, 5, dtype=torch.int32, device="npu")
        for i in range(self.B):
            for c in range(5):
                self.block_map[i, c] = 3 * i + 1 + c

        # gating takes flattened [B*T, HV] and returns [1, B*T, HV] (the model
        # then views it as [B, T, HV])
        from rtp_llm.models_py.kernels.ascend.common import fused_gdn_gating

        g, beta = fused_gdn_gating(
            self.alog,
            self.a.reshape(self.B, self.HV),
            self.b.reshape(self.B, self.HV),
            self.dt_bias,
        )
        self.g = g.view(self.B, 1, self.HV)
        self.beta = beta.view(self.B, 1, self.HV)

    def _host_adapter(self, base, ssm, conv):
        """Replica of the pre-plan-C eager host-metadata adapters."""

        from fla_npu.ops import ascendc
        from fla_npu.ops.ascendc import npu_causal_conv1d

        mapping = self.block_map.detach().cpu().tolist()
        lens = self.lengths.detach().cpu().tolist()
        read_pages, write_pages = [], []
        for bi, length in enumerate(lens):
            rb = max(length - 2, 0) // self.PAGE
            wb = (length - 1) // self.PAGE
            read_pages.append(mapping[bi][rb])
            write_pages.append(mapping[bi][wb])
        for bi in range(self.B):
            if read_pages[bi] > 0 and write_pages[bi] > 0 and read_pages[bi] != write_pages[bi]:
                ssm[write_pages[bi]].copy_(ssm[read_pages[bi]])
                conv[write_pages[bi]].copy_(conv[read_pages[bi]])
        out_c = npu_causal_conv1d(
            x=self.xq.clone(), weight=self.weight.t().contiguous(), bias=None,
            conv_states=conv,
            cache_indices=write_pages,
            activation_mode=1, pad_slot_id=-1, run_mode=1, head_num=0,
        )
        qn = F.normalize(self.q.float(), dim=-1)
        kn = F.normalize(self.k.float(), dim=-1)
        asl = torch.tensor([0] + [1] * self.B, dtype=torch.int32, device="npu")
        res = ascendc.npu_recurrent_gated_delta_rule(
            qn.to(torch.bfloat16).reshape(self.B, self.NK, self.DK),
            kn.to(torch.bfloat16).reshape(self.B, self.NK, self.DK),
            self.v.reshape(self.B, self.HV, self.DV),
            ssm,
            beta=self.beta.reshape(self.B, -1).to(torch.bfloat16),
            scale=self.scale,
            actual_seq_lengths=asl,
            ssm_state_indices=torch.tensor(write_pages, dtype=torch.int32, device="npu"),
            g=self.g.reshape(self.B, -1).float(),
        )
        out_r = res[0] if isinstance(res, (tuple, list)) else res
        return out_c, out_r, base

    def test_device_path_matches_host_adapter(self):
        base_h, ssm_h, conv_h = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM, seed=11)
        out_c_h, out_r_h, _ = self._host_adapter(base_h, ssm_h, conv_h)

        # independent pristine pool with the same seed (the host adapter above
        # mutates its own pool in place)
        base_d, ssm_d, conv_d = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM, seed=11)

        out_c_d = self._conv_update(
            self.x_conv, conv_d, self.weight, bias=None, activation="silu",
            block_map=self.block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths,
        ).squeeze(-1)
        out_r_d, _ = self._recurrent(
            self.q, self.k, self.v, self.g, self.beta,
            scale=self.scale, initial_state=ssm_d, inplace_final_state=True,
            block_map=self.block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths, use_qk_l2norm_in_kernel=True,
        )
        out_r_d = out_r_d.reshape(self.B, out_r_d.shape[-2], out_r_d.shape[-1])

        self.assertTrue(torch.equal(out_c_h, out_c_d))
        self.assertTrue(torch.equal(out_r_h, out_r_d))
        self.assertTrue(torch.equal(base_h, base_d))

    def test_padding_row_is_isolated(self):
        """Graph-bucket padding rows (block map zeroed -> page 0) must not
        disturb the valid rows: page 0 is the null slot, outputs of null rows
        are undefined and discarded in production."""

        base_n, ssm_n, conv_n = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM, seed=21)
        out_c_n = self._conv_update(
            self.x_conv, conv_n, self.weight, bias=None, activation="silu",
            block_map=self.block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths,
        )
        out_r_n, _ = self._recurrent(
            self.q, self.k, self.v, self.g, self.beta,
            scale=self.scale, initial_state=ssm_n, inplace_final_state=True,
            block_map=self.block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths, use_qk_l2norm_in_kernel=True,
        )
        out_r_n = out_r_n.reshape(self.B, out_r_n.shape[-2], out_r_n.shape[-1])

        # independent pristine pool with the same seed; only the padding row's
        # block map differs
        base_p, ssm_p, conv_p = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM, seed=21)
        block_map = self.block_map.clone()
        block_map[self.PAD_ROW].zero_()
        out_c_p = self._conv_update(
            self.x_conv, conv_p, self.weight, bias=None, activation="silu",
            block_map=block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths,
        )
        out_r_p, _ = self._recurrent(
            self.q, self.k, self.v, self.g, self.beta,
            scale=self.scale, initial_state=ssm_p, inplace_final_state=True,
            block_map=block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths, use_qk_l2norm_in_kernel=True,
        )
        out_r_p = out_r_p.reshape(self.B, out_r_p.shape[-2], out_r_p.shape[-1])

        valid = slice(0, self.PAD_ROW)
        self.assertTrue(torch.equal(out_c_n[valid], out_c_p[valid]))
        self.assertTrue(torch.equal(out_r_n[valid], out_r_p[valid]))
        # valid rows' write pages evolve identically in both runs
        mapping = self.block_map.detach().cpu().tolist()
        lens = self.lengths.detach().cpu().tolist()
        for i in range(self.PAD_ROW):
            write_page = mapping[i][(lens[i] - 1) // self.PAGE]
            self.assertTrue(
                torch.equal(base_n[write_page], base_p[write_page]),
                f"page {write_page} of valid row {i} diverged",
            )
        # the padding row's would-be write page is untouched by the padding
        # run (conv skips the null slot; recurrent writes page 0 only)
        pad_write = mapping[self.PAD_ROW][(lens[self.PAD_ROW] - 1) // self.PAGE]
        base_pristine, _, _ = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM, seed=21)
        self.assertTrue(torch.equal(base_p[pad_write], base_pristine[pad_write]))

    def test_graph_capture_replay_matches_eager(self):
        base_g, ssm_g, conv_g = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM, seed=12)
        base_e = base_g.clone()
        ssm_e = torch.empty(0, dtype=base_e.dtype, device="npu")
        ssm_e.set_(base_e.untyped_storage(), 0, ssm_g.shape, ssm_g.stride())
        conv_e = torch.empty(0, dtype=base_e.dtype, device="npu")
        conv_e.set_(base_e.untyped_storage(), conv_g.storage_offset(),
                    conv_g.shape, conv_g.stride())

        def step(block_map, lengths, conv_pool, ssm_pool):
            out_c = self._conv_update(
                self.x_conv, conv_pool, self.weight, bias=None, activation="silu",
                block_map=block_map, seq_size_per_block=self.PAGE,
                sequence_lengths=lengths,
            )
            out_r, _ = self._recurrent(
                self.q, self.k, self.v, self.g, self.beta,
                scale=self.scale, initial_state=ssm_pool, inplace_final_state=True,
                block_map=block_map, seq_size_per_block=self.PAGE,
                sequence_lengths=lengths, use_qk_l2norm_in_kernel=True,
            )
            return out_c, out_r

        # eager 3-step reference on its own pool clone
        outs_e = []
        len_e = self.lengths.clone()
        for _ in range(3):
            len_e = len_e + 1
            oc, orr = step(self.block_map, len_e, conv_e, ssm_e)
            outs_e.append((oc.clone(), orr.clone()))

        # capture one step on the graph pool (untouched by the eager loop),
        # replay 3 times with refreshed lengths
        saved = base_g.clone()
        s = torch.npu.Stream()
        s.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(s):
            for _ in range(2):
                step(self.block_map, self.lengths, conv_g, ssm_g)
        torch.npu.current_stream().wait_stream(s)
        graph = torch.npu.NPUGraph()
        len_g = self.lengths.clone()
        with torch.npu.graph(graph, stream=s):
            out_c_g, out_r_g = step(self.block_map, len_g, conv_g, ssm_g)
        torch.npu.synchronize()
        base_g.copy_(saved)

        for t in range(3):
            len_g.copy_(len_g + 1)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            dc = (outs_e[t][0].float() - out_c_g.float()).abs().max().item()
            dr = (outs_e[t][1].float() - out_r_g.float()).abs().max().item()
            self.assertEqual(dc, 0.0, f"conv step {t}")
            self.assertEqual(dr, 0.0, f"recurrent step {t}")
        self.assertTrue(torch.equal(base_g, base_e))

    def test_torch_reference_tolerance(self):
        base, ssm, conv = _build_pool(
            self.PAGES, self.HV, self.DV, self.DK, self.STATE_LEN, self.DIM, seed=13)
        self._conv_update(
            self.x_conv, conv, self.weight, bias=None, activation="silu",
            block_map=self.block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths,
        )
        # torch GDN reference (operator FP32 semantics, see fla_npu ATK golden);
        # mirror the cross-block seeding first — recurrent mutates ssm in place
        mapping = self.block_map.detach().cpu().tolist()
        lens = self.lengths.detach().cpu().tolist()
        pages = [mapping[i][(lens[i] - 1) // self.PAGE] for i in range(self.B)]
        for i in range(self.B):
            read_page = mapping[i][max(lens[i] - 2, 0) // self.PAGE]
            if read_page != pages[i]:
                ssm[pages[i]].copy_(ssm[read_page])
        pre_state = ssm[pages].clone()
        out_r, _ = self._recurrent(
            self.q, self.k, self.v, self.g, self.beta,
            scale=self.scale, initial_state=ssm, inplace_final_state=True,
            block_map=self.block_map, seq_size_per_block=self.PAGE,
            sequence_lengths=self.lengths, use_qk_l2norm_in_kernel=True,
        )
        out_r = out_r.reshape(self.B, out_r.shape[-2], out_r.shape[-1]).float()

        qn = F.normalize(self.q.float(), dim=-1).squeeze(1)
        kn = F.normalize(self.k.float(), dim=-1).squeeze(1)
        qr = qn.repeat_interleave(self.HV // self.NK, dim=1)
        kr = kn.repeat_interleave(self.HV // self.NK, dim=1)
        vf = self.v.squeeze(1).float()
        ref = torch.zeros_like(vf)
        for i in range(self.B):
            h = pre_state[i].float() * torch.exp(self.g.reshape(self.B, -1).float()[i]).view(-1, 1, 1)
            att = (h @ kr[i].unsqueeze(-1)).squeeze(-1)
            delta = (vf[i] - att) * self.beta.reshape(self.B, -1).float()[i].view(-1, 1)
            h = h + delta.unsqueeze(-1) * kr[i].unsqueeze(-2)
            ref[i] = ((h @ (qr[i] * self.scale).unsqueeze(-1)).squeeze(-1))
        rel = (out_r - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(rel, 0.03)


if __name__ == "__main__":
    unittest.main()
