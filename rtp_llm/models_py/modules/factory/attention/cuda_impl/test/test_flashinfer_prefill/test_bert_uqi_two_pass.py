"""Numeric parity tests for the two-pass user-profile attention core.

Compares run_two_pass against BOTH:
  (a) an eager masked-attention oracle (fp32 math), and
  (b) the old FlashInfer custom_mask path (same GPU library, dense mask),
on random ragged batches covering mixed B/no-B, B-at-start, B-to-end and
all-A cases. Loads block_mask.py / bert_uqi_two_pass_core.py by file path —
torch + flashinfer only, no rtp_llm compiled deps.

Run inside the dev container:
    CUDA_VISIBLE_DEVICES=3 /opt/conda310/bin/python3 <this file>
"""

import importlib.util
import os
import random
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATTN_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ATTN_DIR, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load("block_mask_under_test", "block_mask.py")
core = _load("two_pass_core_under_test", os.path.join("cuda_impl", "bert_uqi_two_pass_core.py"))

CLS_UQI = 2
SEP = 102


def eager_masked(q, k, v, seg, cu, scale):
    """fp32 oracle: visible[i,j] = seg[i]==1 or seg[j]==0, per-sequence softmax."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        qi, ki, vi = (t[s:e].float() for t in (q, k, v))
        si = seg[s:e]
        m = (si[:, None] == 1) | (si[None, :] == 0)  # [q, k] True=visible
        lg = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        lg = lg.masked_fill(~m[None], float("-inf"))
        out[s:e] = torch.einsum("hqk,khd->qhd", lg.softmax(-1), vi)
    return out


def make_seq(rng, length, mode):
    filler = lambda n: [rng.randint(200, 30000) for _ in range(n)]
    if mode == "none" or length < 3:
        return filler(length)
    if mode == "all_b":
        return [CLS_UQI] + filler(max(0, length - 2)) + [SEP]
    if mode == "start":
        b_len = rng.randint(0, max(0, length - 2))
        return [CLS_UQI] + filler(b_len) + [SEP] + filler(length - b_len - 2)
    if mode == "nosep":
        pos = rng.randint(0, length - 1)
        return filler(pos) + [CLS_UQI] + filler(length - pos - 1)
    pre = rng.randint(1, max(1, length - 3))
    b_len = rng.randint(0, max(0, length - pre - 2))
    return filler(pre) + [CLS_UQI] + filler(b_len) + [SEP] + filler(length - pre - 2 - b_len)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTwoPassParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        cls.dev = torch.device("cuda")
        ws = lambda: torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=cls.dev)
        cls.w_p1 = BatchPrefillWithRaggedKVCacheWrapper(ws(), backend="auto")
        cls.w_p2 = BatchPrefillWithRaggedKVCacheWrapper(ws(), backend="auto")
        cls.w_mask = BatchPrefillWithRaggedKVCacheWrapper(ws(), backend="auto")

    def _case(self, rng, num_seq, modes, dtype, heads=4, kv_heads=4, dim=64, eager_p2=False):
        ids, cu = [], [0]
        for i in range(num_seq):
            seq = make_seq(rng, rng.randint(3, 96), modes[i % len(modes)])
            ids += seq
            cu.append(cu[-1] + len(seq))
        total = len(ids)
        ids_dev = torch.tensor(ids, dtype=torch.int32, device=self.dev)
        cu_host = torch.tensor(cu, dtype=torch.int32)
        cu_dev = cu_host.to(self.dev)
        seg = bm.derive_bert_uqi_segment_ids_hostlen(ids_dev, cu_dev, cu_host, CLS_UQI, SEP)
        sched = bm.build_bert_uqi_two_pass_schedule(seg, cu_dev, cu_host)
        torch.manual_seed(hash((num_seq, total)) % (2**31))
        q = torch.randn(total, heads, dim, dtype=dtype, device=self.dev) / 3
        k = torch.randn(total, kv_heads, dim, dtype=dtype, device=self.dev) / 3
        v = torch.randn(total, kv_heads, dim, dtype=dtype, device=self.dev)
        scale = dim ** -0.5

        # two-pass on permuted layout, then unpermute
        if sched.has_b:
            qp, kp, vp = q[sched.perm], k[sched.perm], v[sched.perm]
        else:
            qp, kp, vp = q, k, v
        w2 = None if eager_p2 else self.w_p2
        core.plan_two_pass(self.w_p1, w2, sched, heads, kv_heads, dim, dtype)
        out_p = core.run_two_pass(self.w_p1, w2, sched, qp, kp, vp)
        out_two = out_p[sched.inv_perm] if sched.has_b else out_p

        ref = eager_masked(q, k, v, seg, cu, scale)
        return q, k, v, seg, cu_host, out_two, ref

    def _assert_close(self, out, ref, dtype, label):
        atol = 2e-2 if dtype == torch.bfloat16 else 2e-3
        diff = (out.float() - ref).abs().max().item()
        self.assertLess(diff, atol, f"{label}: maxdiff {diff:.5f} vs atol {atol}")

    def test_vs_eager_oracle(self):
        rng = random.Random(7)
        for eager_p2 in (False, True):
            for dtype in (torch.float16, torch.bfloat16):
                for modes in (["mid"], ["mid", "none"], ["start", "nosep", "mid"],
                              ["all_b", "none"], ["none"]):
                    for num_seq in (1, 4, 8):
                        q, k, v, seg, cu_host, out, ref = self._case(
                            rng, num_seq, modes, dtype, eager_p2=eager_p2)
                        self._assert_close(
                            out, ref, dtype, f"p2eager={eager_p2}/{dtype}/{modes}/{num_seq}")

    def test_vs_custom_mask_path(self):
        """Two GPU impls (two-pass vs dense custom_mask) must agree tightly."""
        rng = random.Random(13)
        for dtype in (torch.float16, torch.bfloat16):
            for modes in (["mid", "none"], ["start", "mid", "nosep"]):
                q, k, v, seg, cu_host, out_two, _ = self._case(rng, 6, modes, dtype)
                cu = cu_host.tolist()
                cu_dev = cu_host.to(self.dev)
                mask = bm.build_bert_uqi_flashinfer_mask(seg, cu_dev)
                # NOTE: the custom_mask path REQUIRES device indptrs (segment_packbits
                # asserts kDLCUDA) — i.e. the old path structurally cannot take the
                # sync-free host-indptr shortcut the two-pass path uses.
                self.w_mask.plan(
                    cu_dev, cu_dev, 4, 4, 64, head_dim_vo=64,
                    custom_mask=mask, causal=False, q_data_type=dtype,
                )
                out_mask = self.w_mask.run(q, k, v)
                diff = (out_two.float() - out_mask.float()).abs().max().item()
                # different kernels (FA3 no-mask vs FA2 masked) => a few ulps apart;
                # eps(bf16)=2^-7=0.0078 at |x|~1, so allow 2e-2 for bf16.
                atol = 2e-2 if dtype == torch.bfloat16 else 5e-3
                self.assertLess(diff, atol, f"{dtype}/{modes}: two-pass vs mask {diff:.5f}")

    def test_real_shape_30_docs(self):
        """线上形状: 30 docs × ~202 tok (item191+query4+[2,161,102]+vision4)."""
        rng = random.Random(30)
        ids, cu = [], [0]
        for _ in range(30):
            seq = ([rng.randint(200, 30000) for _ in range(195)] + [2, 161, 102]
                   + [rng.randint(200, 30000)] * 4)
            ids += seq
            cu.append(cu[-1] + len(seq))
        ids_dev = torch.tensor(ids, dtype=torch.int32, device=self.dev)
        cu_host = torch.tensor(cu, dtype=torch.int32)
        cu_dev = cu_host.to(self.dev)
        seg = bm.derive_bert_uqi_segment_ids_hostlen(ids_dev, cu_dev, cu_host, CLS_UQI, SEP)
        sched = bm.build_bert_uqi_two_pass_schedule(seg, cu_dev, cu_host)
        total = len(ids)
        dtype = torch.bfloat16
        q = torch.randn(total, 4, 64, dtype=dtype, device=self.dev) / 3
        k = torch.randn(total, 4, 64, dtype=dtype, device=self.dev) / 3
        v = torch.randn(total, 4, 64, dtype=dtype, device=self.dev)
        core.plan_two_pass(self.w_p1, self.w_p2, sched, 4, 4, 64, dtype)
        out = core.run_two_pass(self.w_p1, self.w_p2, sched, q[sched.perm], k[sched.perm], v[sched.perm])[sched.inv_perm]
        ref = eager_masked(q, k, v, seg, cu, 64 ** -0.5)
        diff = (out.float() - ref).abs().max().item()
        self.assertLess(diff, 2e-2, f"30-doc maxdiff {diff:.5f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
