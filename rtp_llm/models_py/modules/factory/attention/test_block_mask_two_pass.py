"""Parity tests for the two-pass schedule builder in block_mask.py.

Pure torch + CUDA, zero rtp_llm deps — block_mask.py is loaded by file path so
the rtp_llm package __init__ chain (which pulls in compiled .so) never runs.

Run inside the dev container:
    /opt/conda310/bin/python3 -m unittest \
        rtp_llm.models_py.modules.factory.attention.test_block_mask_two_pass  # (or by path)
    /opt/conda310/bin/python3 <this file>
"""

import importlib.util
import os
import random
import unittest

import torch

_SPEC = importlib.util.spec_from_file_location(
    "block_mask_under_test",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "block_mask.py"),
)
bm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bm)

CLS_UQI = 2
SEP = 102


def ref_segments(ids, cu, cls_id=CLS_UQI, sep_id=SEP):
    """Brute-force per-sequence segment derivation (mirrors the docstring rule)."""
    segs = []
    for i in range(len(cu) - 1):
        seq = ids[cu[i] : cu[i + 1]]
        seg = [0] * len(seq)
        if cls_id in seq:
            s = seq.index(cls_id)
            e = None
            for j in range(s, len(seq)):
                if seq[j] == sep_id:
                    e = j
                    break
            if e is None:
                e = len(seq) - 1
            for j in range(s, e + 1):
                seg[j] = 1
        segs.append(seg)
    return segs


def ref_schedule(segs, cu):
    """Brute-force perm / indptrs / b_rows from per-seq segment lists."""
    perm = []
    for i, seg in enumerate(segs):
        base = cu[i]
        perm += [base + j for j, t in enumerate(seg) if t == 0]
        perm += [base + j for j, t in enumerate(seg) if t == 1]
    a_lens = [seg.count(0) for seg in segs]
    b_lens = [seg.count(1) for seg in segs]
    qo_p1 = [0]
    for a, b in zip(a_lens, b_lens):
        qo_p1.append(qo_p1[-1] + a)
        qo_p1.append(qo_p1[-1] + b)
    qo_p2 = [0]
    for b in b_lens:
        qo_p2.append(qo_p2[-1] + b)
    b_rows = []
    for i in range(len(segs)):
        b_rows += list(range(cu[i] + a_lens[i], cu[i + 1]))
    return perm, qo_p1, qo_p2, b_rows


def make_seq(rng, length, mode):
    """mode: 'none' | 'mid' | 'start' | 'nosep' | 'all_b'."""
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
    # mid: [pre] CLS_UQI [profile] SEP [post]
    pre = rng.randint(1, max(1, length - 3))
    b_len = rng.randint(0, max(0, length - pre - 2))
    post = length - pre - 2 - b_len
    return filler(pre) + [CLS_UQI] + filler(b_len) + [SEP] + filler(post)


def make_batch(rng, num_seq, modes=None):
    ids, cu = [], [0]
    for i in range(num_seq):
        length = rng.randint(1, 64)
        mode = (modes or ["none", "mid", "start", "nosep", "all_b"])[
            rng.randint(0, 4) if modes is None else i % len(modes)
        ]
        seq = make_seq(rng, length, mode)
        ids += seq
        cu.append(cu[-1] + len(seq))
    return ids, cu


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTwoPassSchedule(unittest.TestCase):
    def setUp(self):
        self.dev = torch.device("cuda")
        self.rng = random.Random(20260708)

    def _tensors(self, ids, cu):
        ids_dev = torch.tensor(ids, dtype=torch.int32, device=self.dev)
        cu_host = torch.tensor(cu, dtype=torch.int32)
        cu_dev = cu_host.to(self.dev)
        return ids_dev, cu_dev, cu_host

    def test_hostlen_matches_original(self):
        for _ in range(30):
            ids, cu = make_batch(self.rng, self.rng.randint(1, 8))
            ids_dev, cu_dev, cu_host = self._tensors(ids, cu)
            a = bm.derive_bert_uqi_segment_ids(ids_dev, cu_dev, CLS_UQI, SEP)
            b = bm.derive_bert_uqi_segment_ids_hostlen(ids_dev, cu_dev, cu_host, CLS_UQI, SEP)
            self.assertTrue(torch.equal(a, b))

    def test_schedule_matches_reference(self):
        for trial in range(30):
            ids, cu = make_batch(self.rng, self.rng.randint(1, 8))
            ids_dev, cu_dev, cu_host = self._tensors(ids, cu)
            seg = bm.derive_bert_uqi_segment_ids_hostlen(ids_dev, cu_dev, cu_host, CLS_UQI, SEP)
            segs_ref = ref_segments(ids, cu)
            flat_ref = [x for s in segs_ref for x in s]
            self.assertEqual(seg.cpu().tolist(), flat_ref, f"trial {trial}: seg mismatch")
            sched = bm.build_bert_uqi_two_pass_schedule(seg, cu_dev, cu_host)
            perm_ref, p1_ref, p2_ref, brows_ref = ref_schedule(segs_ref, cu)
            if sum(len([t for t in s if t]) for s in segs_ref) == 0:
                self.assertFalse(sched.has_b)
                self.assertIsNone(sched.perm)
                self.assertEqual(sched.qo_indptr_p1.tolist(), cu)
                continue
            self.assertTrue(sched.has_b)
            self.assertEqual(sched.perm.cpu().tolist(), perm_ref, f"trial {trial}: perm")
            self.assertEqual(sched.qo_indptr_p1.tolist(), p1_ref, f"trial {trial}: p1")
            self.assertEqual(sched.qo_indptr_p2.tolist(), p2_ref, f"trial {trial}: p2")
            self.assertEqual(sched.kv_indptr_p2.tolist(), cu, f"trial {trial}: kv p2")
            self.assertEqual(sched.b_rows.cpu().tolist(), brows_ref, f"trial {trial}: b_rows")
            # inv_perm undoes perm
            total = len(ids)
            x = torch.arange(total, device=self.dev)
            self.assertTrue(torch.equal(x[sched.perm][sched.inv_perm], x))
            # indptr dtypes/devices: CPU int32 for flashinfer plan fast path
            for t in (sched.qo_indptr_p1, sched.qo_indptr_p2, sched.kv_indptr_p2):
                self.assertEqual(t.dtype, torch.int32)
                self.assertEqual(t.device.type, "cpu")

    def test_all_a_batch(self):
        ids, cu = make_batch(self.rng, 4, modes=["none"])
        ids_dev, cu_dev, cu_host = self._tensors(ids, cu)
        seg = bm.derive_bert_uqi_segment_ids_hostlen(ids_dev, cu_dev, cu_host, CLS_UQI, SEP)
        self.assertEqual(int(seg.sum()), 0)
        sched = bm.build_bert_uqi_two_pass_schedule(seg, cu_dev, cu_host)
        self.assertFalse(sched.has_b)
        self.assertEqual(sched.qo_indptr_p1.tolist(), cu)

    def test_edge_modes(self):
        for modes in (["start"], ["nosep"], ["all_b"], ["mid", "none"], ["none", "all_b", "mid"]):
            ids, cu = make_batch(self.rng, len(modes) * 2, modes=modes)
            ids_dev, cu_dev, cu_host = self._tensors(ids, cu)
            seg = bm.derive_bert_uqi_segment_ids_hostlen(ids_dev, cu_dev, cu_host, CLS_UQI, SEP)
            segs_ref = ref_segments(ids, cu)
            self.assertEqual(seg.cpu().tolist(), [x for s in segs_ref for x in s], f"{modes}")
            sched = bm.build_bert_uqi_two_pass_schedule(seg, cu_dev, cu_host)
            if sched.has_b:
                perm_ref, p1_ref, p2_ref, brows_ref = ref_schedule(segs_ref, cu)
                self.assertEqual(sched.perm.cpu().tolist(), perm_ref, f"{modes}")
                self.assertEqual(sched.qo_indptr_p1.tolist(), p1_ref, f"{modes}")
                self.assertEqual(sched.b_rows.cpu().tolist(), brows_ref, f"{modes}")

    def test_single_seq_real_shape(self):
        # 线上真实形状: item(191) + query(4) + [CLS_UQI, profile, SEP] + vision(4)
        ids = [self.rng.randint(200, 30000) for _ in range(195)] + [2, 161, 102] + [-7] * 4
        cu = [0, len(ids)]
        ids_dev, cu_dev, cu_host = self._tensors(ids, cu)
        seg = bm.derive_bert_uqi_segment_ids_hostlen(ids_dev, cu_dev, cu_host, CLS_UQI, SEP)
        self.assertEqual(seg.cpu().tolist(), [0] * 195 + [1, 1, 1] + [0] * 4)
        sched = bm.build_bert_uqi_two_pass_schedule(seg, cu_dev, cu_host)
        self.assertTrue(sched.has_b)
        self.assertEqual(sched.qo_indptr_p1.tolist(), [0, 199, 202])
        self.assertEqual(sched.b_rows.cpu().tolist(), [199, 200, 201])
        # vision 尾巴回 A 段: permuted 布局里 vision 行在 B 之前
        self.assertEqual(sched.perm.cpu().tolist()[195:199], [198, 199, 200, 201])


if __name__ == "__main__":
    unittest.main(verbosity=2)
