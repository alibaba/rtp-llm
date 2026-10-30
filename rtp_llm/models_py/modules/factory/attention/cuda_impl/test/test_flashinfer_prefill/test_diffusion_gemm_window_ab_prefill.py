"""DiffusionGemma 4c-attribution A/B: window semantics vs kernel efficiency.

r75 stop-over contract (opt r5, 4c lane; log tag [dgm-wab], run log
``runs/diffusiongemma-26b-a4b/build_logs/opt_r5_c*.log``). Diagnostic
only - this file adds a measurement, it does not touch model behavior.

Background: the 4c gate row measured the dgm 4096 full-window prefill's
main attention kernel (flashinfer fa3 ragged prefill, 25 sliding-layer
invocations) at 29.4 ms / 7.8e11 effective FLOPs / 17.93% MFU vs the
adjudicated H20 BF16 peak of 148 TFLOPS. The open question: is the low
MFU structural (Q=4096 x window 1024 cuts the mainloop to ~0.44x of full
causal while the kernel's fixed costs do not shrink) or a kernel-path
efficiency problem?

Method: drive the production ``DiffusionGemmaAttnOp`` ragged path
(``BatchPrefillWithRaggedKVCacheWrapper``, backend auto -> fa3 on sm90)
at the real per-invocation shape of the 25 sliding layers - a single
ragged request, Q=4096, GQA 16:8, head_dim 256, causal, bf16,
sm_scale=1.0 - under two window regimes:

  arm W: window_left=-1023  (production sliding window, HF mask
        [q-1023, q]; per-query attended kv = min(q+1, 1024), so
        avg_kv = 896.0 over the 4096 queries)
  arm F: window_left=-1     (full causal; avg_kv = 2048.5)

Effective-FLOPs accounting follows the 4c gate row exactly:
``FLOPs = 2 * Q * avg_kv * 16 * 256`` per invocation (avg_kv computed
from the causal/window semantics above, exact integer arithmetic), MFU
against 148 TFLOPS.

Verdict rule (contract):
  F > 40% and W < 40% -> STRUCTURAL_WINDOW: 4c 17.93% is window-semantics
    structural (Q=4096 x window 1024: mainloop work ~1/4 of the
    un-windowed square while per-CTA fixed cost does not shrink).
    Closing line: window-semantics attribution; hd512 Triton/split-K is
    the next-stage structural improvement entry.
  F <= 40%           -> KERNEL_PATH_EFFICIENCY: the fa2 / blockwise
    comparison rows in the log are the evidence table.
  arms byte-identical -> WINDOW_IGNORED: the kernel executed full-causal
    work under window_left=-1023 and the 4c FLOPs accounting is invalid.

The same file drives the fa2 backend through DGM_WAB_BACKEND=fa2 in a
separate py_test process (crash isolation, the same convention as the
test_diffusion_gemma_flashinfer_prefill_fa2 pair). test ordering note:
``test_ab_bench_mfu_and_verdict`` runs before
``test_semantics_window_bites_at_q4096`` (alphabetical), so the timing
rows reach the log even if the semantics assertions then fail.
"""

import os
import time
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.diffusion_gemma import (
    LAYER_TYPE_SLIDING,
    DiffusionGemmaAttnOp,
)

HEAD_NUM = 16
KV_HEAD_NUM = 8
HEAD_DIM = 256
WINDOW = 1024
PAGE = 64
QO_LEN = 4096
BLOCK_TABLE_WIDTH = 64
DTYPE = torch.bfloat16
# Adjudicated H20 BF16 dense peak (run policy: 148 TFLOPS, performance
# standard ruling used by the 4c gate row).
PEAK_TFLOPS = 148.0
# The 4c gate row aggregated the 25 sliding-layer invocations of one
# 4096 prefill (29.4 ms / 7.8e11 FLOPs); used for the cross-check line.
GATE_LAYERS = 25
GATE_T_MS = 29.40
GATE_FLOPS = 7.8e11
GATE_MFU = 0.1793
VERDICT_THRESHOLD = 0.40

TEST_BACKEND = os.environ.get("DGM_WAB_BACKEND", "auto")
WARMUP = int(os.environ.get("DGM_WAB_WARMUP", "25"))
ITERS = int(os.environ.get("DGM_WAB_ITERS", "200"))
BLOCK_Q = 512
REF_ITERS = int(os.environ.get("DGM_WAB_REF_ITERS", "10"))


def avg_attended_kv(qo_len: int, window: int) -> float:
    """Average per-query attended KV length under causal + window mask.

    window semantics (flashinfer window_left, inclusive lower bound):
    query q attends keys in [max(0, q - (window - 1)), q] for a sliding
    window, and [0, q] for the full-causal window_left=-1 regime.
    """
    if window <= 0:
        return (qo_len + 1) / 2.0
    total = sum(min(q + 1, window) for q in range(qo_len))
    return total / float(qo_len)


def effective_flops(qo_len: int, window: int) -> float:
    """4c gate accounting: FLOPs = 2 * Q * avg_kv * 16 * 256."""
    return 2.0 * qo_len * avg_attended_kv(qo_len, window) * HEAD_NUM * HEAD_DIM


class DiffusionGemmWindowAbPrefillTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        self.device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(self.device)
        props = torch.cuda.get_device_properties(self.device)
        print(
            f"[dgm-wab] device={name} cc=sm{props.major}{props.minor} "
            f"sm_count={props.multi_processor_count} backend={TEST_BACKEND}",
            flush=True,
        )
        if "H20" not in name:
            self.fail(
                f"MFU denominator (148 TFLOPS) is the adjudicated H20 peak; "
                f"this run landed on {name} and the numbers would be invalid."
            )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _make_op(self, window: int) -> DiffusionGemmaAttnOp:
        op = DiffusionGemmaAttnOp(
            LAYER_TYPE_SLIDING,
            HEAD_NUM,
            KV_HEAD_NUM,
            HEAD_DIM,
            window,
            PAGE,
            DTYPE,
            None,
            backend=TEST_BACKEND,
        )
        self.assertTrue(op.use_flashinfer, "flashinfer ragged path unavailable")
        # 2026-10-20 K123 fix: flashinfer 0.6.9 only compiles the sliding
        # window for window_left >= 0 — the AttnOp now passes the positive
        # window-1 (attend kv >= q-(window-1), the exact HF mask).
        expected_wl = (window - 1) if window > 0 else -1
        self.assertEqual(op._window_left(), expected_wl)
        return op

    def _make_inputs(self, qo_len: int):
        cu = torch.tensor([0, qo_len], dtype=torch.int32, device=self.device)
        block_table = torch.arange(
            1, 1 + BLOCK_TABLE_WIDTH, dtype=torch.int32
        ).unsqueeze(0)
        return SimpleNamespace(
            prefix_lengths=torch.tensor(
                [0], dtype=torch.int32, device=self.device
            ),
            input_lengths=torch.tensor(
                [qo_len], dtype=torch.int32, device=self.device
            ),
            cu_seqlens=cu.cpu(),
            cu_seqlens_device=cu,
            kv_cache_kernel_block_id=block_table,
            kv_cache_kernel_block_id_device=block_table.to(self.device),
            sequence_lengths=None,
        )

    def _make_qkv(self, qo_len: int):
        # Same input regime as the dgm flashinfer regression: scale q/k so
        # pre-softmax scores stay smooth under sm_scale=1.0.
        torch.manual_seed(1234)
        q = torch.randn(qo_len, HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device) * 0.25
        k = torch.randn(qo_len, KV_HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device) * 0.25
        v = torch.randn(qo_len, KV_HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device)
        return q, k, v

    def _bench_arm(self, tag: str, op: DiffusionGemmaAttnOp, window: int):
        """Plan once, then CUDA-event time ITERS consecutive ragged runs.

        The event window covers only the ragged_run kernel stream (plan is
        host-side metadata and is timed separately as a wall-clock
        diagnostic), matching the kernel-time semantics of the 4c timeline
        row.
        """
        q, k, v = self._make_qkv(QO_LEN)
        inputs = self._make_inputs(QO_LEN)
        op.plan(inputs, causal=True)
        resolved = getattr(op.ragged_wrapper, "_backend", "?")
        wl = op._window_left()
        print(
            f"[dgm-wab] {tag} planned: backend={resolved} window_left={wl} "
            f"(wrapper use_swa={wl >= 0}) qo_len={QO_LEN} "
            f"heads={HEAD_NUM}/{KV_HEAD_NUM} head_dim={HEAD_DIM} causal=True",
            flush=True,
        )
        run = op.ragged_wrapper.run
        for _ in range(WARMUP):
            out, lse = run(q, k, v, return_lse=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            out, lse = run(q, k, v, return_lse=True)
        end.record()
        torch.cuda.synchronize()
        t_ms = start.elapsed_time(end) / ITERS
        # wall-clock diagnostic: the production per-step cost is plan+run
        wall_n = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(wall_n):
            op.plan(inputs, causal=True)
            run(q, k, v, return_lse=True)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) / wall_n
        flops = effective_flops(QO_LEN, window)
        achieved_tf = flops / (t_ms * 1e-3) / 1e12
        mfu = achieved_tf / PEAK_TFLOPS
        print(
            f"[dgm-wab] RESULT backend={resolved} arm={tag} "
            f"window={window if window > 0 else 'full'} wl={wl} "
            f"avg_kv={avg_attended_kv(QO_LEN, window):.1f} t_kernel={t_ms:.4f}ms "
            f"flops={flops:.4e} achieved={achieved_tf:.2f}TF "
            f"mfu={mfu * 100:.2f}% peak={PEAK_TFLOPS:.0f}TF "
            f"(plan+run wall={wall_ms:.4f}ms)",
            flush=True,
        )
        self.assertGreater(t_ms, 0.0)
        self.assertGreater(mfu, 0.0)
        self.assertLess(mfu, 1.5)
        return {
            "tag": tag,
            "window": window,
            "backend": resolved,
            "t_ms": t_ms,
            "wall_ms": wall_ms,
            "flops": flops,
            "mfu": mfu,
            "out": out,
            "q": q,
            "k": k,
            "v": v,
        }

    def _profile_kernel_name(self, res) -> None:
        """One torch-profiler pass over the timed arm to name the kernel."""
        try:
            from torch.profiler import ProfilerActivity, profile

            run = None
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    out, _ = res["op"].ragged_wrapper.run(
                        res["q"], res["k"], res["v"], return_lse=True
                    )
                torch.cuda.synchronize()
            rows = [
                evt
                for evt in prof.key_averages()
                if getattr(evt, "device_time_total", 0) > 0
            ]
            rows = sorted(rows, key=lambda e: -getattr(e, "device_time_total", 0))
            for evt in rows[:3]:
                name = evt.key[:300]
                print(
                    f"[dgm-wab] KERNEL backend={res['backend']} arm={res['tag']} "
                    f"cuda_time={getattr(evt, 'device_time_total', 0) / 3:.4f}us "
                    f"calls={evt.count} name={name}",
                    flush=True,
                )
        except Exception as error:  # profiler is evidence, not a gate
            print(f"[dgm-wab] KERNEL profiling skipped: {error}", flush=True)

    # ------------------------------------------------------------------ #
    # blockwise (chunked) reference: numerics + the 分块 timing row
    # ------------------------------------------------------------------ #

    def _prep_blockwise(self, q, k, v):
        group = HEAD_NUM // KV_HEAD_NUM
        ke = k.repeat_interleave(group, dim=1).transpose(0, 1).contiguous()
        ve = v.repeat_interleave(group, dim=1).transpose(0, 1).contiguous()
        qh = q.transpose(0, 1).contiguous()
        return qh, ke, ve

    def _blockwise_mask(self, window: int):
        qi = torch.arange(QO_LEN, device=self.device).unsqueeze(1)
        kj = torch.arange(QO_LEN, device=self.device).unsqueeze(0)
        keep = kj <= qi
        if window > 0:
            keep = keep & (kj >= qi - (window - 1))
        return ~keep  # True = masked out

    def _blockwise_attention(self, qh, ke, ve, mask, score_dtype):
        """Chunked Q-block attention (the 分块 reference). bf16 GEMMs with
        fp32 softmax; score_dtype=torch.float32 gives the numerics
        reference, torch.bfloat16 the fair timing variant."""
        out = torch.empty_like(qh)
        ket = ke.transpose(1, 2).to(score_dtype).contiguous()
        qhs = qh.to(score_dtype)
        for s in range(0, QO_LEN, BLOCK_Q):
            e = min(s + BLOCK_Q, QO_LEN)
            scores = torch.bmm(qhs[:, s:e], ket).float()
            scores = scores.masked_fill(mask[s:e].unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(ve.dtype)
            out[:, s:e] = torch.bmm(probs, ve)
        return out

    def _bench_blockwise(self, q, k, v, window: int):
        qh, ke, ve = self._prep_blockwise(q, k, v)
        mask = self._blockwise_mask(window)
        for _ in range(2):
            self._blockwise_attention(qh, ke, ve, mask, torch.bfloat16)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REF_ITERS):
            self._blockwise_attention(qh, ke, ve, mask, torch.bfloat16)
        end.record()
        torch.cuda.synchronize()
        t_ms = start.elapsed_time(end) / REF_ITERS
        flops = effective_flops(QO_LEN, window)
        mfu = flops / (t_ms * 1e-3) / 1e12 / PEAK_TFLOPS
        print(
            f"[dgm-wab] REF blockwise-bf16 arm={'window' if window > 0 else 'full'} "
            f"window={window if window > 0 else 'full'} avg_kv={avg_attended_kv(QO_LEN, window):.1f} "
            f"t={t_ms:.4f}ms flops={flops:.4e} mfu={mfu * 100:.2f}% peak={PEAK_TFLOPS:.0f}TF",
            flush=True,
        )
        return t_ms

    # ------------------------------------------------------------------ #
    # tests
    # ------------------------------------------------------------------ #

    def test_ab_bench_mfu_and_verdict(self):
        """The contract A/B: same fa3 kernel, same shape, two windows.

        Two passes in W,F,F,W order: the two arms do byte-identical work
        whenever the kernel ignores window_left, so the W/F time delta of
        identical-work passes doubles as the measurement-noise bound for
        the 40%-line verdict; pass means average out clock-ramp ordering
        effects.
        """
        op_w = self._make_op(WINDOW)
        op_f = self._make_op(0)
        order = (
            ("W", op_w, WINDOW),
            ("F", op_f, 0),
            ("F", op_f, 0),
            ("W", op_w, WINDOW),
        )
        passes = {"W": [], "F": []}
        for idx, (tag, op, window) in enumerate(order):
            res = self._bench_arm(f"{tag}p{idx}", op, window)
            res["tag"] = tag
            passes[tag].append(res)

        def _mean(tag, field):
            return sum(p[field] for p in passes[tag]) / len(passes[tag])

        res_w = {k: _mean("W", k) for k in ("t_ms", "flops", "mfu")}
        res_f = {k: _mean("F", k) for k in ("t_ms", "flops", "mfu")}
        res_w["backend"] = passes["W"][0]["backend"]
        res_f["backend"] = passes["F"][0]["backend"]
        noise = abs(res_w["t_ms"] - res_f["t_ms"]) / (
            (res_w["t_ms"] + res_f["t_ms"]) / 2
        )
        print(
            f"[dgm-wab] MEAN arm=W t={res_w['t_ms']:.4f}ms mfu={res_w['mfu'] * 100:.2f}% | "
            f"arm=F t={res_f['t_ms']:.4f}ms mfu={res_f['mfu'] * 100:.2f}% | "
            f"identical-work W/F delta={noise * 100:.2f}% (=noise bound)",
            flush=True,
        )
        res_w["q"], res_w["k"], res_w["v"] = (
            passes["W"][-1]["q"],
            passes["W"][-1]["k"],
            passes["W"][-1]["v"],
        )
        prof_w = dict(passes["W"][0])
        prof_w["op"] = op_w
        prof_f = dict(passes["F"][0])
        prof_f["op"] = op_f
        self._profile_kernel_name(prof_w)
        self._profile_kernel_name(prof_f)
        self._bench_blockwise(res_w["q"], res_w["k"], res_w["v"], WINDOW)
        self._bench_blockwise(res_w["q"], res_w["k"], res_w["v"], 0)

        d_arms = (
            (passes["W"][-1]["out"].float() - passes["F"][-1]["out"].float())
            .abs()
            .max()
            .item()
        )
        print(
            f"[dgm-wab] ARMS-DIFF max|out(W)-out(F)|={d_arms:.6f} "
            f"(0 -> window_left was ignored by the kernel)",
            flush=True,
        )

        # model-scale cross-check against the 4c gate row
        layers_t = GATE_LAYERS * res_w["t_ms"]
        layers_flops = GATE_LAYERS * res_w["flops"]
        implied_mfu = layers_flops / (layers_t * 1e-3) / 1e12 / PEAK_TFLOPS
        print(
            f"[dgm-wab] XCHECK model-scale: {GATE_LAYERS} sliding layers -> "
            f"{GATE_LAYERS}*t={layers_t:.1f}ms (gate row {GATE_T_MS}ms), "
            f"{GATE_LAYERS}*flops={layers_flops:.3e} (gate row {GATE_FLOPS:.1e}), "
            f"implied MFU={implied_mfu * 100:.2f}% (gate row {GATE_MFU * 100:.2f}%)",
            flush=True,
        )

        mfu_w, mfu_f = res_w["mfu"], res_f["mfu"]
        if d_arms < 1e-4:
            verdict = "WINDOW_IGNORED"
        elif mfu_f > VERDICT_THRESHOLD and mfu_w <= VERDICT_THRESHOLD:
            verdict = "STRUCTURAL_WINDOW"
        elif mfu_f <= VERDICT_THRESHOLD:
            verdict = "KERNEL_PATH_EFFICIENCY"
        else:
            verdict = "WINDOW_NOT_LIMITING"
        print(
            f"[dgm-wab] VERDICT backend={res_w['backend']} "
            f"mfu_window={mfu_w * 100:.2f}% mfu_full={mfu_f * 100:.2f}% "
            f"threshold={VERDICT_THRESHOLD * 100:.0f}% -> {verdict}",
            flush=True,
        )
        if verdict == "STRUCTURAL_WINDOW":
            print(
                "[dgm-wab] CLOSING 4c 17.93% = 窗语义结构性归因: Q=4096 x 窗1024 "
                "mainloop 工作量 ~0.44x 而 kernel 固定成本不缩; 收口径径 = "
                "「窗语义归因, hd512 Triton/split-K 为下阶段结构提升入口」",
                flush=True,
            )
        elif verdict == "KERNEL_PATH_EFFICIENCY":
            print(
                "[dgm-wab] CLOSING 4c = kernel 路径效率问题(非窗语义): fa2/fa3/分块 "
                "对照行见上; 下阶段按 fa2->fa3->分块 差距排序入口",
                flush=True,
            )
        if verdict == "WINDOW_IGNORED":
            corrected_flops = GATE_LAYERS * res_f["flops"]
            corrected_tf = corrected_flops / (layers_t * 1e-3) / 1e12
            print(
                f"[dgm-wab] CORRECTED 4c row (kernel ran un-windowed work): "
                f"{GATE_LAYERS}*flops_full={corrected_flops:.3e} / {GATE_LAYERS}*t="
                f"{layers_t:.1f}ms -> {corrected_tf:.2f}TF = "
                f"{corrected_tf / PEAK_TFLOPS * 100:.2f}% MFU; the gate's "
                f"{GATE_FLOPS:.1e} windowed flops over un-windowed kernel time "
                f"manufactured the {GATE_MFU * 100:.2f}% row",
                flush=True,
            )
        print(
            f"[dgm-wab] REPORT 4c-attribution backend={res_w['backend']} "
            f"Q={QO_LEN} GQA={HEAD_NUM}:{KV_HEAD_NUM} d={HEAD_DIM} causal "
            f"window_mfu={mfu_w * 100:.2f}% (avg_kv={avg_attended_kv(QO_LEN, WINDOW):.1f}, "
            f"t={res_w['t_ms']:.4f}ms) full_mfu={mfu_f * 100:.2f}% "
            f"(avg_kv={avg_attended_kv(QO_LEN, 0):.1f}, t={res_f['t_ms']:.4f}ms) "
            f"peak={PEAK_TFLOPS:.0f}TF verdict={verdict}",
            flush=True,
        )

    def test_semantics_window_bites_at_q4096(self):
        """Both arms must match a blockwise reference with their own mask.

        This pins the two accounting premises at a length where the window
        actually bites (the 1024-window first drops keys at query 1024,
        invisible to the sub-1025 regression sweep): the kernel honors
        window_left=-1023 (W matches the windowed reference and differs
        from F), and F matches the plain causal reference. If the kernel
        ignored the window, the 4c effective-FLOPs accounting would be
        invalid and the bench verdict row already flags WINDOW_IGNORED.

        Self-contained on purpose: unittest builds a fresh instance per
        test method, so this re-drives one ragged run per arm (same seed
        -> same tensors as the bench, JIT already warm in-process).

        Each arm's output is compared against BOTH references before any
        assertion, so the log carries the decisive which-mask evidence
        even when the assertion then fails: out(W) matching the causal
        reference (not the windowed one) proves the kernel executed the
        full-causal mask under window_left=-1023.
        """
        q, k, v = self._make_qkv(QO_LEN)
        outs = {}
        for tag, window in (("W", WINDOW), ("F", 0)):
            op = self._make_op(window)
            op.plan(self._make_inputs(QO_LEN), causal=True)
            out, _ = op.ragged_wrapper.run(q, k, v, return_lse=True)
            torch.cuda.synchronize()
            outs[tag] = (out, window)
        d_arms = (outs["W"][0].float() - outs["F"][0].float()).abs().max().item()
        print(
            f"[dgm-wab] SEMANTICS-ARMS-DIFF backend={TEST_BACKEND} "
            f"max|out(W)-out(F)|={d_arms:.6f}",
            flush=True,
        )
        qh, ke, ve = self._prep_blockwise(q, k, v)
        refs = {}
        for window in (WINDOW, 0):
            mask = self._blockwise_mask(window)
            ref = self._blockwise_attention(qh, ke, ve, mask, torch.float32)
            refs[window] = ref.transpose(0, 1).to(DTYPE)
        failures = []
        for tag, (out, window) in outs.items():
            d_win = (out.float() - refs[WINDOW].float()).abs().max().item()
            d_cau = (out.float() - refs[0].float()).abs().max().item()
            print(
                f"[dgm-wab] SEMANTICS backend={TEST_BACKEND} arm={tag} "
                f"requested={'window(wl=%d)' % -(WINDOW - 1) if window > 0 else 'full_causal'} "
                f"vs windowed-ref max_diff={d_win:.4e} "
                f"vs causal-ref max_diff={d_cau:.4e} "
                f"-> matches {'windowed-ref' if d_win < d_cau else 'causal-ref'}",
                flush=True,
            )
            try:
                torch.testing.assert_close(
                    out,
                    refs[window],
                    rtol=3e-2,
                    atol=3e-2,
                    msg=(
                        f"arm {tag} (window_left="
                        f"{-(window - 1) if window > 0 else -1}) did not produce "
                        f"its requested mask semantics at Q={QO_LEN}: "
                        f"window_left=-1023 is ignored by this flashinfer build "
                        f"(use_swa compiles only for window_left>=0), so the "
                        f"kernel ran full-causal work and the 4c windowed-FLOPs "
                        f"accounting is invalid"
                    ),
                )
            except AssertionError as error:
                failures.append(str(error))
        if failures:
            self.fail("\n".join(failures))
        self.assertGreater(
            d_arms,
            1e-3,
            f"window_left={-(WINDOW - 1)} appears IGNORED by backend "
            f"{TEST_BACKEND} at Q={QO_LEN}: out(W) == out(F) "
            f"(max diff {d_arms:.2e}); the 4c windowed FLOPs accounting "
            f"would be invalid",
        )


if __name__ == "__main__":
    unittest.main()
