"""DiffusionGemma 4c contract test: Triton split-K windowed prefill vs fa3.

opt r8 contract 4c (r7 gate row: 29.19% windowed MFU / 39.56% full-causal
reference, target >40%). Three lanes, log tag [dgm-skab], run log
``runs/diffusiongemma-26b-a4b/build_logs/opt_r8_c4c_*.log``:

  1. numerics A/B: the Triton split-K windowed prefill kernel
     (splitk_window_attn) vs the production fa3 ragged wrapper
     (BatchPrefillWithRaggedKVCacheWrapper) on the same bf16 q/k/v at the
     real sliding-layer shape (Q=4096, GQA 16:8, head_dim 256, causal,
     window 1024, sm_scale=1.0), for split_k 1/2/4 — output agreement to
     bf16 noise (the fa3-vs-blockwise precedent: rtol/atol 3e-2) plus the
     lse convention check (median ratio ~1.0 -> same base-2 convention as
     flashinfer's cascade merge; ~1.4427 would mean natural-vs-log2).
  2. gate integration: DiffusionGemmaAttnOp with DGM_SPLITK_ATTN=1 (set
     before construction) produces the fa3-arm output (the paged-prefix
     merge path stays untouched; this lane runs WITHOUT a prefix so the
     merge is the identity here).
  3. kernel-level timing: CUDA events over ITERS runs, the 4c gate
     accounting (FLOPs = 2 * Q * avg_kv * 16 * 256, avg_kv from the exact
     causal+window mask), MFU against the adjudicated H20 148 TFLOPS, for
     fa3 vs split_k 1/2/4 — the verdict line reports whether the >40%
     windowed gate is met and the best split arm.

Verdict rules (contract):
  * mfu_splitk_best > 40%                    -> SPLITK_GATE_MET
  * mfu_splitk_best <= 40% but > fa3 arm     -> SPLITK_IMPROVED_NOT_GATE
  * numerics mismatch on any arm             -> hard failure (assert)

The same fa2-style process isolation as the wab pair is NOT needed: this
test only drives the Triton kernel and the already-warm fa3 wrapper.
"""

import os
import unittest

import torch

os.environ.setdefault("DGM_SPLITK_ATTN", "0")

from rtp_llm.models_py.model_desc.diffusion_gemma import (  # noqa: E402
    LAYER_TYPE_SLIDING,
    DiffusionGemmaAttnOp,
)
from rtp_llm.models_py.triton_kernels.common.splitk_window_attn import (  # noqa: E402
    splitk_window_attn,
)

HEAD_NUM = 16
KV_HEAD_NUM = 8
HEAD_DIM = 256
WINDOW = 1024
PAGE = 64
QO_LEN = 4096
DTYPE = torch.bfloat16
# Adjudicated H20 BF16 dense peak (run policy: 148 TFLOPS, the 4c gate row).
PEAK_TFLOPS = 148.0
VERDICT_THRESHOLD = 0.40
# the r7 gate row: 25 sliding-layer fa3 invocations, 29.19% windowed MFU.
GATE_LAYERS = 25
GATE_MFU = 0.2919
WARMUP = int(os.environ.get("DGM_SKAB_WARMUP", "25"))
ITERS = int(os.environ.get("DGM_SKAB_ITERS", "200"))


def avg_attended_kv(qo_len: int, window: int) -> float:
    """Average per-query attended KV length under causal + window mask
    (query q attends keys in [max(0, q-(window-1)), q])."""
    if window <= 0:
        return (qo_len + 1) / 2.0
    total = sum(min(q + 1, window) for q in range(qo_len))
    return total / float(qo_len)


def effective_flops(qo_len: int, window: int) -> float:
    """4c gate accounting: FLOPs = 2 * Q * avg_kv * 16 * 256."""
    return 2.0 * qo_len * avg_attended_kv(qo_len, window) * HEAD_NUM * HEAD_DIM


class DiffusionGemmaSplitkAbPrefillTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        self.device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(self.device)
        props = torch.cuda.get_device_properties(self.device)
        print(
            f"[dgm-skab] device={name} cc=sm{props.major}{props.minor} "
            f"sm_count={props.multi_processor_count}",
            flush=True,
        )
        if "H20" not in name:
            self.fail(
                f"MFU denominator (148 TFLOPS) is the adjudicated H20 peak; "
                f"this run landed on {name} and the numbers would be invalid."
            )
        torch.manual_seed(1234)
        self.q = (
            torch.randn(QO_LEN, HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device)
            * 0.25
        )
        self.k = (
            torch.randn(QO_LEN, KV_HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device)
            * 0.25
        )
        self.v = torch.randn(
            QO_LEN, KV_HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        self.cu = torch.tensor([0, QO_LEN], dtype=torch.int32, device=self.device)

    def _make_op(self, window: int, splitk_env: str = "0") -> DiffusionGemmaAttnOp:
        prev = os.environ.get("DGM_SPLITK_ATTN", "0")
        os.environ["DGM_SPLITK_ATTN"] = splitk_env
        try:
            op = DiffusionGemmaAttnOp(
                LAYER_TYPE_SLIDING,
                HEAD_NUM,
                KV_HEAD_NUM,
                HEAD_DIM,
                window,
                PAGE,
                DTYPE,
                None,
            )
        finally:
            os.environ["DGM_SPLITK_ATTN"] = prev
        return op

    def _plan_op(self, op: DiffusionGemmaAttnOp) -> None:
        from types import SimpleNamespace

        block_table = torch.arange(1, 65, dtype=torch.int32).unsqueeze(0)
        inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([0], dtype=torch.int32, device=self.device),
            input_lengths=torch.tensor([QO_LEN], dtype=torch.int32, device=self.device),
            cu_seqlens=self.cu.cpu(),
            cu_seqlens_device=self.cu,
            kv_cache_kernel_block_id=block_table,
            kv_cache_kernel_block_id_device=block_table.to(self.device),
            sequence_lengths=None,
        )
        op.plan(inputs, causal=True)

    # ------------------------------------------------------------------ #
    # numerics + timing A/B
    # ------------------------------------------------------------------ #

    def test_splitk_matches_fa3_and_mfu(self):
        op_fa3 = self._make_op(WINDOW)
        self._plan_op(op_fa3)
        ref_out, ref_lse = op_fa3.ragged_wrapper.run(
            self.q, self.k, self.v, return_lse=True
        )
        torch.cuda.synchronize()

        flops = effective_flops(QO_LEN, WINDOW)
        results = {}
        # tile sweep (straight-line loop body so Triton's pipeliner runs;
        # BM=128/warps=8 is the wgmma-compatible M split for sm90 —
        # 2 warpgroups x m64; BM=64/warps=8 and BM=128/warps=16 cannot
        # tile M=64 per warpgroup and fell to slower MMA paths in the r8
        # remote runs at 16.6-18.9%). split_k=2 on the wgmma tile as the
        # split arm.
        arms = [
            # wgmma tile (28.40% at tile-major order, r8 third remote);
            # head-major grid gives the GQA group L2-exact k/v sharing;
            # BN=128/stages=1 trades pipelining for bigger dots (shared
            # 2x(64+64)KB at stages=2 exceeds the 232448-byte limit, so
            # only stages=1 fits);
            # split_k=2 on the wgmma tile as the split arm;
            # bm64 tile-major is the small-tile reference arm
            # (BM=128/stages=3 OOMs shared memory and is not runnable).
            ("bm128w8s2", dict(block_m=128, block_n=64, num_warps=8, num_stages=2, head_major=False)),
            ("bm128w8s2hm", dict(block_m=128, block_n=64, num_warps=8, num_stages=2, head_major=True)),
            ("bm128w8s1bn128hm", dict(block_m=128, block_n=128, num_warps=8, num_stages=1, head_major=True)),
            ("bm128w8s2k2hm", dict(block_m=128, block_n=64, num_warps=8, num_stages=2, split_k=2, head_major=True)),
            ("bm64w8s3", dict(block_m=64, block_n=64, num_warps=8, num_stages=3, head_major=False)),
        ]
        for tag, cfg in arms:
            split_k = cfg.pop("split_k", 1)
            cfg.setdefault("head_major", True)
            out, lse = splitk_window_attn(
                self.q,
                self.k,
                self.v,
                self.cu,
                causal=True,
                window=WINDOW,
                sm_scale=1.0,
                split_k=split_k,
                **cfg,
            )
            torch.cuda.synchronize()
            diff = (out.float() - ref_out.float()).abs()
            absmax = float(ref_out.float().abs().max())
            ratio = float(diff.max()) / absmax
            lse_flat = lse.reshape(-1)
            ref_flat = ref_lse.reshape(-1).to(torch.float32)
            both = torch.isfinite(lse_flat) & torch.isfinite(ref_flat)
            lse_ratio = (
                float((lse_flat[both] / ref_flat[both]).median())
                if bool(both.any())
                else float("nan")
            )
            print(
                f"[dgm-skab] AB {tag} split_k={split_k}: "
                f"max_diff={float(diff.max()):.4e} "
                f"absmax={absmax:.3f} ratio={ratio:.5f} | "
                f"lse_median_ratio={lse_ratio:.4f} "
                f"lse_shape_mine={tuple(lse.shape)} "
                f"lse_shape_fa3={tuple(ref_lse.shape)}",
                flush=True,
            )
            torch.testing.assert_close(
                out,
                ref_out,
                rtol=3e-2,
                atol=3e-2,
                msg=f"split-K ({tag}, split_k={split_k}) deviates beyond bf16 noise",
            )
            # lse must share the flashinfer convention (base-2): ratio
            # 1.0, not log2(e)=1.4427 (natural).
            self.assertGreater(lse_ratio, 0.97)
            self.assertLess(lse_ratio, 1.03)

            # kernel-level timing (CUDA events over ITERS runs)
            for _ in range(WARMUP):
                splitk_window_attn(
                    self.q, self.k, self.v, self.cu, causal=True,
                    window=WINDOW, sm_scale=1.0, split_k=split_k, **cfg,
                )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(ITERS):
                splitk_window_attn(
                    self.q, self.k, self.v, self.cu, causal=True,
                    window=WINDOW, sm_scale=1.0, split_k=split_k, **cfg,
                )
            end.record()
            torch.cuda.synchronize()
            t_ms = start.elapsed_time(end) / ITERS
            mfu = flops / (t_ms * 1e-3) / 1e12 / PEAK_TFLOPS
            results[(tag, split_k)] = (t_ms, mfu)
            print(
                f"[dgm-skab] RESULT {tag} split_k={split_k} t_kernel={t_ms:.4f}ms "
                f"flops={flops:.4e} mfu={mfu * 100:.2f}% peak={PEAK_TFLOPS:.0f}TF",
                flush=True,
            )
            self.assertGreater(t_ms, 0.0)
            self.assertLess(mfu, 1.5)

        # fa3 arm timed the same way (fresh wrapper, same plan)
        run = op_fa3.ragged_wrapper.run
        for _ in range(WARMUP):
            run(self.q, self.k, self.v, return_lse=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            run(self.q, self.k, self.v, return_lse=True)
        end.record()
        torch.cuda.synchronize()
        t_fa3 = start.elapsed_time(end) / ITERS
        mfu_fa3 = flops / (t_fa3 * 1e-3) / 1e12 / PEAK_TFLOPS
        print(
            f"[dgm-skab] RESULT fa3 t_kernel={t_fa3:.4f}ms "
            f"flops={flops:.4e} mfu={mfu_fa3 * 100:.2f}% peak={PEAK_TFLOPS:.0f}TF",
            flush=True,
        )

        best_arm = min(results, key=lambda s: results[s][0])
        best_t, best_mfu = results[best_arm]
        layers_t = GATE_LAYERS * best_t
        print(
            f"[dgm-skab] XCHECK model-scale: {GATE_LAYERS} sliding layers -> "
            f"{GATE_LAYERS}*t={layers_t:.1f}ms "
            f"(r7 gate row: {GATE_LAYERS}*0.694=17.36ms, mfu={GATE_MFU * 100:.2f}%)",
            flush=True,
        )
        if best_mfu > VERDICT_THRESHOLD:
            verdict = "SPLITK_GATE_MET"
        elif best_mfu > mfu_fa3:
            verdict = "SPLITK_IMPROVED_NOT_GATE"
        else:
            verdict = "SPLITK_NO_GAIN"
        print(
            f"[dgm-skab] VERDICT best_arm={best_arm} "
            f"mfu_splitk={best_mfu * 100:.2f}% mfu_fa3={mfu_fa3 * 100:.2f}% "
            f"threshold={VERDICT_THRESHOLD * 100:.0f}% -> {verdict}",
            flush=True,
        )
        # and land in the same regime as the fa3 baseline (29.19%): the
        # numerics asserts above are the correctness gate; the timing
        # verdict line carries the honest >40% target status.
        self.assertGreater(best_mfu, GATE_MFU * 0.5)

    # ------------------------------------------------------------------ #
    # gate integration (DGM_SPLITK_ATTN=1 through the AttnOp)
    # ------------------------------------------------------------------ #

    def test_attn_op_splitk_gate_matches_fa3(self):
        os.environ["DGM_SPLITK_ATTN"] = "1"
        try:
            op_on = self._make_op(WINDOW, splitk_env="1")
        finally:
            os.environ.pop("DGM_SPLITK_ATTN", None)
        self.assertTrue(op_on.splitk_enabled, "DGM_SPLITK_ATTN gate not honored")
        op_off = self._make_op(WINDOW, splitk_env="0")
        self.assertFalse(op_off.splitk_enabled)

        self._plan_op(op_on)
        self._plan_op(op_off)
        from types import SimpleNamespace

        inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([0], dtype=torch.int32, device=self.device),
            input_lengths=torch.tensor([QO_LEN], dtype=torch.int32, device=self.device),
            cu_seqlens=self.cu.cpu(),
            cu_seqlens_device=self.cu,
            kv_cache_kernel_block_id=None,
            kv_cache_kernel_block_id_device=None,
            sequence_lengths=None,
        )
        out_on = op_on.forward(
            self.q, self.k, self.v, None, inputs, None, causal=True, write_kv=False
        )
        out_off = op_off.forward(
            self.q, self.k, self.v, None, inputs, None, causal=True, write_kv=False
        )
        diff = (out_on.float() - out_off.float()).abs()
        absmax = float(out_off.float().abs().max())
        print(
            f"[dgm-skab] GATE-AB attn_op splitk-on vs fa3: "
            f"max_diff={float(diff.max()):.4e} absmax={absmax:.3f} "
            f"ratio={float(diff.max()) / max(absmax, 1e-6):.5f}",
            flush=True,
        )
        torch.testing.assert_close(
            out_on, out_off, rtol=3e-2, atol=3e-2,
            msg="DGM_SPLITK_ATTN=1 path deviates from the fa3 path",
        )


if __name__ == "__main__":
    unittest.main()
