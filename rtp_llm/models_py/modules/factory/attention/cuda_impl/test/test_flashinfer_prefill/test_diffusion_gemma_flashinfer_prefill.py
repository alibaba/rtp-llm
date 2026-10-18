"""DiffusionGemma sliding-layer FlashInfer prefill regression.

Reproduces the exact FlashInfer call sequence of
DiffusionGemmaAttnOp (LAYER_TYPE_SLIDING: GQA 16:8, head_dim 256, page 64,
window 1024, sm_scale 1.0, bf16) with the mock paged KV cache the smoke uses,
across the prefill lengths seen in the dgm smokes:
24 (chat text prefill, passing), 256 (canvas commit, passing) and
278/539 (vision prefill lengths, CUBLAS_STATUS_EXECUTION_FAILED downstream).

Each stage (ragged run / paged KV write / paged prefix run / merge) is
followed by a torch.cuda.synchronize() with a labeled log line so a CI log
pinpoints which kernel poisons the context. The output is compared against
the op's own torch reference for every length.
"""

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
POOL_BLOCKS = 64
BLOCK_TABLE_WIDTH = 64
DTYPE = torch.bfloat16


class DiffusionGemmaFlashinferPrefillTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        self.device = torch.device("cuda:0")
        self._diag_qo_len = -1

    def _log(self, stage: str) -> None:
        print(
            f"[dgm-fi-diag] {stage} ok (qo_len={self._diag_qo_len})", flush=True
        )

    def _make_op(self, backend: str = "auto") -> DiffusionGemmaAttnOp:
        op = DiffusionGemmaAttnOp(
            LAYER_TYPE_SLIDING,
            HEAD_NUM,
            KV_HEAD_NUM,
            HEAD_DIM,
            WINDOW,
            PAGE,
            DTYPE,
            None,
            backend=backend,
        )
        self.assertTrue(
            op.use_flashinfer,
            "flashinfer path unavailable; this regression is about the "
            "flashinfer sliding prefill",
        )
        # Instrument every GPU stage with a synchronize + labeled log so the
        # first failing kernel is named in the test log.
        test = self
        ragged_run = op.ragged_wrapper.run

        def ragged_run_logged(*args, **kwargs):
            out = ragged_run(*args, **kwargs)
            torch.cuda.synchronize()
            test._log("ragged run")
            return out

        op.ragged_wrapper.run = ragged_run_logged

        merge_state = op.merge_state_in_place

        def merge_logged(*args, **kwargs):
            out = merge_state(*args, **kwargs)
            torch.cuda.synchronize()
            test._log("merge_state")
            return out

        op.merge_state_in_place = merge_logged

        paged_run = op.paged_wrapper.run

        def paged_run_logged(*args, **kwargs):
            out = paged_run(*args, **kwargs)
            torch.cuda.synchronize()
            test._log("paged prefix run")
            return out

        op.paged_wrapper.run = paged_run_logged

        write_forward = op.write_op.forward

        def write_logged(key, value, kv_cache):
            write_forward(key, value, kv_cache)
            torch.cuda.synchronize()
            test._log("paged KV write")

        op.write_op.forward = write_logged
        return op

    def _make_inputs(self, qo_len: int, prefix_len: int):
        cu = torch.tensor([0, qo_len], dtype=torch.int32, device=self.device)
        block_table = torch.arange(
            1, 1 + BLOCK_TABLE_WIDTH, dtype=torch.int32
        ).unsqueeze(0)
        return SimpleNamespace(
            prefix_lengths=torch.tensor(
                [prefix_len], dtype=torch.int32, device=self.device
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

    def _make_cache(self):
        pool = torch.zeros(
            (POOL_BLOCKS, 2 * KV_HEAD_NUM * PAGE * HEAD_DIM),
            dtype=DTYPE,
            device=self.device,
        )
        return SimpleNamespace(kv_cache_base=pool, tag="swa_kv")

    def _forward(self, op, qo_len, prefix_len, causal, write_kv, kv_cache=None,
                 prefix_k=None, prefix_v=None):
        torch.manual_seed(1234)
        q = torch.randn(qo_len, HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device)
        k = torch.randn(qo_len, KV_HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device)
        v = torch.randn(qo_len, KV_HEAD_NUM, HEAD_DIM, dtype=DTYPE, device=self.device)
        attn_inputs = self._make_inputs(qo_len, prefix_len)
        if kv_cache is None:
            kv_cache = self._make_cache()
        self._diag_qo_len = qo_len
        op.plan(attn_inputs, causal)
        torch.cuda.synchronize()
        self._log("plan")
        out = op.forward(
            q,
            k,
            v,
            kv_cache,
            attn_inputs,
            torch.arange(qo_len, device=self.device),
            causal,
            write_kv,
        )
        torch.cuda.synchronize()
        self._log("forward end")
        ref = op._torch_attention(
            q,
            k,
            v,
            prefix_k,
            prefix_v,
            causal,
            attn_inputs.prefix_lengths,
            torch.tensor([0, qo_len], dtype=torch.int32),
        )
        max_diff = (out.float() - ref.float()).abs().max().item()
        print(
            f"[dgm-fi-diag] correctness qo_len={qo_len} causal={causal} "
            f"prefix={prefix_len} write={write_kv} max_diff={max_diff:.4e}",
            flush=True,
        )
        torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
        return kv_cache, q, k, v

    def test_prefill_causal_length_sweep(self):
        """Causal first-prefill lengths: 24/256 pass in CI smoke, 278 fails."""
        op = self._make_op()
        for qo_len in (24, 64, 128, 200, 256, 257, 278, 320, 539, 1025):
            with self.subTest(qo_len=qo_len):
                self._forward(op, qo_len, prefix_len=0, causal=True, write_kv=True)

    def test_prefill_causal_length_sweep_fa2(self):
        """Same sweep on the explicit fa2 backend: if auto (fa3 on sm90)
        fails and fa2 passes, the ragged kernel template is the culprit and
        fa2 is the fix candidate."""
        op = self._make_op(backend="fa2")
        for qo_len in (24, 256, 278, 539):
            with self.subTest(backend="fa2", qo_len=qo_len):
                self._forward(op, qo_len, prefix_len=0, causal=True, write_kv=True)

    def test_canvas_noncausal_with_prefix(self):
        """The propose-path shape: 256-token non-causal canvas over a prefix.

        The prefix KV is seeded by a causal prefill first (the smoke's own
        order: vision prefill 278 -> canvas 256).
        """
        op = self._make_op()
        cache, _, k_seed, v_seed = self._forward(
            op, 278, prefix_len=0, causal=True, write_kv=True
        )
        self._forward(
            op, 256, prefix_len=278, causal=False, write_kv=False,
            kv_cache=cache, prefix_k=k_seed, prefix_v=v_seed,
        )
        # The verify/commit round of the same vision request: causal canvas
        # forward over the 278-token prompt prefix, writing the canvas KV.
        self._forward(
            op, 256, prefix_len=278, causal=True, write_kv=True,
            kv_cache=cache, prefix_k=k_seed, prefix_v=v_seed,
        )

    def test_commit_causal_with_prefix(self):
        """The verify/commit shape from the passing text smoke."""
        op = self._make_op()
        cache, _, k_seed, v_seed = self._forward(
            op, 24, prefix_len=0, causal=True, write_kv=True
        )
        self._forward(
            op, 256, prefix_len=24, causal=True, write_kv=True,
            kv_cache=cache, prefix_k=k_seed[:24], prefix_v=v_seed[:24],
        )


if __name__ == "__main__":
    unittest.main()
