"""GPU A/B for DiffusionGemmaExperts: fused Triton MoE vs reference loop.

Loads one real layer's routed-expert + router weights from the real 26B
checkpoint (env DGM_REAL_CKPT, bf16, cuda), computes the REAL router
top-8 (the routing chain is identical for both executors), then runs the
fused executor and the gather-per-expert loop on the same routing and
compares elementwise. Agreement criterion is bf16 GEMM/accumulate noise
(run35/36 precedent: max_diff / absmax < 3e-2); the fused path must be
asserted active so the A/B cannot silently degenerate to loop-vs-loop.
"""

import importlib.util
import os
import unittest

import torch

CKPT = os.environ.get("DGM_REAL_CKPT", "")

# Load the realweight helper module by path (test dir is not a package):
# its module-level constants/geometry (H=2816, EXPERTS=128, TOPK=8, ...) and
# its _load_hf_weights/_make_real_config loaders are reused here.
_RW_PATH = os.path.join(os.path.dirname(__file__), "diffusion_gemma_realweight_test.py")
_spec = importlib.util.spec_from_file_location("dgm_rw_helpers", _RW_PATH)
_rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rw)

from rtp_llm.ops import ParallelismConfig  # noqa: E402
from rtp_llm.models_py.model_desc import diffusion_gemma as dgm  # noqa: E402
from rtp_llm.utils.model_weight import W  # noqa: E402


@unittest.skipUnless(CKPT, "DGM_REAL_CKPT not set; real-weights A/B skipped")
class DiffusionGemmaFusedMoeABTest(unittest.TestCase):
    def _run_ab(self, layer_id: int, layer_type: str, num_tokens: int, seed: int) -> float:
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        hf = _rw._load_hf_weights(layer_id, layer_type)
        cfg = _rw._make_real_config()
        device = torch.device("cuda:0")
        expert_weights = {
            W.moe_w1: hf["moe_gate_up"].contiguous().to(torch.bfloat16).to(device),
            W.moe_w2: hf["moe_down"].contiguous().to(torch.bfloat16).to(device),
        }
        experts_fused = dgm.DiffusionGemmaExperts(cfg, expert_weights)
        experts_loop = dgm.DiffusionGemmaExperts(
            cfg, {k: v for k, v in expert_weights.items()}
        )
        experts_loop._fused_ready = False
        router = dgm.DiffusionGemmaRouter(
            cfg,
            ParallelismConfig(),
            {
                W.moe_gate: hf["router_proj"].T.contiguous().to(torch.bfloat16).to(device),
                W.dgm_router_scale: hf["router_scale"].reshape(-1).to(torch.bfloat16).to(device),
                W.dgm_router_per_expert_scale: hf["per_expert_scale"].reshape(-1).to(torch.bfloat16).to(device),
            },
            None,
        )
        torch.manual_seed(seed)
        hidden = torch.randn(num_tokens, _rw.H, device=device, dtype=torch.bfloat16)
        top_w, top_i = router(hidden)
        self.assertTrue(experts_fused._fused_usable(hidden))
        out_fused = experts_fused(hidden, top_w, top_i)
        out_loop = experts_loop(hidden, top_w, top_i)
        diff = (out_fused.float() - out_loop.float()).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        absmax = float(out_loop.float().abs().max())
        self.assertGreater(absmax, 0.0)
        ratio = max_diff / absmax
        print(
            f"[dgm-moe-ab] layer={layer_id}({layer_type}) M={num_tokens} seed={seed}: "
            f"max_diff={max_diff:.5f} mean_diff={mean_diff:.6f} absmax={absmax:.3f} "
            f"ratio={ratio:.5f} active_experts={int(top_i.unique().numel())}",
            flush=True,
        )
        return ratio

    def test_fused_moe_matches_expert_loop(self):
        # 1024x8/128 = 64 rows/expert -> BLOCK_M=64 branch; 21 -> BLOCK_M=16
        # incremental-prefill shape; 256 -> BLOCK_M=16 batched-prefill shape.
        ratios = [
            self._run_ab(0, "sliding_attention", 1024, 11),
            self._run_ab(0, "sliding_attention", 21, 12),
            self._run_ab(5, "full_attention", 256, 13),
        ]
        for ratio in ratios:
            self.assertLess(ratio, 3e-2)


if __name__ == "__main__":
    unittest.main()
