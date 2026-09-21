"""Exercise serving's actual prefill method, initial cache load and writeback."""

import json
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNetPrefill
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    RmsNormGated,
    layer_norm_fwd,
)
from rtp_llm.models_py.triton_kernels.common.prefill_fusion import prefill_fusion_scope
from rtp_llm.models_py.triton_kernels.fla.gdn_gating_prefill import gdn_gating_prefill


class PrefillDispatch(unittest.TestCase):
    @torch.inference_mode()
    def test_dispatch_and_cache(self):
        torch.manual_seed(815)
        report = {"status": "RUNNING", "cases": []}
        out = Path(
            os.environ.get(
                "GDN_BENCH_OUTPUT", os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")
            )
        )
        out.mkdir(parents=True, exist_ok=True)
        for lengths, prefix in [
            ([1], 0),
            ([63], 0),
            ([2048], 0),
            ([2053], 37),
            ([10007], 2048),
            ([24601], 0),
            ([24601], 37),
            ([40009], 0),
            ([37, 4096, 20468], 0),
        ]:
            t = sum(lengths)
            hv, hq = (4, 2) if t < 2048 else (64, 16)
            # scatter_qkv serving contract: post-conv packed QKV is contiguous.
            mixed = torch.randn(
                (2 * hq + hv) * 128, t, device="cuda", dtype=torch.bfloat16
            ).T.contiguous()
            packed = torch.randn(t, hv * 2, device="cuda", dtype=torch.bfloat16)
            a, b = packed.split(hv, dim=1)
            slots = max((prefix + n + 2047) // 2048 for n in lengths) + 1
            mapping = (
                torch.arange(
                    1, len(lengths) * slots + 1, device="cuda", dtype=torch.int32
                )
                .flip(0)
                .reshape(len(lengths), slots)
                .contiguous()
            )
            args = SimpleNamespace(
                input_lengths=torch.tensor(lengths),
                cu_seqlens_device=torch.tensor(
                    [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                    device="cuda",
                    dtype=torch.int32,
                ),
                prefix_lengths_device=torch.full(
                    (len(lengths),), prefix, device="cuda", dtype=torch.int32
                ),
                kv_cache_kernel_block_id_device=mapping,
            )
            cache = (
                torch.randn(1 + slots * len(lengths), hv + 1, 128, 128, device="cuda")[
                    :, :hv
                ]
                * 0.001
            )
            # Preserve a padded stride between cache blocks.
            initial = torch.empty(
                1 + slots * len(lengths), hv + 1, 128, 128, device="cuda"
            )
            initial[:, :hv].copy_(cache)
            initial[:, hv:] = 99.0
            obj = SimpleNamespace(
                alog=torch.randn(hv, device="cuda", dtype=torch.bfloat16),
                dt_bias=torch.randn(hv, device="cuda", dtype=torch.bfloat16),
                local_num_v_heads=hv,
                local_num_k_heads=hq,
                head_k_dim=128,
                head_v_dim=128,
                ssm_state_dtype=torch.float32,
                _get_ssm_states=lambda c: c[:, :hv],
            )

            def run(backend, fused):
                c = initial.clone()
                with prefill_fusion_scope(True), patch.dict(
                    os.environ,
                    {
                        "RTP_QWEN35_GDN_PREFILL_BACKEND": backend,
                        "RTP_QWEN35_PREFILL_FLASHINFER_GATING": "0",
                    },
                ), ExitStack() as stack:
                    if not fused:
                        # Reference dispatch deliberately invokes the original kernels.
                        stack.enter_context(
                            patch(
                                "rtp_llm.models_py.model_desc.qwen3_next.supports_gdn_gating_prefill",
                                return_value=False,
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rtp_llm.models_py.triton_kernels.fla.chunk.supports_exact_qk_norm",
                                return_value=False,
                            )
                        )
                    y = Qwen3NextGatedDeltaNetPrefill._fla(
                        obj, mixed, b, a, c, 2048, args
                    )
                return y, c

            ref, rc = run("native", False)
            with patch(
                "rtp_llm.models_py.model_desc.qwen3_next.gdn_gating_prefill",
                wraps=gdn_gating_prefill,
            ) as observed:
                exact, ec = run("native", True)
                self.assertEqual(observed.call_count, int(t >= 2048))
            self.assertTrue(torch.equal(ref, exact))
            self.assertTrue(torch.equal(rc, ec))
            actual, ac = run("flashinfer", True)
            torch.testing.assert_close(actual, ref, atol=0.01, rtol=0.01)
            torch.testing.assert_close(ac, rc, atol=0.01, rtol=0.01)
            self.assertTrue(torch.equal(ac[:, hv:], initial[:, hv:]))
            self.assertTrue(torch.equal(ac[0], initial[0]))
            report["cases"].append(
                {
                    "lengths": lengths,
                    "prefix": prefix,
                    "native_fused_bit_exact": True,
                    "flashinfer_max_abs": (actual - ref).abs().max().item(),
                    "flashinfer_cache_max_abs": (ac - rc).abs().max().item(),
                }
            )
            (out / "dispatch.json").write_text(json.dumps(report, indent=2))
            del (
                mixed,
                packed,
                a,
                b,
                args,
                cache,
                initial,
                obj,
                ref,
                rc,
                exact,
                ec,
                actual,
                ac,
            )
        # Test generic small normalization groups, including non-powers of two.
        for d in (32, 64, 96, 128, 192, 256):
            for shared in (False, True):
                x = torch.randn(2053, 4 * d, device="cuda", dtype=torch.bfloat16)
                z = torch.randn_like(x)
                for act in ("silu", "sigmoid"):
                    w = torch.randn(
                        d if shared else 4 * d, device="cuda", dtype=torch.bfloat16
                    )
                    bias = torch.randn_like(w)
                    norm = RmsNormGated(w, bias, group_size=d, activation=act)
                    expected = layer_norm_fwd(
                        x,
                        w,
                        bias,
                        norm.eps,
                        z=z,
                        group_size=d,
                        norm_before_gate=True,
                        is_rms_norm=True,
                        activation=act,
                    )[0]
                    with prefill_fusion_scope(True):
                        actual = norm(x, z)
                    self.assertTrue(torch.equal(actual, expected), (d, shared, act))
        report["status"] = "PASS"
        (out / "dispatch.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
