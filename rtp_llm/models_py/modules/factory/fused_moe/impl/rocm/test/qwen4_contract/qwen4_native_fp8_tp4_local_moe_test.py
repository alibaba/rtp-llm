"""Bounded Qwen4 TP4/EP1 loader/executor contract; no checkpoint I/O or RCCL.

Official E512/K2560/topk10/N640; block128 TP4 padding makes local N256.
E16 exercises native FN split/postprocess into the actual ROCm executor.
E512 uses native local weights after real postprocess for changed-input graph replay.
"""

import gc
import json
import sys
import unittest

import torch
import torch.nn.functional as F
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.test import (
    rocm_fp8_fused_moe_test as scaffold,
)


class Qwen4LocalMoeTest(unittest.TestCase):
    E, K, N, TOP_K = 512, 2560, 256, 10

    def setUp(self):
        # This explicit ROCm target must not silently PASS via SkipTest.
        self.assertIsNone(scaffold._IMPORT_ERROR)
        self.assertTrue(torch.cuda.is_available() and torch.version.hip)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(512)
        self.dtype = scaffold.get_rocm_fp8_dtype()
        self.assertEqual(self.dtype, torch.float8_e4m3fnuz)
        self.dtype = torch.float8_e4m3fn
        self.w1, self.s1 = self.make_weight(2 * self.N, self.K)
        self.w2, self.s2 = self.make_weight(self.K, self.N)
        config = scaffold._make_config_adapter(self.E, self.TOP_K, 1024)
        config.parallelism_config.tp_size = 4
        config.parallelism_config.world_size = 4
        config.model_config.attn_config.head_num = 40
        config.model_config.hidden_size = self.K
        config.model_config.moe_inter_size = 1024
        config = scaffold.MoEConfigAdapter(
            model_config=config.model_config,
            parallelism_config=config.parallelism_config,
            moe_config=scaffold.MoeConfig(),
        )
        self.assertEqual(config.hidden_size, self.K)
        self.assertEqual(config.tp_size, 4)
        self.executor = scaffold.RocmExpertsFp8PerBlock(
            config,
            scaffold.FusedMoEQuantConfig(),
            self.postprocess_local_weights(),
        )

    def postprocess_local_weights(self):
        from types import SimpleNamespace

        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.device import get_current_device
        from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
        from rtp_llm.models.qwen4_exp.native_fp8 import Qwen4NativeFp8MoeWeight
        from rtp_llm.utils.model_weight import CkptWeightInfo, W

        cfg = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            use_swizzleA=False,
            exported_device=get_current_device(),
        )
        local = {}
        for name, projections in ((W.moe_w1, ("up", "gate")), (W.moe_w2, ("down",))):
            raw = MoeAtomicWeight(
                name,
                [
                    CkptWeightInfo(f"layers.{{i}}.experts.{{expert_id}}.{x}.weight")
                    for x in projections
                ],
                config=MoeConfig(expert_num=self.E),
            )
            desc = Qwen4NativeFp8MoeWeight(
                raw, Fp8BlockWiseQuantConfig(is_quanted=True)
            )
            if name == W.moe_w1:
                # Reference stores logical gate/up; loader input is up/gate.
                values = {
                    name: torch.cat((self.w1[:, self.N :], self.w1[:, : self.N]), 1),
                    W.moe_s1: torch.cat(
                        (self.s1[:, self.N // 128 :], self.s1[:, : self.N // 128]), 1
                    ),
                }
            else:
                values = {name: self.w2.clone(), W.moe_s2: self.s2.clone()}
            local.update(desc._postprocess(values, "cuda", cfg))
        self.assertEqual(local[W.moe_w1].dtype, torch.float8_e4m3fnuz)
        return local

    def make_weight(self, rows, cols):
        q = torch.empty(self.E, rows, cols, dtype=self.dtype, device="cuda")
        scale = torch.empty(self.E, rows // 128, cols // 128, device="cuda")
        # No full BF16/FP32 expert bank: bounded eight-expert temporary.
        for start in range(0, self.E, 8):
            w = torch.randn(8, rows, cols, dtype=torch.bfloat16, device="cuda") * 0.04
            blocks = w.reshape(8, rows // 128, 128, cols // 128, 128)
            sw = blocks.abs().amax((2, 4)).clamp_min(1e-8).float() / 448
            qw = (blocks / sw[:, :, None, :, None]).to(self.dtype).reshape_as(w)
            q[start : start + 8].copy_(qw)
            scale[start : start + 8].copy_(sw)
        return q, scale

    def payload(self, m, offset):
        ids = (
            (
                torch.arange(m * self.TOP_K, device="cuda").reshape(m, self.TOP_K) * 13
                + offset
            )
            % self.E
        ).to(torch.int32)
        return scaffold.ExpertForwardPayload(
            expert_x=torch.randn(m, self.K, dtype=torch.bfloat16, device="cuda") * 0.2,
            expert_x_origin_dtype=torch.bfloat16,
            expert_x_scale=None,
            expert_tokens_meta=scaffold.ExpertTokensMetadata(None, None, None),
            expert_topk_ids=ids,
            expert_topk_weights=torch.softmax(
                torch.randn(m, self.TOP_K, device="cuda"), -1
            ),
        )

    def forward(self, p):
        return self.executor.execute(
            payload=p,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        ).fused_expert_output

    def reference(self, p):
        # Same independent BF16 arithmetic as torch_moe_ref, but dequantize
        # only one selected expert at a time (no ~2GiB BF16 reference bank).
        out = torch.zeros(
            p.expert_x.shape[0], self.TOP_K, self.K, dtype=torch.bfloat16, device="cuda"
        )
        for e in p.expert_topk_ids.unique().cpu().tolist():
            mask = p.expert_topk_ids == e
            x = p.expert_x[:, None, :].expand(-1, self.TOP_K, -1)[mask]
            w1 = scaffold._dequant_per_block(self.w1[e : e + 1], self.s1[e : e + 1])[0]
            w2 = scaffold._dequant_per_block(self.w2[e : e + 1], self.s2[e : e + 1])[0]
            gate, up = (x @ w1.T).chunk(2, -1)
            out[mask] = (F.silu(gate) * up) @ w2.T
        return (out * p.expert_topk_weights[..., None]).sum(1).to(torch.bfloat16)

    def test_eager_and_changed_input_graph(self):
        evidence = []
        for m in (1, 16):
            with self.subTest(tokens=m):
                p, changed = self.payload(m, 0), self.payload(m, 511)
                ref = self.reference(p)
                expected = self.forward(p)
                self.assertTrue(torch.isfinite(expected).all().item())
                # Report relative L2 as well; zero output cannot pass this gate.
                rel = (
                    (expected.float() - ref.float()).norm() / ref.float().norm()
                ).item()
                self.assertLess(rel, 0.08)
                torch.testing.assert_close(expected, ref, atol=0.05, rtol=0.05)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.forward(p)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self.forward(p)
                graph.replay()

                def difference(a, b):
                    return dict(
                        max_abs=(a.float() - b.float()).abs().max().item(),
                        relative_l2=(
                            (a.float() - b.float()).norm() / b.float().norm()
                        ).item(),
                        exact=torch.equal(a, b),
                    )

                def check_reference(value, reference):
                    self.assertTrue(torch.isfinite(value).all().item())
                    error = difference(value, reference)["relative_l2"]
                    self.assertLess(error, 0.08)
                    torch.testing.assert_close(value, reference, atol=0.05, rtol=0.05)
                    return error

                initial_diff = difference(captured, expected)
                graph_reference_l2 = [check_reference(captured, ref)]
                eager_diffs = []
                for _ in range(2):
                    another = self.forward(p)
                    check_reference(another, ref)
                    eager_diffs.append(difference(another, expected))
                repeated = captured.clone()
                graph.replay()
                replay_diff = difference(captured, repeated)
                graph_reference_l2.append(check_reference(captured, ref))
                before = captured.clone()
                for field in ("expert_x", "expert_topk_ids", "expert_topk_weights"):
                    getattr(p, field).copy_(getattr(changed, field))
                changed_ref = self.reference(p)
                changed_eager = self.forward(p)
                check_reference(changed_eager, changed_ref)
                graph.replay()
                torch.cuda.synchronize()
                changed_diff = difference(captured, changed_eager)
                changed_rel = (
                    (captured.float() - changed_ref.float()).norm()
                    / changed_ref.float().norm()
                ).item()
                self.assertLess(changed_rel, 0.08)
                torch.testing.assert_close(captured, changed_ref, atol=0.05, rtol=0.05)
                self.assertFalse(torch.equal(before, captured), "stale graph output")
                peak = torch.cuda.max_memory_allocated()
                self.assertLess(peak, 4 * 1024**3, "bounded component exceeded 4GiB")
                print(
                    "GRAPH_NUMERICS="
                    + json.dumps(
                        dict(
                            M=m,
                            reference_l2=rel,
                            graph_reference_l2=graph_reference_l2,
                            changed_reference_l2=changed_rel,
                            initial=initial_diff,
                            eager_repeats=eager_diffs,
                            replay_repeat=replay_diff,
                            changed=changed_diff,
                            peak_allocated_bytes=peak,
                        )
                    ),
                    flush=True,
                )
                evidence.append(
                    dict(
                        M=m,
                        E=self.E,
                        K=self.K,
                        N=self.N,
                        top_k=self.TOP_K,
                        relative_l2=rel,
                        changed_relative_l2=changed_rel,
                        peak_allocated_bytes=peak,
                        graph_changed_inputs=True,
                    )
                )
                del graph, captured, expected, before
                gc.collect()
        print("QWEN4_LOCAL_MOE_EVIDENCE=" + json.dumps(evidence), flush=True)


class NativeLoaderContractTest(unittest.TestCase):
    def test_native_fn_split_postprocess_executor(self):
        from types import SimpleNamespace

        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.device import get_current_device
        from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
        from rtp_llm.models.qwen4_exp.native_fp8 import Qwen4NativeFp8MoeWeight
        from rtp_llm.utils.model_weight import CkptWeightInfo, W

        self.assertIsNone(scaffold._IMPORT_ERROR)
        self.assertTrue(torch.cuda.is_available() and torch.version.hip)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(640)
        h = Qwen4LocalMoeTest()
        h.E, h.K, h.N, h.TOP_K = 16, 2560, 256, 10
        p = h.payload(16, 0)
        source = {}
        for name, rows, cols in ((W.moe_w1, 1280, 2560), (W.moe_w2, 2560, 640)):
            w = torch.randn(16, rows, cols, dtype=torch.bfloat16, device="cuda") * 0.04
            # Official FN data, deliberately block-dependent nonuniform scales.
            blocks = w.reshape(16, rows // 128, 128, cols // 128, 128)
            scale = blocks.abs().amax((2, 4)).clamp_min(1e-8).float() / 448
            q = (
                (blocks / scale[:, :, None, :, None])
                .to(torch.float8_e4m3fn)
                .reshape_as(w)
            )
            source[name] = q
            source[W.moe_s1 if name == W.moe_w1 else W.moe_s2] = scale
        device = get_current_device()
        evidence = []
        for rank in range(4):
            cfg = SimpleNamespace(
                tp_size=4,
                tp_rank=rank,
                ep_size=1,
                ep_rank=0,
                dp_size=1,
                dp_rank=0,
                ffn_tp_size=4,
                ffn_tp_rank=rank,
                hidden_size=2560,
                head_num=40,
                head_num_kv=40,
                size_per_head=64,
                moe_pure_tp_mode=True,
                moe_pure_tp_preshard=True,
                bit=8,
                merge_lora=False,
                compute_dtype=torch.bfloat16,
                use_swizzleA=False,
                get_selected_experts=lambda layer, count: range(count),
                exported_device=device,
            )
            local = {}
            for name, projections in (
                (W.moe_w1, ("up", "gate")),
                (W.moe_w2, ("down",)),
            ):
                raw = MoeAtomicWeight(
                    name,
                    [
                        CkptWeightInfo(f"layers.{{i}}.experts.{{expert_id}}.{x}.weight")
                        for x in projections
                    ],
                    config=MoeConfig(expert_num=16),
                )
                desc = Qwen4NativeFp8MoeWeight(
                    raw, Fp8BlockWiseQuantConfig(is_quanted=True)
                )
                scales = W.moe_s1 if name == W.moe_w1 else W.moe_s2
                split = desc._split(
                    {name: source[name].clone(), scales: source[scales].clone()}, cfg
                )
                local.update(desc._postprocess(split, "cuda", cfg))
            self.assertEqual(tuple(local[W.moe_w1].shape), (16, 512, 2560))
            self.assertEqual(local[W.moe_w1].dtype, torch.float8_e4m3fnuz)
            # Independently slice original logical [up,gate] FN weights; no
            # loader shuffle/rewrite calls participate in this reference.
            start, valid = rank * 256, max(0, min(256, 640 - rank * 256))
            h.w1 = torch.zeros(16, 512, 2560, dtype=torch.float8_e4m3fn, device="cuda")
            h.w2 = torch.zeros(16, 2560, 256, dtype=torch.float8_e4m3fn, device="cuda")
            h.s1 = torch.ones(16, 4, 20, device="cuda")
            h.s2 = torch.ones(16, 20, 2, device="cuda")
            if valid:
                h.w1[:, :valid].copy_(
                    source[W.moe_w1][:, 640 + start : 640 + start + valid]
                )
                h.w1[:, 256 : 256 + valid].copy_(
                    source[W.moe_w1][:, start : start + valid]
                )
                h.w2[:, :, :valid].copy_(source[W.moe_w2][:, :, start : start + valid])
                b, n = start // 128, valid // 128
                h.s1[:, :n].copy_(source[W.moe_s1][:, 5 + b : 5 + b + n])
                h.s1[:, 2 : 2 + n].copy_(source[W.moe_s1][:, b : b + n])
                h.s2[:, :, :n].copy_(source[W.moe_s2][:, :, b : b + n])
            ref = h.reference(p)
            adapter = scaffold._make_config_adapter(16, 10, 1024)
            h.executor = scaffold.RocmExpertsFp8PerBlock(
                adapter, scaffold.FusedMoEQuantConfig(), local
            )
            out = h.forward(p)
            self.assertTrue(torch.isfinite(out).all().item())
            rel = (
                (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20)
            ).item()
            print(
                "NATIVE_LOADER_RANK="
                + json.dumps(
                    dict(
                        rank=rank,
                        valid_channels=valid,
                        relative_l2=rel,
                        output_norm=out.float().norm().item(),
                        reference_norm=ref.float().norm().item(),
                    )
                ),
                flush=True,
            )
            if valid:
                self.assertLess(rel, 0.08)
            torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)
            if not valid:
                self.assertEqual(out.count_nonzero().item(), 0)
            peak = torch.cuda.max_memory_allocated()
            self.assertLess(peak, 1024**3)
            evidence.append(
                dict(
                    rank=rank,
                    valid_channels=valid,
                    relative_l2=rel,
                    peak_allocated_bytes=peak,
                )
            )
        print("NATIVE_LOADER_EVIDENCE=" + json.dumps(evidence), flush=True)


if __name__ == "__main__":
    test_argv = sys.argv[:]
    # Device construction parses server flags; keep unittest selectors separate.
    sys.argv = sys.argv[:1]
    unittest.main(argv=test_argv)
