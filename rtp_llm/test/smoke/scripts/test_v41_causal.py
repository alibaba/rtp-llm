"""CPU-only protocol and live-call wiring; not a substitute for paired GB200 inference."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules import v41_causal as trace
from rtp_llm.test.smoke.scripts.compare_v41_causal import CORE, PROMPT, compare


class Identity(torch.nn.Module):
    def forward(self, x, *args):
        return x


class HC(Identity):
    def pre(self, x):
        return x[:, 0], x[..., :1], x[..., :4]


class Attn(Identity):
    layer_id = 1
    compress_ratio = 0

    def _swa_entries_per_block(self):
        return 128

    def _swa_prefill_concat(self, qkv, common):
        return torch.full((1, 398, 512), self.seed, dtype=torch.bfloat16)

    def _prefill_compute_qkv(self, x, common):
        return NS(qr=x, kv_full=x)

    def _project_prefill_q(self, x):
        return x

    def _prefill_sparse_attention(self, q, *, indices, topk_length):
        return q

    def _prefill_output_proj_into(self, x, *, out):
        out.copy_(x)

    def _try_fused_qr_kv(self, *args):
        return None

    def _host_prefill_prefixes(self, _):
        return [512]

    def _host_prefill_lengths(self, _):
        return [271]


class CausalTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
    )
    @torch.inference_mode()
    def test_actual_gate_launch_is_observed_without_changing_output(self):
        from rtp_llm.models_py.modules.dsv41.engram import gated_engram_residual
        from rtp_llm.test.smoke.scripts.compare_v41_causal import compare_gate

        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("requires native BF16")
        rng = torch.Generator(device="cuda").manual_seed(81)
        h, kv, q, k = [
            torch.randn(s, generator=rng, device="cuda", dtype=torch.bfloat16)
            for s in ((17, 4, 5120), (17, 25600), (4, 5120), (4, 5120))
        ]
        mask = torch.arange(17, device="cuda") % 5 != 0
        expected = gated_engram_residual(h.clone(), kv, q, k, 1e-20, mask)
        record = {}
        out = trace._gate_witness(
            gated_engram_residual, (h, kv, q, k, 1e-20, mask), {"out": h}, mask, record
        )
        self.assertTrue(torch.equal(expected, out))
        self.assertEqual(out.data_ptr(), h.data_ptr())
        result = compare_gate(record, record)
        self.assertEqual(result["actual_path"], ["triton", "triton"])
        self.assertRegex(result["cubin_sha256"][0], "^[0-9a-f]{64}$")
        self.assertEqual(result["actual_matches_native"], [True, True])
        launch = record["engram_gate"]["launches"]["actual"]
        self.assertTrue(launch["tensor_stream"])
        self.assertRegex(
            launch["code_segments"][".text.engram_inject_kernel"], "^[0-9a-f]{64}$"
        )
        self.assertEqual(
            set(record["engram_gate"]["launches"]), {"actual", "dense_native"}
        )

    def test_native_linear_and_shared_workspace_operands(self):
        raw, packed, actual = (torch.tensor([n]) for n in (1, 2, 3))
        direct = NS(weight=actual, weight_scales=packed)
        wrapped = NS(weight=raw, weight_scales=raw, v41_runtime_linear=direct)
        self.assertEqual(trace._native_pair(wrapped), (actual, packed))
        self.assertEqual(
            trace._native_pair(NS(weight=raw, v41_runtime_linear=None)), (None, None)
        )
        shared = NS(w13=wrapped, w2=direct)
        fast = NS(
            _prepared_shared_experts=shared,
            _w13_parts=(packed, actual),
            _w2_parts=(raw, actual),
        )
        self.assertEqual(
            trace._shared_pairs(shared), ((actual, packed), (actual, packed))
        )
        self.assertEqual(
            trace._shared_pairs(shared, fast), (fast._w13_parts, fast._w2_parts)
        )
        with self.assertRaisesRegex(ValueError, "unbound"):
            trace._shared_pairs(NS(), fast)

    def test_lookup_uses_live_uva_source_and_target_index_device(self):
        h = torch.tensor([[1], [2], [3]], dtype=torch.uint8)
        # Exercise source selection on CPU; do not claim this runs a CUDA UVA read.
        table = NS(weight=h, scales=h, _uva=(h + 10, h + 20))
        ids = torch.tensor([2, 0])
        host = dict(trace._lookup_rows(table, ids, False))
        gpu_abi = dict(trace._lookup_rows(table, ids, True))
        self.assertEqual(host["weight"].flatten().tolist(), [3, 1])
        self.assertEqual(gpu_abi["weight"].flatten().tolist(), [13, 11])
        self.assertEqual(gpu_abi["scales"].flatten().tolist(), [23, 21])

    def test_real_call_private_transport(self):
        self._real_call_private_transport("cpu")

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
    )
    @torch.inference_mode()
    def test_hit512_installs_engram_and_moe_hooks_together(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("requires native BF16")
        self._real_call_private_transport("cuda")

    def _real_call_private_transport(self, device):
        from safetensors import safe_open

        with TemporaryDirectory() as a, TemporaryDirectory() as b, TemporaryDirectory() as tmp:
            nonce, run = "9" * 32, "a" * 32
            fixture = Path(tmp) / "w/rtp_llm/test/smoke/arm_causal_inputs"
            fixture.mkdir(parents=True)

            def invoke(role, root, seed, mode="natural"):
                root = Path(root)
                rank = 0
                ids = torch.tensor([2, 4])
                valid = torch.tensor([True, False])
                cp = NS(
                    cp_size=2,
                    cp_rank=rank,
                    local_is_real=valid,
                    global_positions=torch.tensor([639, 1]),
                )
                h = torch.full((2, 4, 5120), seed, dtype=torch.bfloat16, device=device)
                prev = NS(pre_mix_out=torch.full((2, 4), seed, dtype=torch.float32))
                hc = HC()
                hc._previous_ref = lambda: prev
                net = NS(
                    attn=Attn(),
                    attn_hc=hc,
                    ffn_hc=HC(),
                    ffn_norm=Identity(),
                    attn_norm=Identity(),
                    engram=Identity(),
                    ffn=Identity(),
                    engram_token_mask=valid,
                    engram_hashes=torch.zeros((2, 24), dtype=torch.long),
                )
                net.engram.wkv = Identity()
                net.engram.q_weight = torch.ones_like(h[0])
                net.engram.k_weight = torch.ones_like(h[0])
                net.engram.embed_tokens = NS(
                    weight=torch.zeros((1, 256), dtype=torch.uint8),
                    scales=torch.zeros((1, 8), dtype=torch.uint8),
                )
                net.attn.seed = seed
                for field in ("wq_a", "wkv", "wq_b", "wo_b"):
                    setattr(net.attn, field, Identity())
                net.ffn.gate = NS(
                    _weight_bf16=lambda: h[0], prepare_gate_payload=lambda: NS(scores=h[:, 0])
                )
                net.ffn.fused_moe = NS(
                    fused_experts=NS(forward_gate_pack=lambda: None),
                    includes_shared_expert=False,
                )
                net.ffn.shared_experts = None
                net._try_mega_mhc = lambda *args: None
                net.ffn_hc.pre_mix_out = prev.pre_mix_out
                state = NS(root=root, run=run, phase="hit", prefix=512)
                if mode != "natural":
                    (root / "v41_causal_replay_id").write_text(run + " " + mode)
                used = {}

                def block(x, input_ids):
                    if device == "cuda":
                        from rtp_llm.models_py.modules.dsv41 import engram

                        kv = torch.cat([x.flatten(1), x[:, 0]], dim=-1)
                        engram.gated_engram_residual(
                            x, kv, net.engram.q_weight, net.engram.k_weight,
                            1e-20, valid.to(device),
                        )
                    net.ffn.gate.prepare_gate_payload()
                    common = NS(
                        batch_size=1,
                        cp_on=True,
                        any_cont=True,
                        swa_meta=NS(
                            cache_gather_lens=torch.tensor([127]),
                            cache_slot_mapping=torch.arange(385, 512).view(1, -1),
                        ),
                    )
                    buf = net.attn._swa_prefill_concat(None, common)
                    used.update(
                        x=x.clone(), mix=prev.pre_mix_out.clone(), raw=buf.clone()
                    )
                    return x

                with patch.dict(
                    os.environ,
                    dict(
                        RTP_V41_CAUSAL_ROLE=role,
                        RTP_ARM_PAIR_NONCE=nonce,
                        RTP_V41_EXPECTED_LOADER=role,
                        TEST_TARGET=trace.TARGET,
                        TEST_SRCDIR=tmp,
                        TEST_WORKSPACE="w",
                    ),
                ):
                    trace.observe(state, block, net, cp, h, ids)
                path = root / f"v41_causal_{mode}_r{rank}.json"
                document = json.loads(path.read_bytes())
                self.assertIn("gate.scores", dict(document["events"]))
                if device == "cuda":
                    self.assertTrue(document["engram_gate"]["complete"])
                self.assertEqual(prev.pre_mix_out[0, 0].item(), seed)
                return used

            original = invoke("oldfst", a, 3)
            old_path = Path(a) / "arm_causal_private/v41_r0.safetensors"
            with safe_open(str(old_path), framework="pt") as f:
                self.assertEqual(set(f.keys()), trace.KEYS)
                self.assertEqual(f.get_tensor("incoming").shape, (1, 4, 5120))
            (fixture / "v41_r0.safetensors").symlink_to(old_path)
            for mode, (use_mix, use_swa) in trace.CELLS.items():
                old = invoke("oldfst", a, 3, mode)
                new = invoke("newfst", b, 9, mode)
                self.assertTrue(torch.equal(old["x"], new["x"]))
                self.assertEqual(new["mix"][0, 0].item(), 3 if use_mix else 9)
                self.assertEqual(new["raw"][0, 126, 0].item(), 3 if use_swa else 9)
                self.assertEqual(new["raw"][0, 127, 0].item(), 9)
            self.assertEqual(
                original["x"][1, 0, 0].item(), 3
            )  # natural padding unchanged
            self.assertFalse((Path(b) / "arm_causal_private").exists())
            self.assertTrue(
                torch.equal(old["x"], new["x"]) and torch.equal(old["mix"], new["mix"])
            )
            self.assertTrue(torch.equal(old["raw"][:, :127], new["raw"][:, :127]))
            self.assertEqual(new["raw"][0, 127, 0].item(), 9)  # fresh KV never replaced
            self.assertEqual(new["x"][1].count_nonzero().item(), 0)
            with patch.dict(os.environ, dict(TEST_SRCDIR=tmp, TEST_WORKSPACE="w")):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    trace._read(0, "0" * 32, torch.tensor([639]))
                with self.assertRaisesRegex(ValueError, "different rows"):
                    trace._read(0, nonce, torch.tensor([640]))

    def test_compare_rejects_nonidentical_replay(self):
        with TemporaryDirectory() as tmp:
            roots = [Path(tmp) / k for k in ["oldfst", "newfst"]]
            for root in roots:
                root.mkdir()
                for rank in range(2):
                    pos = list(range(512 + rank, 783, 2))
                    for i, mode in enumerate(("natural", *trace.CELLS)):
                        val = dict(
                            sha256="2" * 64, shape=[len(pos), 4], dtype="torch.float32"
                        )
                        names = set(CORE)
                        if mode != "natural":
                            names.update(("incoming.used", "previous_mix.used"))
                        rec = dict(
                            schema=1,
                            family="v41",
                            nonce="7" * 32,
                            mode=mode,
                            rank=rank,
                            role=root.name,
                            positions=pos,
                            cp=dict(
                                size=2,
                                rank=rank,
                                local_rows=len(pos),
                                live_rows=list(range(len(pos))),
                            ),
                            request_id=f"{i+1:032x}",
                            swa_geometry=dict(
                                entries=128,
                                logical_start=385,
                                slots=[[0, k % 128] for k in range(385, 512)],
                            ),
                            outcome=dict(
                                input_digest=PROMPT,
                                reuse=512,
                                decode_reuse=0,
                                output_count=2,
                                output_hash="c" * 64,
                                accepts=[0],
                                matches_golden=True,
                            ),
                            weights={"w": val},
                            skipped=[],
                            events=[[s, val] for s in sorted(names)],
                        )
                        (root / f"v41_causal_{mode}_r{rank}.json").write_text(
                            json.dumps(rec)
                        )
            self.assertEqual(compare(*roots)["status"], "compared")
            # Give only the New mix a causal effect; SWA has no contrast.
            for rank in (0, 1):
                for mode in ("natural", *trace.CELLS):
                    path = roots[1] / f"v41_causal_{mode}_r{rank}.json"
                    doc = json.loads(path.read_text())
                    ev = dict(doc["events"])
                    ev["previous_mix.observed"]["sha256"] = "a" * 64
                    local_mix = mode == "natural" or not trace.CELLS[mode][0]
                    if local_mix:
                        if mode != "natural":
                            ev["previous_mix.used"]["sha256"] = "a" * 64
                        ev["outgoing"]["sha256"] = "d" * 64
                        doc["outcome"].update(output_hash="e" * 64, accepts=[1])
                    path.write_text(json.dumps(doc))
            result = compare(*roots)
            mix = result["new_factor_contrasts"]["mix_at_natural_swa"]
            self.assertEqual(
                (
                    mix["operand_changed"],
                    mix["layer1_changed"],
                    mix["output_changed"],
                    mix["accepted"],
                ),
                (True, True, True, [1, 0]),
            )
            self.assertFalse(
                result["new_factor_contrasts"]["swa_at_natural_mix"]["operand_changed"]
            )
            target = roots[1] / "v41_causal_replay_r1.json"
            doc = json.loads(target.read_text())
            dict(doc["events"])["incoming.used"]["sha256"] = "d" * 64
            target.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "not shared"):
                compare(*roots)


if __name__ == "__main__":
    unittest.main()
