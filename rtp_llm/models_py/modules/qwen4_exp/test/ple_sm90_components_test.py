"""Exact PLE leaf and speculative-state contracts on supported CUDA devices."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from ple_sm90_test_utils import (
    capture_kernel,
    exact,
    metadata,
    reference_conv,
    reference_gate,
    save,
    sha,
)

from rtp_llm.models_py.modules.qwen4_exp.ple import ple_conv_triton as conv
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_gate_triton as gate
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_page_projection as projection
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer


class PleSM90ComponentsTest(unittest.TestCase):
    def tensors(self, batch, rows, seed):
        torch.manual_seed(seed)
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        key = torch.randn(batch, rows, 10240, **kw)
        value = torch.randn(batch, rows, 2560, **kw)
        hidden = torch.randn_like(key)
        gammas = [torch.randn(10240, **kw) * 0.07 for _ in range(3)]
        return key, value, hidden, *gammas

    def test_cpu_guards(self):
        key = torch.empty(1, 1, 10240, dtype=torch.bfloat16)
        value = torch.empty(1, 1, 2560, dtype=torch.bfloat16)
        gamma = torch.empty(10240, dtype=torch.bfloat16)
        weight = torch.empty(10240, 1, 4, dtype=torch.bfloat16)
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("CPU must not query CUDA"),
        ):
            self.assertIsNone(
                gate.try_ple_gate(key, value, key, gamma, gamma, gamma, 1e-6)
            )
            self.assertIsNone(conv.try_ple_conv(key, key, None, weight, 3))
        for page in (64, 128, 192, 256, 320, 512, 768, 1024, 1088, 32768):
            self.assertTrue(projection._valid_page_size(page))
        for page in (True, False, 0, -64, 1, 32, 63, 65, 96, 64.0, "64"):
            self.assertFalse(projection._valid_page_size(page))
            with self.assertRaises(ValueError):
                projection.quantize_pages(
                    torch.zeros(1, 2560), torch.ones(1), torch.ones(1), page
                )
        for capability, expected in (
            ((9, 0), True),
            ((9, 1), True),
            ((12, 0), True),
            ((12, 1), False),
            ((10, 0), False),
            ((8, 9), False),
        ):
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                for module in (gate, conv, projection):
                    self.assertEqual(
                        module._sm_supported(torch.device("cuda")), expected
                    )
        save("component_cpu_guards", {"status": "PASS"})

    def test_gate_conv_exact(self):
        record = {
            "metadata": metadata(),
            "source": {"gate": sha(gate.__file__), "conv": sha(conv.__file__)},
            "cases": [],
            "status": "RUNNING",
        }
        try:
            for batch, rows in [
                (1, n) for n in (1, 4, 8, 9, 63, 64, 65, 127, 128, 129, 5120, 20480)
            ] + [(2, 65)]:
                tensors = self.tensors(batch, rows, 12000 + rows + batch)
                expected = reference_gate(*tensors)
                with capture_kernel(gate, "_ple_gate") as gcalls:
                    actual = gate.try_ple_gate(*tensors, 1e-6)
                row = {
                    "batch": batch,
                    "rows": rows,
                    "gate_calls": gcalls,
                    "gate": [exact(a, b) for a, b in zip(actual, expected)],
                }
                record["cases"].append(row)
                self.assertEqual(len(gcalls), 1)
                self.assertTrue(all(x["pass"] for x in row["gate"]), row)
                weight = (
                    torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1
                )
                row["conv"] = []
                for has_initial, has_residual in (
                    (False, False),
                    (True, False),
                    (True, True),
                ):
                    initial = (
                        torch.randn(
                            batch, 9, 10240, device="cuda", dtype=torch.bfloat16
                        )
                        if has_initial
                        else None
                    )
                    before = initial.clone() if initial is not None else None
                    residual = tensors[2] if has_residual else None
                    ref = reference_conv(*expected, initial, weight, residual)
                    with capture_kernel(conv, "_ple_conv") as ccalls:
                        out = conv.try_ple_conv(
                            *actual, initial, weight, 3, residual=residual
                        )
                    check = exact(out, ref)
                    check.update(
                        initial=has_initial, residual=has_residual, calls=ccalls
                    )
                    row["conv"].append(check)
                    self.assertEqual(len(ccalls), 1)
                    self.assertTrue(check["pass"], check)
                    if initial is not None:
                        self.assertTrue(torch.equal(initial, before))
                save("component_exact", record)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("component_exact", record)

    def test_projected_page_domains(self):
        record = {"metadata": metadata(), "cases": [], "status": "RUNNING"}
        try:
            for page in (64, 128, 192, 256, 320, 512, 768, 1024):
                rows = 2 * page + 1
                key, value, hidden, *gammas = self.tensors(1, rows, 44000 + page)
                scales = [
                    torch.rand((rows + page - 1) // page, device="cuda") + 0.1
                    for _ in range(2)
                ]
                sk, sv = [
                    s.repeat_interleave(page)[:rows].view(1, rows, 1) for s in scales
                ]
                expected = reference_gate(
                    (key.float() * sk).bfloat16(),
                    (value.float() * sv).bfloat16(),
                    hidden,
                    *gammas,
                )
                actual = gate.try_ple_gate(
                    key.float(),
                    value.float(),
                    hidden,
                    *gammas,
                    1e-6,
                    projection_scales=scales,
                    page_size=page,
                )
                checks = [exact(a, b) for a, b in zip(actual, expected)]
                record["cases"].append({"page": page, "rows": rows, "checks": checks})
                self.assertTrue(all(x["pass"] for x in checks), checks)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("component_projected_pages", record)

    def test_decode_chunk_graph_state_refresh(self):
        record = {
            "metadata": metadata(),
            "cases": [],
            "status": "RUNNING",
            "scope": "real decode_chunk with BF16 projections; FP8 integration separate",
        }
        try:
            for batch in (1, 2):
                torch.manual_seed(72000 + batch)
                kw = {"device": "cuda", "dtype": torch.bfloat16}
                layer = Qwen4ExpPLELayer(
                    SimpleNamespace(ngram_size=3, context_len=2),
                    torch.randn(10240, 2560, **kw) * 0.01,
                    torch.randn(2560, 2560, **kw) * 0.01,
                    torch.randn(10240, 1, 4, **kw) * 0.1,
                    *[torch.randn(10240, **kw) * 0.07 for _ in range(3)],
                    hc_mult=4,
                    hidden_size=2560,
                    conv_kernel_size=4,
                    norm_eps=1e-6,
                )
                hidden = torch.randn(batch, 4, 10240, **kw)
                emb = torch.randn(batch, 4, 2560, **kw)
                initial = torch.randn(batch, 9, 10240, **kw)
                tokens = torch.zeros(batch, 6, device="cuda", dtype=torch.int64)

                def call():
                    return layer.decode_chunk(hidden, tokens, initial, embeddings=emb)

                for _ in range(3):
                    call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output, candidates = call()
                for seed in (1, 2):
                    torch.manual_seed(73000 + seed)
                    hidden.normal_()
                    emb.normal_()
                    initial.normal_()
                    tokens.random_(0, 248320)
                    before = initial.clone()
                    key = F.linear(emb, layer.key_proj)
                    value = F.linear(emb, layer.value_proj)
                    g, n = reference_gate(
                        key,
                        value,
                        hidden,
                        layer.norm_key,
                        layer.norm_query,
                        layer.norm_conv,
                    )
                    expected = reference_conv(g, n, initial, layer.conv_weight)
                    graph.replay()
                    checks = {
                        "output": exact(output, expected),
                        "candidate": exact(candidates, n),
                        "initial_unchanged": exact(initial, before),
                        "accepted": [],
                    }
                    for accepted in range(5):
                        a = torch.cat((initial, candidates[:, :accepted]), 1)[:, -9:]
                        b = torch.cat((before, n[:, :accepted]), 1)[:, -9:]
                        checks["accepted"].append(exact(a, b))
                    record["cases"].append(
                        {"batch": batch, "seed": seed, "checks": checks}
                    )
                    self.assertTrue(
                        all(
                            checks[k]["pass"]
                            for k in ("output", "candidate", "initial_unchanged")
                        ),
                        checks,
                    )
                    self.assertTrue(all(x["pass"] for x in checks["accepted"]), checks)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("component_decode_graph", record)


if __name__ == "__main__":
    unittest.main()
