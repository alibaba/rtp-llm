"""Page-local native FP8 equivalence and resumable PLE integration contracts."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from ple_native_quant_adapter import (
    native_projection_adapter,
    provenance,
    scaled_fp8_per_tensor_quant,
)
from ple_sm90_test_utils import (
    capture_kernel,
    exact,
    metadata,
    reference_conv,
    reference_gate,
    save,
    sha,
)

from rtp_llm.models_py.modules.base.common.fp8_linear import project_fp8_pair
from rtp_llm.models_py.modules.qwen4_exp.ple import ple as owner
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_conv_triton as conv
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_gate_triton as gate
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_page_projection as projection

PAGES = (64, 128, 192, 256, 320, 512, 768, 1024)


def make_layer(seed=28000, quantized_weights=False):
    torch.manual_seed(seed)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    if quantized_weights:
        key, key_scale = scaled_fp8_per_tensor_quant(
            torch.randn(10240, 2560, **kw) * 0.02
        )
        value, value_scale = scaled_fp8_per_tensor_quant(
            torch.randn(2560, 2560, **kw) * 0.05
        )
    else:
        key = (torch.randn(10240, 2560, device="cuda") * 0.02).to(torch.float8_e4m3fn)
        value = (torch.randn(2560, 2560, device="cuda") * 0.02).to(torch.float8_e4m3fn)
        key_scale = torch.tensor([0.03], device="cuda")
        value_scale = torch.tensor([0.07], device="cuda")
    return owner.Qwen4ExpPLELayer(
        SimpleNamespace(ngram_size=3, context_len=2),
        key,
        value,
        torch.randn(10240, 1, 4, **kw) * 0.1,
        *[torch.randn(10240, **kw) * 0.07 for _ in range(3)],
        hc_mult=4,
        hidden_size=2560,
        conv_kernel_size=4,
        norm_eps=1e-6,
        projection_scales=[key_scale, value_scale]
    )


def fixture(rows, page, seed):
    torch.manual_seed(seed)
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    hidden = torch.randn(1, rows, 10240, **kw)
    embeddings = torch.randn(1, rows, 2560, **kw)
    # Adjacent pages must not accidentally share one amax domain.
    for number, start in enumerate(range(0, rows, page)):
        embeddings[:, start : start + page].mul_(2.0 ** (number % 7 - 3))
    if rows > page + 5:
        embeddings[:, page : min(2 * page, rows)].zero_()
    embeddings[0, -1, -1] = 4096.0
    history = torch.zeros(1, rows + 2, device="cuda", dtype=torch.int64)
    initial = torch.randn(1, 9, 10240, **kw)
    return hidden, embeddings, history, initial


def reference_pages(layer, hidden, embeddings, page, initial, residual=False):
    """Frozen 366 page loop; never calls a candidate gate/conv/page projector."""
    outputs, snapshots = [], []
    state = initial
    for start in range(0, hidden.shape[1], page):
        end = min(start + page, hidden.shape[1])
        key, value = project_fp8_pair(
            embeddings[:, start:end].contiguous(),
            layer.key_projection,
            layer.value_projection,
        )
        gated, normed = reference_gate(
            key,
            value,
            hidden[:, start:end],
            layer.norm_key,
            layer.norm_query,
            layer.norm_conv,
        )
        outputs.append(reference_conv(gated, normed, state, layer.conv_weight))
        if state is None:
            padding = normed.new_zeros(1, max(9 - normed.shape[1], 0), 10240)
            state = torch.cat((padding, normed), 1)[:, -9:]
        else:
            state = torch.cat((state, normed), 1)[:, -9:]
        snapshots.append(state)
    output = torch.cat(outputs, 1)
    return (hidden + output if residual else output), snapshots


@contextlib.contextmanager
def capture_scaled_mm():
    original, calls = torch._scaled_mm, []

    def wrapped(a, b, *args, **kwargs):
        calls.append(
            {
                "a": list(a.shape),
                "b": list(b.shape),
                "scale_a": list(kwargs["scale_a"].shape),
                "scale_b": list(kwargs["scale_b"].shape),
            }
        )
        return original(a, b, *args, **kwargs)

    with patch.object(torch, "_scaled_mm", wrapped):
        yield calls


class PleSM90PagesTest(unittest.TestCase):
    def setUp(self):
        self.native_adapter = native_projection_adapter()
        self.native_adapter.__enter__()
        self.addCleanup(self.native_adapter.__exit__, None, None, None)
        save("native_binding_provenance", provenance())

    def test_realistic_weight_domains(self):
        record = {"metadata": metadata(), "cases": [], "status": "RUNNING"}
        try:
            layer = make_layer(28001, quantized_weights=True)
            record["weights"] = {
                "key_amax_fp8": float(layer.key_proj.float().abs().max()),
                "value_amax_fp8": float(layer.value_proj.float().abs().max()),
                "key_scale": float(layer.key_projection.scale),
                "value_scale": float(layer.value_projection.scale),
                "origin": "synthetic BF16 matrices quantized with original native per-tensor CUDA op",
            }
            for page, rows in (
                (64, 129),
                (128, 129),
                (192, 385),
                (320, 641),
                (64, 1537),
            ):
                hidden, emb, history, initial = fixture(rows, page, 38000 + page + rows)
                keys, values = [], []
                for start in range(0, rows, page):
                    k, v = project_fp8_pair(
                        emb[:, start : start + page].contiguous(),
                        layer.key_projection,
                        layer.value_projection,
                    )
                    keys.append(k)
                    values.append(v)
                result = projection.project_pages_pair(
                    emb, layer.key_projection, layer.value_projection, page
                )
                self.assertIsNotNone(result)
                key, value, _ = result
                row = {
                    "page": page,
                    "rows": rows,
                    "key": exact(key, torch.cat(keys, 1)),
                    "value": exact(value, torch.cat(values, 1)),
                }
                record["cases"].append(row)
                self.assertTrue(row["key"]["pass"] and row["value"]["pass"], row)
                expected, states = reference_pages(
                    layer, hidden, emb, page, initial, True
                )
                actual, candidates = layer.prefill_page_snapshots(
                    hidden, history, page, initial, embeddings=emb, add_residual=True
                )
                row["output"] = exact(actual, expected)
                row["snapshots"] = [exact(a, b) for a, b in zip(candidates, states)]
                self.assertEqual(len(candidates), len(states))
                self.assertTrue(
                    row["output"]["pass"] and all(x["pass"] for x in row["snapshots"]),
                    row,
                )
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("realistic_weight_exact", record)

    def test_legacy_positive_page_fallback(self):
        record = {"metadata": metadata(), "cases": [], "status": "RUNNING"}
        try:
            layer = make_layer(29001, quantized_weights=True)
            for page in (32, 96):
                hidden, emb, history, initial = fixture(
                    2 * page + 7, page, 45000 + page
                )
                self.assertIsNone(
                    projection.project_pages_pair(
                        emb, layer.key_projection, layer.value_projection, page
                    )
                )
                for prefix in (False, True):
                    state = initial if prefix else None
                    ref, refs = reference_pages(layer, hidden, emb, page, state, True)
                    actual, snapshots = layer.prefill_page_snapshots(
                        hidden, history, page, state, embeddings=emb, add_residual=True
                    )
                    checks = {
                        "page": page,
                        "prefix": prefix,
                        "output": exact(actual, ref),
                        "snapshots": [exact(a, b) for a, b in zip(snapshots, refs)],
                    }
                    record["cases"].append(checks)
                    self.assertEqual(len(snapshots), len(refs))
                    self.assertTrue(
                        checks["output"]["pass"]
                        and all(x["pass"] for x in checks["snapshots"]),
                        checks,
                    )
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("legacy_page_fallback_exact", record)

    def test_native_quant_bytes_and_scales(self):
        record = {
            "metadata": metadata(),
            "cases": [],
            "status": "RUNNING",
            "source": {"projection": sha(projection.__file__)},
        }
        try:
            ks = torch.tensor([0.03], device="cuda")
            vs = torch.tensor([0.07], device="cuda")
            for page in PAGES + (1088, 32768):
                for rows in (
                    129,
                    min(page - 1, 513),
                    min(page + 1, 1089),
                    min(2 * page + 3, 2051),
                ):
                    _, x, _, _ = fixture(rows, page, 29000 + rows + page)
                    quantized, scales = projection.quantize_pages(
                        x[0], ks, vs, page_size=page
                    )
                    row_q, row_scales = projection.quantize_pages(
                        x[0], ks, vs, page_size=page, row_scales=True
                    )
                    row = {"page": page, "rows": rows, "checks": []}
                    record["cases"].append(row)
                    for i, start in enumerate(range(0, rows, page)):
                        end = min(start + page, rows)
                        native_q, native_s = scaled_fp8_per_tensor_quant(
                            x[0, start:end].contiguous()
                        )
                        checks = {
                            "bytes": exact(
                                quantized[start:end].view(torch.uint8),
                                native_q.view(torch.uint8),
                            ),
                            "raw_scale": exact(scales[0][i : i + 1], native_s),
                            "key_scale": exact(scales[1][i : i + 1], native_s * ks),
                            "value_scale": exact(scales[2][i : i + 1], native_s * vs),
                            "row_bytes": exact(
                                row_q[start:end].view(torch.uint8),
                                native_q.view(torch.uint8),
                            ),
                            "row_key_scale": exact(
                                row_scales[1][start:end],
                                (native_s * ks).expand(end - start, 1),
                            ),
                            "row_value_scale": exact(
                                row_scales[2][start:end],
                                (native_s * vs).expand(end - start, 1),
                            ),
                        }
                        padded_end = min(start + page, row_q.shape[0])
                        checks["padded_key_scale"] = exact(
                            row_scales[1][start:padded_end],
                            (native_s * ks).expand(padded_end - start, 1),
                        )
                        checks["padded_value_scale"] = exact(
                            row_scales[2][start:padded_end],
                            (native_s * vs).expand(padded_end - start, 1),
                        )
                        row["checks"].append(checks)
                        self.assertTrue(all(c["pass"] for c in checks.values()), checks)
                    self.assertEqual(
                        int(
                            torch.count_nonzero(
                                quantized[rows:].view(torch.uint8)
                            ).item()
                        ),
                        0,
                    )
                    self.assertEqual(
                        int(torch.count_nonzero(row_q[rows:].view(torch.uint8)).item()),
                        0,
                    )
                    save("native_quant_exact", record)
            for value in (0.0, 1e-38, 1e-37, 1e-36, 5.25e-36, 5.27e-36, 1e-35, 1e-20):
                x = torch.full((193, 2560), value, device="cuda", dtype=torch.bfloat16)
                q, s = projection.quantize_pages(x, ks, vs, page_size=192)
                nq, ns = scaled_fp8_per_tensor_quant(x[:192].contiguous())
                checks = {
                    "bytes": exact(q[:192].view(torch.uint8), nq.view(torch.uint8)),
                    "scale": exact(s[0][:1], ns),
                }
                record["cases"].append({"tiny_value": value, "checks": checks})
                self.assertTrue(all(c["pass"] for c in checks.values()), checks)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("native_quant_exact", record)

    def test_batched_projection_exact(self):
        record = {"metadata": metadata(), "cases": [], "status": "RUNNING"}
        try:
            layer = make_layer()
            for page in PAGES:
                for rows in (max(128, page), 2 * page + 1):
                    _, x, _, _ = fixture(rows, page, 31000 + rows + page)
                    keys, values = [], []
                    for start in range(0, rows, page):
                        k, v = project_fp8_pair(
                            x[:, start : start + page].contiguous(),
                            layer.key_projection,
                            layer.value_projection,
                        )
                        keys.append(k)
                        values.append(v)
                    with capture_scaled_mm() as calls:
                        result = projection.project_pages_pair(
                            x, layer.key_projection, layer.value_projection, page
                        )
                    self.assertIsNotNone(result)
                    key, value, scale = result
                    row = {
                        "page": page,
                        "rows": rows,
                        "calls": calls,
                        "key": exact(key, torch.cat(keys, 1)),
                        "value": exact(value, torch.cat(values, 1)),
                    }
                    record["cases"].append(row)
                    self.assertEqual(len(calls), 2)
                    self.assertIsNone(scale)
                    self.assertTrue(row["key"]["pass"] and row["value"]["pass"], row)
                    save("batched_projection_exact", record)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("batched_projection_exact", record)

    def test_prefill_outputs_every_snapshot(self):
        record = {
            "metadata": metadata(),
            "cases": [],
            "status": "RUNNING",
            "source": {
                "owner": sha(owner.__file__),
                "gate": sha(gate.__file__),
                "conv": sha(conv.__file__),
                "projection": sha(projection.__file__),
            },
        }
        try:
            layer = make_layer()
            cases = [(p, s) for p in PAGES for s in (7, p - 1, p, p + 1, 2 * p + 3)]
            cases += [(64, s) for s in (5120, 10240, 15360, 20480)]
            for page, rows in cases:
                hidden, emb, history, initial = fixture(rows, page, 33000 + rows + page)
                for prefix, residual in (
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ):
                    state = initial if prefix else None
                    before = state.clone() if state is not None else None
                    ref, refs = reference_pages(
                        layer, hidden, emb, page, state, residual
                    )
                    with capture_scaled_mm() as calls, capture_kernel(
                        gate, "_ple_gate"
                    ) as gcalls, capture_kernel(conv, "_ple_conv") as ccalls:
                        out, states = layer.prefill_page_snapshots(
                            hidden,
                            history,
                            page,
                            state,
                            embeddings=emb,
                            add_residual=residual,
                        )
                    row = {
                        "page": page,
                        "rows": rows,
                        "prefix": prefix,
                        "residual": residual,
                        "output": exact(out, ref),
                        "snapshots": [exact(a, b) for a, b in zip(states, refs)],
                        "gemm_calls": calls,
                        "gate_calls": gcalls,
                        "conv_calls": ccalls,
                    }
                    record["cases"].append(row)
                    self.assertEqual(len(states), len(refs))
                    self.assertEqual(len(gcalls), 1)
                    self.assertEqual(len(ccalls), 1)
                    if rows >= 128:
                        self.assertEqual(len(calls), 2)
                    self.assertTrue(row["output"]["pass"], row["output"])
                    self.assertTrue(
                        all(c["pass"] for c in row["snapshots"]), row["snapshots"]
                    )
                    if before is not None:
                        self.assertTrue(torch.equal(state, before))
                    save("prefill_pages_exact", record)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("prefill_pages_exact", record)

    def test_fp8_decode_graph_acceptance(self):
        record = {
            "metadata": metadata(),
            "cases": [],
            "status": "RUNNING",
            "scope": "FP8 decode_chunk candidate outputs and all accepted-prefix states; scheduler not included",
        }
        try:
            layer = make_layer()
            hidden, emb, history, initial = fixture(4, 64, 41000)

            def call():
                return layer.decode_chunk(hidden, history, initial, embeddings=emb)

            for _ in range(3):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, candidates = call()
            for seed in (41001, 41002, 41003):
                torch.manual_seed(seed)
                hidden.normal_()
                emb.normal_()
                initial.normal_()
                history.random_(0, 248320)
                before = initial.clone()
                k, v = project_fp8_pair(
                    emb, layer.key_projection, layer.value_projection
                )
                g, n = reference_gate(
                    k, v, hidden, layer.norm_key, layer.norm_query, layer.norm_conv
                )
                ref = reference_conv(g, n, initial, layer.conv_weight)
                graph.replay()
                row = {
                    "seed": seed,
                    "output": exact(out, ref),
                    "candidate": exact(candidates, n),
                    "initial": exact(initial, before),
                    "accepted": [],
                }
                for accepted in range(5):
                    a = torch.cat((initial, candidates[:, :accepted]), 1)[:, -9:]
                    b = torch.cat((before, n[:, :accepted]), 1)[:, -9:]
                    row["accepted"].append(exact(a, b))
                record["cases"].append(row)
                self.assertTrue(
                    all(row[k]["pass"] for k in ("output", "candidate", "initial")), row
                )
                self.assertTrue(all(c["pass"] for c in row["accepted"]), row)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("fp8_decode_graph_exact", record)


if __name__ == "__main__":
    unittest.main()
