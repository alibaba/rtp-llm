"""Manual MI308 library-only LM-head probes; private inputs come from env.

MKT_LM_HEAD_MODEL_DIR: the original tied-FP32 checkpoint plus tokenizer/manifest.
MKT_LM_HEAD_OUTPUT_DIR: aggregate diagnostics, never raw weights/hidden/logits.
Synthetic H is deterministic BF16, not a captured business hidden-state fixture.
No acceptance result here establishes complete-model ranking/recall quality.
"""

import hashlib
import json
import os
import statistics
import struct
import unittest
from pathlib import Path

import numpy as np
import torch

# Fixed before the first GPU measurement. These are LM-head diagnostic gates,
# not the attention kernel's much looser 0.005/0.03 tolerances.
ATOL = 2e-4
RTOL = 2e-5
L2_LIMIT = 2e-5
LOG_SCORE_ATOL = 1e-3
SEED = 20261029
EXPECTED_N = 65537
EXPECTED_K = 1024


def tensor_hash(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def emit(name, record):
    directory = os.environ.get("MKT_LM_HEAD_OUTPUT_DIR")
    if not directory:
        raise RuntimeError("MKT_LM_HEAD_OUTPUT_DIR is required")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["test_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    record["thresholds"] = dict(
        atol=ATOL, rtol=RTOL, relative_l2=L2_LIMIT, log_score_max_abs=LOG_SCORE_ATOL
    )
    record["fixture_kind"] = "real_FP32_weight__deterministic_synthetic_BF16_H"
    (path / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"metric": name, "path": str(path / (name + ".json"))}), flush=True
    )


def load_selected_weight():
    directory = os.environ.get("MKT_LM_HEAD_MODEL_DIR")
    if not directory:
        raise RuntimeError(
            "MKT_LM_HEAD_MODEL_DIR is required; no random-weight fallback"
        )
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    if config.get("tie_word_embeddings") is not True:
        raise ValueError("this narrow fixture loader requires tied embeddings")
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    vocab = dict(tokenizer["model"]["vocab"])
    vocab.update(
        {entry["content"]: entry["id"] for entry in tokenizer.get("added_tokens", [])}
    )
    groups = json.loads((root / "output_tokens.json").read_text())
    if not (
        isinstance(groups, list)
        and len(groups) == 2
        and all(isinstance(group, list) and len(group) == 32768 for group in groups)
    ):
        raise ValueError("expected the real two organizational SID groups of 32768")
    ids = sorted(
        {vocab[token] for group in groups for token in group} | {config["eos_token_id"]}
    )
    if len(ids) != EXPECTED_N:
        raise ValueError("expected 65537 selected rows including EOS")
    checkpoint = root / "model.safetensors"
    with checkpoint.open("rb") as reader:
        header_bytes = struct.unpack("<Q", reader.read(8))[0]
        header = json.loads(reader.read(header_bytes))
    desc = header["model.embed_tokens.weight"]
    if desc["dtype"] != "F32" or desc["shape"][1] != EXPECTED_K:
        raise ValueError("expected original F32 embedding with hidden size 1024")
    if max(ids) >= desc["shape"][0] or min(ids) < 0:
        raise ValueError("selected vocabulary index outside checkpoint tensor")
    mapped = np.memmap(
        checkpoint,
        mode="r",
        dtype="<f4",
        offset=8 + header_bytes + desc["data_offsets"][0],
        shape=tuple(desc["shape"]),
    )
    weight = torch.from_numpy(np.array(mapped[ids], copy=True))
    if not torch.isfinite(weight).all():
        raise ValueError("non-finite checkpoint weights")
    return weight, {
        "source_tensor": "model.embed_tokens.weight",
        "source_dtype": "F32",
        "source_shape": desc["shape"],
        "selected_shape": list(weight.shape),
        "selected_weight_sha256": tensor_hash(weight),
        "selected_ids_sha256": hashlib.sha256(
            np.asarray(ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (root / "output_tokens.json").read_bytes()
        ).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (root / "tokenizer.json").read_bytes()
        ).hexdigest(),
    }


def split_weight(weight):
    """CPU once, FP32 residual arithmetic; all three GEMM operands are BF16."""
    high = weight.to(torch.bfloat16)
    residual = weight - high.float()
    low = residual.to(torch.bfloat16)
    tail = (residual - low.float()).to(torch.bfloat16)
    two = high.float() + low.float()
    three = two + tail.float()
    difference = three - weight
    # Float64 sum checks the represented real values as well as FP32 reconstruction.
    represented_error = high.double() + low.double() + tail.double() - weight.double()
    stats = {
        "three_term_fp32_reconstruction_byte_exact": torch.equal(
            three.view(torch.int32), weight.view(torch.int32)
        ),
        "three_term_fp32_reconstruction_max_abs": difference.abs().max().item(),
        "three_term_real_value_sum_max_abs": represented_error.abs().max().item(),
        "two_term_reconstruction_relative_l2": (
            (two.double() - weight.double()).norm() / weight.double().norm()
        ).item(),
        "two_term_reconstruction_max_abs": (two - weight).abs().max().item(),
        "one_term_reconstruction_relative_l2": (
            (high.double() - weight.double()).norm() / weight.double().norm()
        ).item(),
        "bf16_exact_elements": int((high.float() == weight).sum().item()),
        "elements": weight.numel(),
        "component_sha256": [tensor_hash(part) for part in (high, low, tail)],
    }
    return (high, low, tail), stats


def make_hidden(rows, scale=1.0, seed=SEED):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn((rows, EXPECTED_K), generator=generator) * scale).to(
        torch.bfloat16
    )


class LibraryArm:
    """The timed path includes H promotion for FP32, and FP32 component additions."""

    def __init__(self, hidden, weight, components, terms):
        self.hidden = hidden
        self.weight = weight
        self.components = components
        self.terms = terms
        self.output = torch.empty(
            (hidden.shape[0], weight.shape[0]),
            dtype=torch.float32,
            device=hidden.device,
        )
        self.partial = torch.empty_like(self.output) if terms > 1 else None

    def __call__(self):
        if self.terms == 0:
            torch.mm(self.hidden.float(), self.weight.t(), out=self.output)
        else:
            torch.mm(
                self.hidden,
                self.components[0].t(),
                out_dtype=torch.float32,
                out=self.output,
            )
            for component in self.components[1 : self.terms]:
                torch.mm(
                    self.hidden,
                    component.t(),
                    out_dtype=torch.float32,
                    out=self.partial,
                )
                self.output.add_(self.partial)
        return self.output


def comparison(actual, expected):
    actual64 = actual.double()
    finite = bool(torch.isfinite(actual64).all() and torch.isfinite(expected).all())
    error = actual64 - expected
    mismatch = (~torch.isfinite(actual64)) | (
        error.abs() > ATOL + RTOL * expected.abs()
    )
    relative_l2 = (error.norm() / expected.norm().clamp_min(1e-300)).item()
    return {
        "finite": finite,
        "max_abs": error.abs().max().item(),
        "relative_l2": relative_l2,
        "mismatch_count": int(mismatch.sum().item()),
        "failed_coordinates_first_64": mismatch.nonzero()[:64].cpu().tolist(),
        "gate_pass": finite and not bool(mismatch.any()) and relative_l2 <= L2_LIMIT,
    }


def score_report(actual, expected):
    # Same complete vocabulary denominator for both arms. No per-position SID mask.
    reference_logp = torch.log_softmax(expected, dim=-1)
    actual_logp = torch.log_softmax(actual, dim=-1).double()
    error = actual_logp - reference_logp
    k = min(1000, expected.numel() - 1)
    # Deterministic synthetic parent scores exercise cross-row beam ranking.
    parents = (
        -torch.arange(
            expected.shape[0], device=expected.device, dtype=torch.float64
        ).remainder(500)
        / 500.0
    )
    ref_scores = reference_logp + parents[:, None]
    candidate_scores = actual_logp + parents[:, None]
    ref_values, ref_ids = ref_scores.flatten().topk(k + 1)
    _, ids = candidate_scores.flatten().topk(k)
    ref_ids = ref_ids[:k]
    overlap = torch.isin(ids, ref_ids).sum().item()
    max_abs = error.abs().max().item()
    margin = (ref_values[k - 1] - ref_values[k]).item()
    return {
        "scope": "synthetic_parent_scores_full_vocabulary_top1000_not_business_quality",
        "log_softmax_max_abs": max_abs,
        "score_gate_pass": max_abs <= LOG_SCORE_ATOL,
        "top1_same": bool(ids[0] == ref_ids[0]),
        "top1000_order_exact": torch.equal(ids, ref_ids),
        "top1000_overlap": int(overlap),
        "top1000_count": k,
        "reference_boundary_margin": margin,
        "boundary_margin_exceeds_twice_observed_score_error": margin > 2 * max_abs,
    }


class LmHeadFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError(
                "manual probe requires a prechecked ROCm GPU; no skip-as-pass"
            )
        torch.set_num_threads(4)
        cls.old_precision = torch.get_float32_matmul_precision()
        cls.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        weight, cls.weight_record = load_selected_weight()
        components, cls.split_record = split_weight(weight)
        cls.weight = weight.cuda()
        cls.components = tuple(part.cuda() for part in components)
        cls.device_record = {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "noncontiguous_weight_T": not cls.weight.t().is_contiguous(),
            "weight_T_stride": list(cls.weight.t().stride()),
            "library": "torch.mm.dtype_out; no custom GEMM",
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_float32_matmul_precision(cls.old_precision)
        torch.backends.cuda.matmul.allow_tf32 = cls.old_tf32
        del cls.weight, cls.components


class LmHeadBF16x3ProbeTest(LmHeadFixture):
    def test_api_accuracy_and_graph(self):
        records = []
        for rows in (1, 500):
            hidden = make_hidden(rows).cuda()
            arms = {
                name: LibraryArm(hidden, self.weight, self.components, terms)
                for name, terms in (("fp32", 0), ("bf16x2", 2), ("bf16x3", 3))
            }
            try:
                outputs = {name: arm().clone() for name, arm in arms.items()}
            except Exception as error:
                emit(
                    "probe_api_failure",
                    {
                        "status": "FAIL",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "rows": rows,
                        "weight": self.weight_record,
                        "split": self.split_record,
                        "device": self.device_record,
                    },
                )
                raise
            reference = hidden.double() @ self.weight.double().t()
            cases = {
                name: {
                    "logits": comparison(value, reference),
                    "scores": score_report(value, reference),
                }
                for name, value in outputs.items()
            }
            # Two-term numerical status is diagnostic. Three-term is the candidate gate.
            arm = arms["bf16x3"]
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    arm()
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                arm()
            graph_records = []
            for scale, seed in ((0.125, SEED + 1), (8.0, SEED + 2)):
                hidden.copy_(make_hidden(rows, scale, seed).to(hidden.device))
                eager = arm().clone()
                graph.replay()
                replay = arm.output.clone()
                expected = hidden.double() @ self.weight.double().t()
                graph_records.append(
                    {
                        "scale": scale,
                        "seed": seed,
                        "own_eager_byte_exact": torch.equal(
                            eager.view(torch.int32), replay.view(torch.int32)
                        ),
                        "logits": comparison(replay, expected),
                        "scores": score_report(replay, expected),
                        "input_sha256": tensor_hash(hidden),
                    }
                )
            record = {"rows": rows, "clean": cases, "graph": graph_records}
            records.append(record)
            emit(
                "probe_m%d" % rows,
                {
                    "weight": self.weight_record,
                    "split": self.split_record,
                    "device": self.device_record,
                    **record,
                },
            )
        # Emit both M values before any numerical assertion, preserving baseline failures.
        self.assertTrue(self.split_record["three_term_fp32_reconstruction_byte_exact"])
        self.assertEqual(self.split_record["three_term_real_value_sum_max_abs"], 0.0)
        for record in records:
            self.assertTrue(
                record["clean"]["bf16x3"]["logits"]["gate_pass"], record["rows"]
            )
            self.assertTrue(
                record["clean"]["bf16x3"]["scores"]["score_gate_pass"], record["rows"]
            )
            for replay in record["graph"]:
                self.assertTrue(replay["own_eager_byte_exact"])
                self.assertTrue(replay["logits"]["gate_pass"])
                self.assertTrue(replay["scores"]["score_gate_pass"])


def capture_arm(arm):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            arm()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        arm()
    return graph


def balanced_orders(names):
    rotations = [names[index:] + names[:index] for index in range(len(names))]
    orders = rotations + [tuple(reversed(order)) for order in rotations]
    for left in names:
        for right in names:
            if left != right:
                assert sum(
                    order.index(left) < order.index(right) for order in orders
                ) == len(names)
        for position in range(len(names)):
            assert sum(order[position] == left for order in orders) == 2
    return orders


class LmHeadBF16x3PerformanceTest(LmHeadFixture):
    def test_complete_library_paths(self):
        import triton.testing

        names = ("fp32", "bf16x2", "bf16x3")
        orders = balanced_orders(names)
        for rows in (1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000):
            hidden = make_hidden(rows).cuda()
            arms = {
                name: LibraryArm(hidden, self.weight, self.components, terms)
                for name, terms in zip(names, (0, 2, 3))
            }
            selected = sorted({0, rows // 2, rows - 1})
            # Preflight is explicitly selected-row FP64, not the separate full-row gate.
            reference = hidden[selected].double() @ self.weight.double().t()
            preflight = {}
            graphs = {}
            for name, arm in arms.items():
                eager = arm().clone()
                graphs[name] = capture_arm(arm)
                graphs[name].replay()
                replay = arm.output.clone()
                preflight[name] = {
                    "sampled_fp64": comparison(replay[selected], reference),
                    "sampled_rows": selected,
                    "all_rows_finite": bool(torch.isfinite(replay).all()),
                    "own_graph_eager_byte_exact": torch.equal(
                        eager.view(torch.int32), replay.view(torch.int32)
                    ),
                }
                del eager, replay
            emit(
                "performance_preflight_m%d" % rows,
                {
                    "preflight": preflight,
                    "weight": self.weight_record,
                    "split": self.split_record,
                    "device": self.device_record,
                },
            )
            self.assertTrue(preflight["bf16x3"]["sampled_fp64"]["gate_pass"])
            for name in names:
                self.assertTrue(preflight[name]["all_rows_finite"])
                self.assertTrue(preflight[name]["own_graph_eager_byte_exact"])
            torch.cuda.synchronize()
            modes = ("eager_cold", "graph_cold", "graph_hot")
            raw = {name: {mode: [] for mode in modes} for name in names}
            for round_index, order in enumerate(orders):
                for name in order:
                    # All copying/comparison/printing is outside each timed closure.
                    values = triton.testing.do_bench(
                        arms[name], warmup=10, rep=30, return_mode="all"
                    )
                    raw[name]["eager_cold"].append(list(values))
                    values = triton.testing.do_bench(
                        graphs[name].replay, warmup=10, rep=30, return_mode="all"
                    )
                    raw[name]["graph_cold"].append(list(values))
                    values = triton.testing.do_bench_cudagraph(
                        arms[name], rep=20, return_mode="all"
                    )
                    raw[name]["graph_hot"].append(list(values))
            summary = {
                name: {
                    mode: {
                        "round_medians_ms": [
                            statistics.median(values) for values in rounds
                        ],
                        "mean_of_round_medians_ms": statistics.mean(
                            statistics.median(values) for values in rounds
                        ),
                        "raw_sample_count": sum(len(values) for values in rounds),
                    }
                    for mode, rounds in by_mode.items()
                }
                for name, by_mode in raw.items()
            }
            ratios = {
                name: {
                    mode: summary["fp32"][mode]["mean_of_round_medians_ms"]
                    / summary[name][mode]["mean_of_round_medians_ms"]
                    for mode in modes
                }
                for name in ("bf16x2", "bf16x3")
            }
            emit(
                "performance_m%d" % rows,
                {
                    "rows": rows,
                    "input_sha256": tensor_hash(hidden),
                    "preflight": preflight,
                    "weight": self.weight_record,
                    "split": self.split_record,
                    "device": self.device_record,
                    "orders": orders,
                    "raw_samples_ms": raw,
                    "summary": summary,
                    "speedup_vs_fp32": ratios,
                    "timer": "triton.testing.do_bench / do_bench_cudagraph; GPU events",
                    "triton_testing_source_sha256": hashlib.sha256(
                        Path(triton.testing.__file__).read_bytes()
                    ).hexdigest(),
                    "timed_scope": "whole LM head: baseline H cast; all component GEMMs and FP32 additions",
                    "excluded": "weight loading/splitting and all accuracy/Graph validation",
                    "cache_modes": {
                        "eager_cold": "do_bench clears driver benchmark cache before each sample; host launches included",
                        "graph_cold": "same cache clear before replay of one whole LM head",
                        "graph_hot": "do_bench_cudagraph unrolled whole-LM-head calls; sample is amortized call mean",
                    },
                    "interpretation": "attention/model/e2e/QPS NOT_MEASURED; M1 is diagnostic, not proposed production route",
                },
            )
            del graphs, arms, hidden, reference, arm


class LmHeadBF16x3AccuracyTest(LmHeadFixture):
    def test_large_rows_multiscale_fp64_and_graph(self):
        records = []
        for rows in (2000, 4000):
            hidden = make_hidden(rows).cuda()
            arms = {
                name: LibraryArm(hidden, self.weight, self.components, terms)
                for name, terms in (("fp32", 0), ("bf16x2", 2), ("bf16x3", 3))
            }
            graph = capture_arm(arms["bf16x3"])
            for scale in (0.125, 1.0, 8.0):
                hidden.copy_(make_hidden(rows, scale, SEED + 3).to(hidden.device))
                outputs = {name: arm().clone() for name, arm in arms.items()}
                graph.replay()
                replay = arms["bf16x3"].output.clone()
                graph_exact = torch.equal(
                    outputs["bf16x3"].view(torch.int32), replay.view(torch.int32)
                )
                # Full independent FP64 GEMM, chunked only along output rows.
                # Chunking the row dimension does not change a row's K reduction.
                weight64 = self.weight.double()
                reference = torch.empty(
                    (rows, EXPECTED_N), device=hidden.device, dtype=torch.float64
                )
                for begin in range(0, rows, 128):
                    end = min(begin + 128, rows)
                    torch.mm(
                        hidden[begin:end].double(),
                        weight64.t(),
                        out=reference[begin:end],
                    )
                del weight64
                result = {
                    name: {
                        "logits": comparison(output, reference),
                        "scores": score_report(output, reference),
                    }
                    for name, output in outputs.items()
                }
                # Triangle decomposition is descriptive: independent FP64 remains the gate.
                baseline_error = outputs["fp32"].double() - reference
                candidate_error = outputs["bf16x3"].double() - reference
                delta = outputs["bf16x3"].double() - outputs["fp32"].double()
                triangle = {
                    "candidate_minus_baseline_l2": delta.norm().item(),
                    "baseline_minus_fp64_l2": baseline_error.norm().item(),
                    "candidate_minus_fp64_l2": candidate_error.norm().item(),
                    "residual_identity_max_abs": (
                        candidate_error - (delta + baseline_error)
                    )
                    .abs()
                    .max()
                    .item(),
                }
                record = {
                    "rows": rows,
                    "scale": scale,
                    "seed": SEED + 3,
                    "own_graph_eager_byte_exact": graph_exact,
                    "comparisons": result,
                    "triangle": triangle,
                }
                records.append(record)
                emit(
                    "accuracy_m%d_scale%s" % (rows, str(scale).replace(".", "p")),
                    {
                        "weight": self.weight_record,
                        "split": self.split_record,
                        "device": self.device_record,
                        **record,
                    },
                )
                del outputs, reference, replay, baseline_error, candidate_error, delta
            del graph, arms, hidden
        # A failing original or two-term diagnostic does not hide three-term evidence.
        for record in records:
            self.assertTrue(record["own_graph_eager_byte_exact"])
            self.assertTrue(record["comparisons"]["bf16x3"]["logits"]["gate_pass"])
            self.assertTrue(
                record["comparisons"]["bf16x3"]["scores"]["score_gate_pass"]
            )


class PaddedLibraryArm:
    """Three library GEMMs, FP32 adds, then remove padding before softmax."""

    def __init__(self, hidden, components):
        self.hidden = hidden
        self.components = components
        self.padded_output = torch.empty(
            (hidden.shape[0], components[0].shape[0]),
            dtype=torch.float32,
            device=hidden.device,
        )
        self.partial = torch.empty_like(self.padded_output)
        self.output = None

    def __call__(self):
        torch.mm(
            self.hidden,
            self.components[0].t(),
            out_dtype=torch.float32,
            out=self.padded_output,
        )
        for component in self.components[1:]:
            torch.mm(
                self.hidden, component.t(), out_dtype=torch.float32, out=self.partial
            )
            self.padded_output.add_(self.partial)
        # This allocation/copy is part of the timed path. Padded zero logits must
        # never enter the beam softmax denominator.
        self.output = self.padded_output.narrow(1, 0, EXPECTED_N).contiguous()
        return self.output


class LmHeadBF16x3PaddingTest(LmHeadFixture):
    def test_padding_complete_path(self):
        import triton.testing

        alignments = (8, 64, 128, 256)
        padded_parts = {}
        for alignment in alignments:
            padded_n = ((EXPECTED_N + alignment - 1) // alignment) * alignment
            parts = tuple(
                torch.zeros(
                    (padded_n, EXPECTED_K),
                    dtype=torch.bfloat16,
                    device=self.weight.device,
                )
                for _ in range(3)
            )
            for padded, original in zip(parts, self.components):
                padded[:EXPECTED_N].copy_(original)
                self.assertTrue(torch.equal(padded[:EXPECTED_N], original))
                self.assertEqual(torch.count_nonzero(padded[EXPECTED_N:]).item(), 0)
            padded_parts[alignment] = parts
        names = ("fp32", "bf16x3") + tuple("pad%d" % value for value in alignments)
        orders = balanced_orders(names)
        all_controls = []
        for rows in (500, 2000, 4000):
            hidden = make_hidden(rows).cuda()
            arms = {
                "fp32": LibraryArm(hidden, self.weight, self.components, 0),
                "bf16x3": LibraryArm(hidden, self.weight, self.components, 3),
            }
            arms.update(
                {
                    "pad%d" % alignment: PaddedLibraryArm(hidden, parts)
                    for alignment, parts in padded_parts.items()
                }
            )
            selected = (
                list(range(rows))
                if rows == 500
                else sorted({0, 1, rows // 2, rows - 1})
            )
            reference = hidden[selected].double() @ self.weight.double().t()
            graphs, graph_outputs, preflight = {}, {}, {}
            for name, arm in arms.items():
                eager = arm().clone()
                graphs[name] = capture_arm(arm)
                # Padded arms allocate their contiguous crop each eager call.
                # Hold the capture's actual output before subsequent calls replace
                # arm.output; otherwise an eager tensor could falsely test replay.
                graph_outputs[name] = arm.output
                graphs[name].replay()
                replay = graph_outputs[name].clone()
                preflight[name] = {
                    "oracle_scope": "all_rows" if rows == 500 else "selected_rows",
                    "oracle_row_count": len(selected),
                    "selected_rows": selected if rows != 500 else None,
                    "fp64": comparison(replay[selected], reference),
                    "scores": score_report(replay[selected], reference),
                    "all_rows_finite": bool(torch.isfinite(replay).all()),
                    "output_shape": list(replay.shape),
                    "output_contiguous": replay.is_contiguous(),
                    "own_graph_eager_byte_exact": torch.equal(
                        eager.view(torch.int32), replay.view(torch.int32)
                    ),
                }
                del eager, replay
            # Alter live H in place; retained graph output is compared against the
            # corresponding arm's own eager, not against its last Python attribute.
            hidden.copy_(make_hidden(rows, 8.0, SEED + 4).to(hidden.device))
            changed_reference = hidden[selected].double() @ self.weight.double().t()
            for name, arm in arms.items():
                eager = arm().clone()
                graphs[name].replay()
                replay = graph_outputs[name].clone()
                preflight[name]["changed_H_scale8"] = {
                    "own_graph_eager_byte_exact": torch.equal(
                        eager.view(torch.int32), replay.view(torch.int32)
                    ),
                    "fp64": comparison(replay[selected], changed_reference),
                    "scores": score_report(replay[selected], changed_reference),
                    "all_rows_finite": bool(torch.isfinite(replay).all()),
                }
                del eager, replay
            hidden.copy_(make_hidden(rows).to(hidden.device))
            torch.cuda.synchronize()
            controls = {
                name: (
                    entry["fp64"]["gate_pass"]
                    and entry["scores"]["score_gate_pass"]
                    and entry["all_rows_finite"]
                    and entry["output_contiguous"]
                    and entry["output_shape"] == [rows, EXPECTED_N]
                    and entry["own_graph_eager_byte_exact"]
                    and entry["changed_H_scale8"]["fp64"]["gate_pass"]
                    and entry["changed_H_scale8"]["scores"]["score_gate_pass"]
                    and entry["changed_H_scale8"]["all_rows_finite"]
                    and entry["changed_H_scale8"]["own_graph_eager_byte_exact"]
                )
                for name, entry in preflight.items()
            }
            all_controls.append(controls)
            emit(
                "padding_preflight_m%d" % rows,
                {
                    "preflight": preflight,
                    "controls": controls,
                    "weight": self.weight_record,
                    "split": self.split_record,
                    "device": self.device_record,
                },
            )
            # No arm with a failed preflight is timed; its explicit failure remains.
            modes = ("graph_cold", "graph_hot")
            raw = {name: {mode: [] for mode in modes} for name in names}
            for order in orders:
                for name in order:
                    if not controls[name]:
                        continue
                    raw[name]["graph_cold"].append(
                        list(
                            triton.testing.do_bench(
                                graphs[name].replay,
                                warmup=10,
                                rep=30,
                                return_mode="all",
                            )
                        )
                    )
                    raw[name]["graph_hot"].append(
                        list(
                            triton.testing.do_bench_cudagraph(
                                arms[name], rep=20, return_mode="all"
                            )
                        )
                    )
            summary = {
                name: {
                    mode: {
                        "round_medians_ms": [
                            statistics.median(values) for values in rounds
                        ],
                        "mean_of_round_medians_ms": statistics.mean(
                            statistics.median(values) for values in rounds
                        ),
                    }
                    for mode, rounds in by_mode.items()
                    if rounds
                }
                for name, by_mode in raw.items()
            }
            emit(
                "padding_performance_m%d" % rows,
                {
                    "rows": rows,
                    "preflight": preflight,
                    "controls": controls,
                    "weight": self.weight_record,
                    "split": self.split_record,
                    "device": self.device_record,
                    "orders": orders,
                    "padded_N": {
                        "pad%d" % alignment: parts[0].shape[0]
                        for alignment, parts in padded_parts.items()
                    },
                    "raw_samples_ms": raw,
                    "summary": summary,
                    "timed_scope": "FP32 baseline H cast; x3 GEMMs/adds; padded x3 also narrow+contiguous crop",
                    "cache_modes": {
                        "graph_cold": "do_bench driver cache clearing before single whole-path replay",
                        "graph_hot": "do_bench_cudagraph amortized whole-path mean; no clearing",
                    },
                    "triton_testing_source_sha256": hashlib.sha256(
                        Path(triton.testing.__file__).read_bytes()
                    ).hexdigest(),
                    "padding_preparation": "once before all M cases; excluded from timing",
                    "scope": "real W/synthetic H; model/e2e/quality NOT_MEASURED",
                },
            )
            del graphs, graph_outputs, arms, hidden, reference, changed_reference, arm
        for controls in all_controls:
            for name, passed in controls.items():
                if name != "fp32":
                    self.assertTrue(passed, name)


if __name__ == "__main__":
    unittest.main()
