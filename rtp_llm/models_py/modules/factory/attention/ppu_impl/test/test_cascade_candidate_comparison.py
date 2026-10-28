"""Manual paired timing of frozen Cascade candidates with the original gates.

The unchanged benchmark owns fixtures, state changes, independent CPU64 checks,
graph capture/replay, validation callbacks, and alternating-order event timing.
This entry only supplies several named candidate factories to that same runner.
"""

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import statistics
import sys
import traceback
import unittest
from pathlib import Path

BENCHMARK_SHA256 = "5cd9c7795fff735d9482dcf054631a6368c4cd3408d4150e54b065ee631fd373"
OPTIONS = None
CONFIG = None


def load_file(path, name, expected_sha256):
    path = Path(path).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"source identity changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_file(
    Path(__file__).with_name("test_cascade_kernel_benchmark.py"),
    "ppu_cascade_paired_base",
    BENCHMARK_SHA256,
)


def read_config(path):
    path = Path(path)
    base.require(
        path.is_absolute() and path.is_file(), "comparison config must be absolute"
    )
    data = json.loads(path.read_text())
    base.require(set(data) == {"candidates"}, "unexpected comparison config keys")
    base.require(2 <= len(data["candidates"]) <= 4, "compare two to four candidates")
    names = set()
    for entry in data["candidates"]:
        base.require(
            set(entry) == {"name", "path", "sha256", "factory"},
            "invalid candidate entry",
        )
        name = entry["name"]
        base.require(
            re.fullmatch(r"candidate_[a-z0-9_]+", name) is not None,
            "invalid candidate name",
        )
        base.require(name not in names, "duplicate candidate name")
        names.add(name)
        source = Path(entry["path"])
        base.require(
            source.is_absolute() and source.is_file(),
            "candidate must be an absolute file",
        )
        base.require(
            re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is not None,
            "invalid source SHA",
        )
        base.require(
            entry["factory"] == "factory",
            "only unprofiled normal factories are admitted",
        )
        base.require(
            base.digest_file(source) == entry["sha256"], "candidate source changed"
        )
    return {"path": str(path.resolve()), "sha256": base.digest_file(path), **data}


def paired_samples(timing, names):
    """Descriptive same-trial ratios, without an independence/significance claim."""
    raw = timing["raw_samples"]
    result = []
    for first, second in itertools.combinations(names, 2):
        a, b = raw[first]["graph_us"], raw[second]["graph_us"]
        base.require(len(a) == len(b) and len(a) > 1, "missing paired timing samples")
        base.require(
            all(math.isfinite(x) and x > 0 for x in a + b), "invalid event timing"
        )
        ratios = [x / y for x, y in zip(a, b)]
        differences = [x - y for x, y in zip(a, b)]
        result.append(
            {
                "baseline": first,
                "candidate": second,
                "trial_count": len(a),
                "paired_speedup_ratios": ratios,
                "median_paired_speedup": statistics.median(ratios),
                "paired_saved_us": differences,
                "median_paired_saved_us": statistics.median(differences),
                "candidate_faster_trials": sum(x > 0 for x in differences),
                "equal_trials": sum(x == 0 for x in differences),
                "candidate_slower_trials": sum(x < 0 for x in differences),
                "claim_scope": "same process/trial with alternating complete order; descriptive, not randomized or independent trials",
            }
        )
    return result


class CascadeCandidateComparison(base.CascadeKernelBenchmark):
    def test_accuracy_graph_fidelity_and_latency(self):
        options, config = OPTIONS, CONFIG
        base.require(
            options is not None and config is not None, "use the Bazel entry point"
        )
        base.require(
            base.get_device_type() == base.DeviceType.Ppu, "requires actual PPU"
        )
        base.require(
            not options.candidate, "use comparison config instead of --candidate"
        )
        base.require(
            not any(
                (
                    options.diagnose_stages,
                    options.diagnostic_timeline,
                    options.diagnose_online_proxy,
                )
            ),
            "paired timing excludes diagnostics",
        )
        base.require(
            os.environ.get("RTP_S1_TRACE_ONCE", "0") == "0", "profiler must be off"
        )
        base.require(options.trials > 1, "paired comparison requires multiple trials")
        base.torch.set_num_threads(4)
        base.torch.manual_seed(options.seed)
        base.require(base.torch.cuda.is_available(), "PPU runtime unavailable")
        output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = output_dir / options.output_name
        self.report = {
            "schema_version": 1,
            "status": "RUNNING",
            "evidence": "RTP_MEASURED",
            "scope": "frozen candidates, same fixture/state and process; off/on retained; all original output and graph gates",
            "benchmark_sha256": BENCHMARK_SHA256,
            "comparison_entry_sha256": base.digest_file(__file__),
            "comparison_config": config,
            "options": vars(options),
            "cpu_threads": base.torch.get_num_threads(),
            "device": base.torch.cuda.get_device_name(),
            "torch_version": base.torch.__version__,
            "lse_unit": "log2",
            "history_semantics": "valid KV tokens = sequence_lengths + 1",
            "legacy_output_tolerance": {
                "atol": base.LEGACY_ATOL,
                "rtol": base.LEGACY_RTOL,
            },
            "lse_reference_gate": "NOT_DEFINED; error measured; repeat/graph require exact O and LSE",
            "cases": [],
            "failures": [],
        }
        factories = {"off": base.DecodeBackend, "on": base.CascadeBackend}
        names = []
        self.write_report()
        try:
            for entry in config["candidates"]:
                name = entry["name"]
                module = load_file(
                    entry["path"],
                    "ppu_cascade_paired_" + name + "_" + entry["sha256"],
                    entry["sha256"],
                )
                factories[name] = getattr(module, entry["factory"])
                names.append(name)
            for index, spec in enumerate(base.case_specs(options)):
                print(f"CASCADE_PAIRED case={spec['name']} stage=begin", flush=True)
                self.run_case(spec, factories, options, index)
                case = self.report["cases"][-1]
                # The base records A (graph plan vs plain) and gates B (replay
                # vs graph-safe eager). Also enforce its recorded A here.
                for state in case["states"]:
                    for name in names:
                        comparison = state["comparisons"][
                            name + ":graph_plan_vs_non_graph"
                        ]
                        for field in ("output", "lse_log2"):
                            if comparison[field]["bitwise_mismatch_count"]:
                                self.report["failures"].append(
                                    {
                                        "check": f"{spec['name']}:state{state['state_index']}:{name}:graph_plan:{field}",
                                        "error": "bitwise mismatch",
                                    }
                                )
                case["paired_comparisons"] = paired_samples(case["timing"], names)
                gc.collect()
                self.write_report()
            self.report["status"] = (
                "REGRESSION_FAIL" if self.report["failures"] else "REGRESSION_PASS"
            )
        except Exception:
            self.report["status"] = "ERROR"
            self.report["exception"] = traceback.format_exc()
            self.write_report()
            raise
        self.write_report()
        self.assertFalse(
            self.report["failures"],
            "see comparison JSON; original on failures are retained",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--comparison-config", required=True)
    custom, remaining = parser.parse_known_args()
    CONFIG = read_config(custom.comparison_config)
    sys.argv = [sys.argv[0]] + remaining
    OPTIONS, unittest_args = base.parse_options()
    unittest.main(argv=[sys.argv[0]] + unittest_args)
