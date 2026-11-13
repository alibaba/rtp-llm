"""Isolated Python-loader unit checks, NOT native/packaged integration.
Bypass eager model_loader imports. Replace only native ops and disk database
boundaries; execute original loader, tensor source, scale and TP Python files.
"""

import argparse
import json
import sys
import types
import unittest
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--isolated-boundaries",
    action="store_true",
    help="Explicitly substitute native ops/database boundaries",
)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
if not args.isolated_boundaries:
    parser.error(
        "Use the real unittest/Bazel target in a built environment, or explicitly opt into --isolated-boundaries"
    )
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
package = types.ModuleType("rtp_llm.model_loader")
package.__path__ = [str(ROOT / "rtp_llm/model_loader")]
sys.modules[package.__name__] = package
import rtp_llm

rtp_llm.model_loader = package
ops = types.ModuleType("rtp_llm.ops")
ops.VitSeparation = type("VitSeparation", (), {})
ops.LinearAttentionConfig = type("LinearAttentionConfig", (), {})
sys.modules[ops.__name__] = ops
database = types.ModuleType("rtp_llm.utils.database")
database.BaseDatabase = type("BaseDatabase", (), {})
database.CkptDatabase = type("CkptDatabase", (database.BaseDatabase,), {})
sys.modules[database.__name__] = database
import torch

print(
    "Torch:",
    torch.__version__,
    "E8M0 dtype:",
    hasattr(torch, "float8_e8m0fnu"),
    flush=True,
)
suite = unittest.defaultTestLoader.loadTestsFromNames(
    [
        "rtp_llm.model_loader.test.test_compressed_tensors_nvfp4",
        "rtp_llm.model_loader.test.test_fp4_exclude_modules",
    ]
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = dict(
    mode="isolated_python_loader_unit_tests",
    torch=torch.__version__,
    tests=result.testsRun,
    failures=len(result.failures),
    errors=len(result.errors),
    skipped=len(result.skipped),
    success=result.wasSuccessful(),
    substituted_boundaries=["rtp_llm.ops", "rtp_llm.utils.database"],
    bypassed_eager_package_init="rtp_llm.model_loader",
    native_layout_tested=False,
    full_linux_integration_tested=False,
)
if args.output:
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(not result.wasSuccessful())
