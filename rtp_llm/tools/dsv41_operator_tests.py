"""Reproducible standalone SM120 operator tests, independent of engine startup."""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

SUITES = (
    "attention",
    "indexer",
    "cache_codec",
    "engram_hash",
    "engram_lookup",
    "engram_inject",
    "mxfp8",
    "expert",
    "routed",
    "engram",
    "dispatch",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", metavar="SUITE")
    parser.add_argument("--without-engine-bindings", action="store_true")
    parser.add_argument("--flashinfer-source-jit", action="store_true")
    parser.add_argument("--extra-package-path", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    args.suites = args.suites or list(SUITES)
    if any(name not in SUITES for name in args.suites):
        parser.error("Unknown suite; choose from " + ", ".join(SUITES))
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.extend(args.extra_package_path)
    if args.flashinfer_source_jit:
        # An optional AOT wheel may target a different CUDA runtime. Keep it
        # untouched and explicitly qualify this process's source JIT instead.
        sys.modules["flashinfer_jit_cache"] = None
    if args.without_engine_bindings:
        name = "rtp_llm.models_py.modules"
        package = types.ModuleType(name)
        package.__path__ = [str(root / "rtp_llm/models_py/modules")]
        package.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        sys.modules[name] = package

    import torch
    import triton

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("V4.1 operator qualification requires an exact SM120 device")
    print(
        "Torch",
        torch.__version__,
        "CUDA",
        torch.version.cuda,
        "Triton",
        triton.__version__,
        flush=True,
    )
    print(
        "GPU",
        torch.cuda.get_device_name(),
        "engine bindings bypassed",
        args.without_engine_bindings,
        flush=True,
    )
    suite = unittest.TestSuite()
    for requested in args.suites:
        name = "test_v41_" + requested + "_hardware"
        path = root / "rtp_llm/models_py/modules/dsv4/fp8/test" / (name + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.TestLoader().loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.manifest:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        manifest = {
            "git_head": subprocess.check_output(
                ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
            ).strip(),
            "runner_sha256": sha256(__file__),
            "engine_bindings_bypassed": args.without_engine_bindings,
            "flashinfer_source_jit": args.flashinfer_source_jit,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": {
                "name": properties.name,
                "uuid": str(properties.uuid),
                "total_memory": properties.total_memory,
                "sm": [properties.major, properties.minor],
            },
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "tests_run": result.testsRun,
            "passed": result.wasSuccessful(),
            "failures": [test.id() for test, _ in result.failures],
            "errors": [test.id() for test, _ in result.errors],
            "suites": args.suites,
            "source_sha256": {
                str(path.relative_to(root)): sha256(path)
                for pattern in (
                    "v41_*.py",
                    "_v41_*.py",
                    "fp8/_v41_*.py",
                    "fp8/test/test_v41_*_hardware.py",
                )
                for path in (root / "rtp_llm/models_py/modules/dsv4").glob(pattern)
            },
        }
        libraries = {}
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if name.startswith("deep_gemm") and filename and filename.endswith(".so"):
                libraries[filename] = sha256(filename)
        for name in ("deep_gemm", "flashinfer"):
            if name in sys.modules:
                module = sys.modules[name]
                manifest[name] = {
                    "version": module.__version__,
                    "path": module.__file__,
                }
        if "flashinfer.jit.core" in sys.modules:
            registry = sys.modules["flashinfer.jit.core"].jit_spec_registry
            for spec in registry.get_all_specs().values():
                path = spec.get_library_path()
                if path.exists():
                    libraries[str(path)] = sha256(path)
        manifest["provider_library_sha256"] = libraries
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Manifest", args.manifest, flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
