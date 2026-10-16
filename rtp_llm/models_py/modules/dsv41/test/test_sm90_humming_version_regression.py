"""Run unchanged RTP oracles in ONE pinned Humming version per process.

DSV41_HUMMING_REGRESSION_SOURCE=installed (default) uses the 0.1.16
dependency. upstream additionally requires DSV41_UPSTREAM_HUMMING pointing
to d113632. Never collect this together with another source-overlay target.
No new numerical reference, tolerance, production selector, or benchmark.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


_REVISION = "d113632a2bbcee5f25fcc8a3a7d92ccdfbac62a8"
_MODE = os.environ.get("DSV41_HUMMING_REGRESSION_SOURCE", "installed")
_HASHES = {
    "installed": {
        "forward.py": "0eb11022b8f4b5cc1d1063f8dd8d645f93b658de445b3bdfd1762f91900dbf71",
        "transform.py": "d92af67fc6c6d329747a78b599c4899ca6f4c72e987053c632f0fffb3787ca1a",
        "config/config.py": "75cee10c83031c4826a5f40518363bb382268dd6410524253e404b74e4beff44",
        "include/humming/mma/wgmma.cuh": "6c7f64c9e1db75f3948b508091e0f1bd6ce7a74251de68b50e96b0f77e02dd09",
    },
    "upstream": {
        "forward.py": "59011c32f58837819d3a24819e4335994552efb9426436b892702dc23964d61d",
        "transform.py": "209cfa1acc08a8ab6b426988012739d45ca2575c28b275e165e10e4c32d5e610",
        "config/config.py": "f83967f03edbe38dedff76d34159ea1251225c07d3b43876b9a681ff3abe6652",
        "include/humming/mma/wgmma.cuh": "88333b8f243c480b89cc2874272e1d6ccf92e69566c7e779a94b21ff162a2a6a",
    },
}


def _select_source():
    if _MODE not in _HASHES:
        raise RuntimeError(f"Unknown regression source {_MODE!r}")
    if any(name == "humming" or name.startswith("humming.") for name in sys.modules):
        raise RuntimeError("Humming already imported; run version regression in a fresh process")
    if _MODE == "upstream":
        root = Path(os.environ["DSV41_UPSTREAM_HUMMING"]).resolve(strict=True)
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != _REVISION:
            raise RuntimeError(f"Expected {_REVISION}, got {revision}")
        subprocess.run(
            ["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "humming"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        sys.path.insert(0, str(root))
        package = root / "humming"
        # Humming may spawn a launcher-builder interpreter: sys.path alone
        # would not select the same source in that fresh executable.
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(root), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
        )
    else:
        distribution = importlib.metadata.distribution("humming-kernels")
        if distribution.version != "0.1.16":
            raise RuntimeError(f"Expected installed 0.1.16, got {distribution.version}")
        # Bazel runfiles may symlink individual files, not the directory.
        package = Path(distribution.locate_file("humming/__init__.py")).resolve(strict=True).parent
    spec = importlib.util.find_spec("humming")
    if spec is None or spec.origin is None:
        raise RuntimeError("Humming import origin is missing")
    if Path(spec.origin).resolve() != package / "__init__.py":
        raise RuntimeError(f"Unexpected {_MODE} import origin: {spec.origin}; expected {package}")
    for relative, expected in _HASHES[_MODE].items():
        actual = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned source mismatch: {relative}: {actual} != {expected}")
    return package


_PACKAGE = _select_source()  # Must run before torch/RTP helpers import Humming.

import pytest
import torch
import humming

from rtp_llm.models_py.modules.dsv41.test import test_sm90_humming_moe as _moe
from rtp_llm.models_py.modules.dsv41.test import test_sm90_humming_moe_shapes as _shapes
from rtp_llm.models_py.modules.dsv41.test import test_sm90_mxfp8_dense as _dense


def _provenance():
    modules = {}
    native_modules = {}
    cache = Path(os.environ.get("HUMMING_CACHE_DIR", Path.home() / ".humming/cache")).resolve()
    for name, module in tuple(sys.modules.items()):
        if not (name == "humming" or name.startswith("humming.")):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(_PACKAGE):
            modules[name] = str(path)
        elif name == "humming._device_info" and path.name == "_device_info.abi3.so" and path.is_relative_to(cache):
            native_modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            raise AssertionError(f"Mixed Humming versions: {name} from {path}")
    return {
        "source": _MODE, "revision": _REVISION if _MODE == "upstream" else "wheel-0.1.16",
        "root": str(_PACKAGE), "reported_version": humming.__version__,
        "verified_sha256": _HASHES[_MODE], "module_paths": modules,
        "native_modules": native_modules, "pid": os.getpid(),
        "scope": "Unmodified production default plus original RTP oracles; not candidate tuning",
    }


@pytest.fixture(scope="module", autouse=True)
def _require_sm90_and_single_version():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This dedicated compatibility target requires SM90; skip is not PASS")
    print("SM90_HUMMING_VERSION_REGRESSION " + json.dumps(_provenance()), flush=True)
    yield
    print("SM90_HUMMING_VERSION_REGRESSION_FINAL " + json.dumps(_provenance()), flush=True)


# Import the original small routing and complete MoE oracles unchanged.
SM90RoutingContractTest = _moe.SM90RoutingContractTest
SM90GroupedExpertTest = _moe.SM90GroupedExpertTest
SM90ActualRoutedShapeTest = _shapes.SM90ActualRoutedShapeTest


class SM90DenseVersionCompatibilityTest(unittest.TestCase):
    """Bounded subset, retaining original functions/references and tolerances.

    Excludes the deliberate non-invariant A/B diagnostic, profiling, and graph
    replay. Includes real TP8/shared shapes because dependency selection also
    changes the production dense path, not just routed MoE.
    """

    setUpClass = _dense.Sm90MXFP8DenseTest.__dict__["setUpClass"]
    setUp = _dense.Sm90MXFP8DenseTest.setUp
    _linear = _dense.Sm90MXFP8DenseTest._linear
    _reference = _dense.Sm90MXFP8DenseTest._reference
    test_prequantized_group32_and_out_match_independent_reference = (
        _dense.Sm90MXFP8DenseTest.test_prequantized_group32_and_out_match_independent_reference
    )
    test_actual_tp8_dense_projection_shapes_at_decode_batch = (
        _dense.Sm90MXFP8DenseTest.test_actual_tp8_dense_projection_shapes_at_decode_batch
    )
    test_empty_rows_preserve_out_without_launch = (
        _dense.Sm90MXFP8DenseTest.test_empty_rows_preserve_out_without_launch
    )
