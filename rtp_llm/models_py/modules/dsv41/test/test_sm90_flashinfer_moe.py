"""Independent gates for internal + open-source FlashInfer coexistence.

The repository exposes two packages; no sys.modules alias, installed-package
replacement, or runtime dependency fallback is allowed. Each import order runs
in a fresh worker with both packages present. Parent owns GPU/Bazel execution.
MoE precision/route-position acceptance is separate from this first norm gate.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_VERSIONS = {"flashinfer": "0.6.9+8c4f4dcf", "flashinfer_opensource": "0.7.1"}


def _coexist_worker(order, output):
    import torch

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    assert not any(name in sys.modules for name in _VERSIONS)
    names = list(_VERSIONS) if order == "internal_first" else list(reversed(_VERSIONS))
    first = importlib.import_module(names[0])
    first_norm = importlib.import_module(names[0] + ".norm")
    first_function = first_norm.rmsnorm
    first_schemas = {
        name
        for name in torch._C._dispatch_get_all_op_names()
        if name.startswith(names[0] + "::")
    }
    old_modules = {
        key: module
        for key, module in sys.modules.items()
        if key == names[0] or key.startswith(names[0] + ".")
    }
    second = importlib.import_module(names[1])
    second_norm = importlib.import_module(names[1] + ".norm")
    assert first is not second and first_norm is not second_norm
    assert first_function is first_norm.rmsnorm
    assert first_function is not second_norm.rmsnorm
    for key, module in old_modules.items():
        assert sys.modules[key] is module, f"Import replaced existing module {key}"
    assert first_schemas == {
        name
        for name in torch._C._dispatch_get_all_op_names()
        if name.startswith(names[0] + "::")
    }

    packages = {name: sys.modules[name] for name in _VERSIONS}
    roots = {
        name: Path(module.__file__).resolve().parent
        for name, module in packages.items()
    }
    assert roots["flashinfer"] != roots["flashinfer_opensource"]
    norms = {name: importlib.import_module(name + ".norm") for name in packages}
    schemas = {}
    cache_roots = {}
    for name, package in packages.items():
        assert package.__name__ == name
        assert package.__version__ == _VERSIONS[name], (name, package.__version__)
        assert norms[name].__name__ == name + ".norm"
        assert Path(norms[name].__file__).resolve().is_relative_to(roots[name])
        env = importlib.import_module(name + ".jit.env")
        paths = {
            key: str(value)
            for key, value in vars(env).items()
            if key.endswith("CACHE_DIR")
        }
        assert paths, f"No observable JIT cache root for {name}"
        cache_roots[name] = paths
    assert set(cache_roots["flashinfer"].values()).isdisjoint(
        cache_roots["flashinfer_opensource"].values()
    )

    def bits(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype
        torch.testing.assert_close(
            a.contiguous().view(torch.uint8),
            b.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )

    def sha(tensor):
        return hashlib.sha256(
            tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()

    results = {}
    for rows, width in ((1, 128), (4, 5120)):
        torch.manual_seed(26091 + rows)
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
        saved_x, saved_weight = x.clone(), weight.clone()
        reference = (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).bfloat16()
        for name in names:
            out = torch.empty_like(x)
            actual = norms[name].rmsnorm(x, weight, eps=1e-6, out=out, enable_pdl=False)
            assert actual.data_ptr() == out.data_ptr()
            # BF16 normalization roundoff gate, not an MoE/native-A8 waiver.
            torch.testing.assert_close(actual, reference, rtol=8e-3, atol=1e-3)
            bits(x, saved_x)
            bits(weight, saved_weight)
            results[f"{name}_{rows}_{width}"] = sha(actual)
            # A second direct call after both packages have built/registered
            # their low-cost operator must still dispatch to this package.
            bits(norms[name].rmsnorm(x, weight, eps=1e-6, enable_pdl=False), actual)

    # These pinned versions intentionally make register_custom_op an identity
    # decorator. Do not enable torch.library registration just to satisfy a
    # test. If a schema does exist, its namespace must still be independent.
    native_modules, native_libraries = {}, {}
    for name in names:
        try:
            schema = torch._C._dispatch_find_schema_or_throw(
                name + "::rmsnorm", ""
            ).schema()
        except RuntimeError:
            utility = importlib.import_module(name + ".utils")

            def identity_probe():
                return None

            assert (
                utility.register_custom_op(
                    name + "::coexist_identity_probe", mutates_args=()
                )(identity_probe)
                is identity_probe
            )
            assert (
                utility.register_fake_op(name + "::coexist_identity_probe")(
                    identity_probe
                )
                is identity_probe
            )
            schemas[name] = (
                "NOT_APPLICABLE: pinned register_custom_op and register_fake_op are identity decorators"
            )
        else:
            schemas[name] = str(schema)
            assert str(schema).startswith(name + "::rmsnorm(")
        native_modules[name] = norms[name].get_norm_module()
        library = norms[name].gen_norm_module().get_library_path().resolve(strict=True)
        native_libraries[name] = dict(
            path=str(library), sha256=hashlib.sha256(library.read_bytes()).hexdigest()
        )
        assert (
            str(library) in Path("/proc/self/maps").read_text()
        ), "Norm library must actually be loaded"
    assert native_modules["flashinfer"] is not native_modules["flashinfer_opensource"]
    assert (
        native_libraries["flashinfer"]["path"]
        != native_libraries["flashinfer_opensource"]["path"]
    )

    # A captured call to BOTH packages checks live input/output isolation,
    # not just that their Python imports happen to finish without errors.
    x = torch.ones(2, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    outs = {name: torch.empty_like(x) for name in names}

    def run_both():
        for name in names:
            norms[name].rmsnorm(x, weight, eps=1e-6, out=outs[name], enable_pdl=False)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run_both()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_both()
    for seed in (26011, 26012, 26013):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        weight.copy_(torch.randn_like(weight))
        expected = {
            name: norms[name].rmsnorm(x, weight, eps=1e-6, enable_pdl=False)
            for name in names
        }
        graph.replay()
        torch.cuda.synchronize()
        for name in names:
            bits(outs[name], expected[name])
    for key, module in old_modules.items():
        assert sys.modules[key] is module
    for name in names:
        assert norms[name].get_norm_module() is native_modules[name]
    record = dict(
        order=order,
        versions={name: p.__version__ for name, p in packages.items()},
        roots={name: str(root) for name, root in roots.items()},
        schemas=schemas,
        native_libraries=native_libraries,
        cache_roots=cache_roots,
        output_hashes=results,
        graph="both packages, changed x/weight, exact vs eager",
        scope="coexistence + RMSNorm only; not MoE acceptance",
    )
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    print("FLASHINFER_COEXIST " + json.dumps(record), flush=True)


def _child_environment(directory):
    env = os.environ.copy()
    paths, seen = [], set()
    for raw in [*sys.path, *env.get("PYTHONPATH", "").split(os.pathsep)]:
        path = Path(raw or os.getcwd()).resolve()
        if path.is_dir() and str(path) not in seen:
            seen.add(str(path))
            paths.append(str(path))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    assert len(env["PYTHONPATH"].encode()) < 96 * 1024
    # These are independent documented package workspaces, not path aliases.
    env["FLASHINFER_WORKSPACE_BASE"] = str(directory / "internal_workspace")
    env["FLASHINFER_OPENSOURCE_WORKSPACE_BASE"] = str(
        directory / "opensource_workspace"
    )
    env["FLASHINFER_USE_CUDA_NORM"] = "1"
    env["FLASHINFER_OPENSOURCE_USE_CUDA_NORM"] = "1"
    return env


@pytest.mark.parametrize("order", ["internal_first", "opensource_first"])
def test_coexist_import_order_real_norm_and_graph(tmp_path, order):
    if os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        tmp_path = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            / f"flashinfer_coexist_{order}"
        )
        tmp_path.mkdir(parents=True, exist_ok=False)
    record = tmp_path / "coexist.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--coexist-worker",
        order,
        "--output",
        str(record),
    ]
    result = subprocess.run(
        command,
        env=_child_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log = tmp_path / "worker.log"
    log.write_text(result.stdout + "\n" + result.stderr)
    print(result.stdout, flush=True)
    assert (
        result.returncode == 0
    ), f"{order} failed; log={log}\n{result.stderr[-16000:]}"
    data = json.loads(record.read_text())
    assert data["versions"] == _VERSIONS and data["order"] == order


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coexist-worker",
        choices=["internal_first", "opensource_first"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _coexist_worker(args.coexist_worker, args.output)
