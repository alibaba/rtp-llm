"""Native GPU cache restore with inaccessible absolute producer paths."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton import knobs

from rtp_llm.utils import jit_cache_manager as jit


@triton.jit
def _current_tree_kernel(x, y, size: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(x + offsets, offsets < size, other=0)
    tl.store(y + offsets, values * 3 + 7, offsets < size)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote_hashes(root):
    return {path.name: digest(path) for path in root.glob("*.jit_snapshot.tar.zst")}


def run_worker(stage, root):
    if os.getuid() == 0:
        raise RuntimeError("use the approved non-root GPU container")
    if not torch.cuda.is_available() or not torch.version.cuda.startswith("13."):
        raise RuntimeError("this native regression requires CUDA13")
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this native regression requires Blackwell")

    local = root / ("local-cold" if stage == "cold" else "local-restored")
    remote = root / "remote"
    remote.mkdir(exist_ok=True)
    jit.LOCAL_JIT_DIR = str(local)
    for component in jit.COMPONENTS:
        os.environ.pop(component.env_name, None)
    os.environ.pop("TRITON_CACHE_MANAGER", None)
    components, _ = jit.setup_jit_cache_env()
    component = next(item for item in components if item.name == "triton")
    reader = "rtp_llm.utils.jit_cache_triton:RelocatableFileCacheManager"
    if os.environ["TRITON_CACHE_MANAGER"] != reader:
        raise AssertionError("managed Triton did not select the supported reader")
    manager = jit.JitCacheManager(remote, (component,))
    old_listener = knobs.compilation.listener
    old_load_hook = knobs.runtime.kernel_load_end_hook
    compilations, loads = [], []

    def compiled(**event):
        compilations.append(event["cache_hit"])
        if old_listener is not None:
            old_listener(**event)

    def loaded(module, function, name, metadata_group, kernel_hash):
        loads.append({"hash": kernel_hash, "group": dict(metadata_group)})
        if old_load_hook is not None:
            old_load_hook(module, function, name, metadata_group, kernel_hash)

    knobs.compilation.listener = compiled
    knobs.runtime.kernel_load_end_hook = loaded
    try:
        remote_before = remote_hashes(remote)
        manager.start_background_sync()
        files_before = {
            name: digest(path) for name, path in manager._snapshot_files().items()
        }
        if stage == "restore":
            if (root / "local-cold").exists():
                raise AssertionError("the absolute producer tree is still accessible")
            if not files_before:
                raise AssertionError(
                    "the original snapshot transport restored no files"
                )
        x = torch.arange(257, dtype=torch.int32, device="cuda")
        y = torch.empty_like(x)
        kernel = _current_tree_kernel[(triton.cdiv(x.numel(), 128),)](
            x, y, x.numel(), 128
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x * 3 + 7, rtol=0, atol=0)
        if compilations != [stage == "restore"]:
            raise AssertionError(
                f"unexpected native Triton compiler hits: {compilations}"
            )
        if len(loads) != 1 or loads[0]["hash"] != kernel.hash:
            raise AssertionError("the actual kernel load was not observed")
        if loads[0]["group"] != kernel.metadata_group:
            raise AssertionError("the executed kernel differs from the returned group")
        for path in kernel.metadata_group.values():
            if component.local_dir.resolve() not in Path(path).resolve().parents:
                raise AssertionError(f"kernel loaded outside the current scope: {path}")
        binary = next(
            Path(path)
            for name, path in kernel.metadata_group.items()
            if name.endswith(".cubin")
        )
        if kernel.kernel != binary.read_bytes():
            raise AssertionError("executed binary bytes differ from the current file")
        group = next(binary.parent.glob("__grp__*.json"))
        archived_children = json.loads(group.read_text())["child_paths"]
        if stage == "restore" and any(
            Path(path).exists() for path in archived_children.values()
        ):
            raise AssertionError(
                "an archived absolute producer path is still accessible"
            )
        if stage == "cold":
            if not manager._dirty.wait(10):
                raise AssertionError(
                    "the original observer did not notice native output"
                )
            manager.publish_pending_snapshot()
        manager.stop()
        files_after = {
            name: digest(path) for name, path in manager._snapshot_files().items()
        }
        remote_after = remote_hashes(remote)
        if stage == "restore":
            if files_after != files_before or remote_after != remote_before:
                raise AssertionError("native restore rewrote cached or archived bytes")
        elif not remote_after:
            raise AssertionError("the original manager published no snapshot")
        report = {
            "stage": stage,
            "pid": os.getpid(),
            "kernel_hash": kernel.hash,
            "binary": str(binary),
            "binary_sha256": digest(binary),
            "group_sha256": digest(group),
            "loaded_group": kernel.metadata_group,
            "native_compiler_cache_hits": compilations,
            "observed_kernel_loads": loads,
            "snapshot_files": files_after,
            "remote_archives": remote_after,
            "source_sha256": digest(__file__),
            "triton_version": triton.__version__,
            "torch_version": torch.__version__,
            "triton_cache_source_sha256": digest(
                Path(triton.__file__).parent / "runtime/cache.py"
            ),
            "output": y.cpu().tolist(),
        }
        (root / f"{stage}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        knobs.compilation.listener = old_listener
        knobs.runtime.kernel_load_end_hook = old_load_hook
        manager.stop()


class TritonNativeCacheRestoreTest(unittest.TestCase):
    def test_actual_kernel_loads_current_tree_with_original_tree_unavailable(self):
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output:
            root = Path(output) / "triton-current-tree"
            root.mkdir(exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
        for stage in ("cold", "restore"):
            if stage == "restore":
                (root / "local-cold").rename(root / "retained-cold")
                self.assertFalse((root / "local-cold").exists())
            log = root / f"{stage}.log"
            with log.open("w") as stream:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--cache-worker",
                        stage,
                        str(root),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    check=False,
                )
            self.assertEqual(result.returncode, 0, log.read_text()[-6000:])
        cold = json.loads((root / "cold.json").read_text())
        restored = json.loads((root / "restore.json").read_text())
        self.assertNotEqual(cold["pid"], restored["pid"])
        self.assertNotEqual(cold["binary"], restored["binary"])
        self.assertEqual(cold["native_compiler_cache_hits"], [False])
        self.assertEqual(restored["native_compiler_cache_hits"], [True])
        for field in (
            "kernel_hash",
            "binary_sha256",
            "group_sha256",
            "snapshot_files",
            "remote_archives",
            "source_sha256",
            "triton_version",
            "torch_version",
            "triton_cache_source_sha256",
            "output",
        ):
            self.assertEqual(cold[field], restored[field], field)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cache-worker"]:
        run_worker(sys.argv[2], Path(sys.argv[3]))
    else:
        unittest.main()
