"""Focused CPU/stdlib proof test; no Torch or CUDA import is needed."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapter"))


def missing_flash_getter():
    raise ModuleNotFoundError("The optional package is absent", name="flash_attn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--einops-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert "torch" not in sys.modules
    sys.path.insert(0, str(Path(args.einops_path).resolve()))
    import einops
    assert einops.__version__ == "0.8.2"
    import q397_torch_padding as shim
    assert "torch" not in sys.modules
    real_hostname, real_pid = socket.gethostname, os.getpid
    checks = []
    with tempfile.TemporaryDirectory(prefix="q397-padding-proof-") as temporary:
        run = Path(temporary)
        receipts = []
        for hostname in ("container-node1", "container-node2", "unsafe/../name:one"):
            module = SimpleNamespace(_get_attention_functions=missing_flash_getter, __file__=__file__)
            shim.install(module)
            with patch.dict(os.environ, {"Q397_RUN_ROOT": str(run)}), patch.object(os, "getpid", return_value=4242), patch.object(socket, "gethostname", return_value=hostname):
                functions = module._get_attention_functions()
                assert functions[0] is shim.index_first_axis
                # An idempotent second call must preserve the first receipt.
                before = {p.name: p.read_bytes() for p in (run / "validation/padding").glob("*.json")}
                module._get_attention_functions()
                assert before == {p.name: p.read_bytes() for p in (run / "validation/padding").glob("*.json")}
            path = run / "validation/padding" / shim._proof_filename(hostname, 4242)
            assert path.parent == run / "validation/padding"
            assert "/" not in path.name and "\\" not in path.name
            proof = json.loads(path.read_text())
            assert proof["hostname"] == hostname and proof["pid"] == 4242
            assert proof["source"]["sha256"] == hashlib.sha256(Path(shim.__file__).read_bytes()).hexdigest()
            assert proof["flash_attention_kernel_used"] is False
            receipts.append({"filename": path.name, "hostname": proof["hostname"], "pid": proof["pid"]})
            checks.append(hostname + ":actual_getter_first_use_and_idempotent_repeat")
        assert len(list((run / "validation/padding").glob("*.json"))) == 3
        assert shim._proof_filename("container-node1", 4242) == "first-use-container-node1-4242.json"
        assert shim._proof_filename("container-node2", 4242) == "first-use-container-node2-4242.json"
        assert shim._proof_filename("unsafe/host", 4242) != shim._proof_filename("unsafe:host", 4242)
        checks.append("same_pid_different_hostnames_do_not_collide")
        checks.append("different_unsafe_hostnames_do_not_collapse_to_one_name")
    assert socket.gethostname is real_hostname and os.getpid is real_pid
    assert "torch" not in sys.modules
    result = {"schema": "q397-padding-proof-filename-cpu-regression-v1", "status": "passed",
              "checks": checks, "receipts": receipts, "torch_imported": False,
              "shim_source_sha256": hashlib.sha256(Path(shim.__file__).read_bytes()).hexdigest(),
              "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
