"""Check the install paths, JIT inputs and RECORD of the production CUDA wheel."""

import base64
import csv
import hashlib
import io
import json
import sys
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = "rtp_llm/libs/"
MANIFEST = "glm53_extensions.json"


def check_wheel(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel has duplicate members")
        (record_path,) = [name for name in names if name.endswith(".dist-info/RECORD")]
        with archive.open(record_path) as stream:
            rows = {
                path: (digest, size)
                for path, digest, size in csv.reader(io.TextIOWrapper(stream))
            }
        manifest = json.loads(archive.read(ROOT + MANIFEST))
        files = manifest["files"]
        required = {
            "glm53_extension_check.py",
            "rtp_fp8_sparse_decode.abi3.so",
            "mega_mhc_isolated/csrc/jit_kernels/impls/sm100_mega_mhc.hpp",
            "mega_mhc_isolated/include/deep_jit/utils/parser.hpp",
            "mega_mhc_isolated/include/cute/tensor.hpp",
            "mega_mhc_isolated/include/cutlass/cutlass.h",
            "fp8_mqa_adapter.py",
        }
        required.update(
            f"mega_mhc_isolated/licenses/{name}.txt"
            for name in (
                "cutlass_mhc",
                "cutlass_mla",
                "deep_gemm",
                "deep_jit",
                "flash_mla",
                "fmt",
            )
        )
        missing = required.difference(files)
        if missing:
            raise ValueError(f"Missing Blackwell extension assets: {sorted(missing)}")
        prefix = "mega_mhc_isolated/mhc_isolated"
        matches = [f for f in files if f.startswith(prefix) and f.endswith(".so")]
        if len(matches) != 1:
            raise ValueError(f"Expected one binary for {prefix}, found {matches}")
        if any(name.startswith(ROOT + "fp8_mqa_isolated/") for name in names):
            raise ValueError("Wheel still includes the old FP8 MQA runtime")

        adapter = archive.read(ROOT + "fp8_mqa_adapter.py")
        legacy_metadata = object()
        deep_gemm = SimpleNamespace(
            fp8_mqa_logits=object(),
            fp8_paged_mqa_logits=object(),
            get_num_sms=object(),
            get_legacy_paged_mqa_logits_metadata=legacy_metadata,
            get_paged_mqa_logits_metadata=object(),
        )
        exports = {}
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            exec(compile(adapter, "fp8_mqa_adapter.py", "exec"), exports)
        if exports["get_paged_mqa_logits_metadata"] is not legacy_metadata:
            raise ValueError("FP32-scale FP8 MQA must use the legacy schedule")

        for name in [MANIFEST, *files]:
            path = ROOT + name
            data = archive.read(path)
            digest = hashlib.sha256(data)
            encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
            if rows.get(path) != ("sha256=" + encoded, str(len(data))):
                raise ValueError(f"Invalid wheel RECORD for {path}")
            if name != MANIFEST and files[name] != {
                "bytes": len(data),
                "sha256": digest.hexdigest(),
            }:
                raise ValueError(
                    f"Invalid Blackwell extension manifest entry for {path}"
                )
            if name.endswith(".so") and not data.startswith(b"\x7fELF"):
                raise ValueError(f"Invalid ELF binary {path}")
        return len(files)


if __name__ == "__main__":
    print(f"Verified {check_wheel(sys.argv[1])} Blackwell extension assets and RECORD")
