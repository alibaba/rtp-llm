"""Check the install paths, JIT inputs and RECORD of the production CUDA wheel."""

import base64
import csv
import hashlib
import io
import json
import sys
import zipfile

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
            "fp8_mqa_isolated/include/deep_gemm/impls/sm100_mqa_logits.cuh",
            "fp8_mqa_isolated/csrc/jit/compiler.hpp",
            "fp8_mqa_isolated/LICENSE",
        }
        required.update(
            f"mega_mhc_isolated/licenses/{name}.txt"
            for name in (
                "cutlass_mhc",
                "cutlass_mla",
                "deep_gemm",
                "deep_gemm_mqa",
                "deep_jit",
                "flash_mla",
                "fmt",
            )
        )
        missing = required.difference(files)
        if missing:
            raise ValueError(f"Missing Blackwell extension assets: {sorted(missing)}")
        for prefix in (
            "mega_mhc_isolated/mhc_isolated",
            "fp8_mqa_isolated/_rtp_llm_fp8_mqa",
        ):
            matches = [f for f in files if f.startswith(prefix) and f.endswith(".so")]
            if len(matches) != 1:
                raise ValueError(f"Expected one binary for {prefix}, found {matches}")
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
