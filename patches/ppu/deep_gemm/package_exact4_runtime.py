"""Package the validated DeepGEMM Python/native combination without rebuilding it."""

import argparse
import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

VERSION = "1.1.0+ppu2.2.0.torch2.11.cu130.exact4runtime3"
EXPECTED = {
    "jit/compiler.py": "c4bd932fa8e57b433749dc236322b96f5e0761a4b8b204b346990e9315985294",
    "_C.cpython-312-x86_64-linux-gnu.so": "6926c504e3945293c65ce78b795b3a357d8d2e22c4078ddf408fca57f7dd778d",
    "mega/__init__.py": "3bc8ca2d154ef8ca34c4ce8085c5f1fb98972e055d9a780f9220a89b76a200e0",
}


# t-head/actlize d00fa98b656810ed49bd1db92d17f6ceb1523f87: include + tools.
HEADER_MANIFEST_SHA256 = (
    "db4292c1aa1d55b3a1e550d47195cfe95ad3c12086c47d49adf2d85bae9a91de"
)

CUDA_HEADER_MANIFEST_SHA256 = (
    "15a2fb3ea8caab497ecd4e2abcbe30e3218a0c08feb5f4ee8f715da9e705aa8a"
)


def package(source: Path, output: Path, actlize_source: Path):
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unvalidated input {name}: {actual}")
    headers = {}
    for folder, prefix in (("include", ""), ("tools", "tools/")):
        root = actlize_source / folder
        for path in root.rglob("*"):
            if path.is_file():
                headers[prefix + str(path.relative_to(root))] = path.read_bytes()
    headers = dict(sorted(headers.items()))
    manifest = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in headers.items()
    )
    if hashlib.sha256(manifest.encode()).hexdigest() != HEADER_MANIFEST_SHA256:
        raise ValueError("Headers do not match the exact4 native source")
    cuda_root = (source / "include/actlize_v1.0.0").resolve()
    cuda_headers = {
        path.relative_to(cuda_root).as_posix(): path.read_bytes()
        for path in sorted(cuda_root.rglob("*"))
        if path.is_file()
    }
    cuda_manifest = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in cuda_headers.items()
    )
    if (
        hashlib.sha256(cuda_manifest.encode()).hexdigest()
        != CUDA_HEADER_MANIFEST_SHA256
    ):
        raise ValueError("Unvalidated CUDA-compatible Python JIT headers")
    files = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (
            not path.is_file()
            or any(
                part in {"__pycache__", "include.legacy-unused"}
                for part in relative.parts
            )
            or relative.parts[:2] == ("include", "actlize_v1.0.0")
            or path.suffix == ".pyc"
        ):
            continue
        files[f"deep_gemm/{relative.as_posix()}"] = path.read_bytes()
    for name, data in headers.items():
        files[f"deep_gemm/include/actlize_v1.0.0/{name}"] = data
    for name, data in cuda_headers.items():
        files[f"deep_gemm/include/actlize_cuda/{name}"] = data
    # Python nvcc and native hgcc require different unchanged header trees.
    compiler_key = "deep_gemm/jit/compiler.py"
    compiler = files[compiler_key].decode()
    anchor = "    if new_jit_include_dir :\n"
    if compiler.count(anchor) != 1:
        raise ValueError("Unrecognized Python JIT include selector")
    compiler = compiler.replace(
        anchor,
        (
            '    if new_jit_include_dir == "actlize_v1.0.0":\n'
            '        new_jit_include_dir = "actlize_cuda"\n' + anchor
        ),
    )
    anchor = "    # Update `interleave_ffma.py`\n"
    if compiler.count(anchor) != 1:
        raise ValueError("Unrecognized Python JIT cache signature")
    compiler = compiler.replace(
        anchor, (f'    md5.update(b"{CUDA_HEADER_MANIFEST_SHA256}")\n\n' + anchor)
    )
    files[compiler_key] = compiler.encode()
    dist = f"deep_gemm-{VERSION}.dist-info"
    files[f"{dist}/LICENSE"] = (source.parent / "LICENSE").read_bytes()
    files[f"{dist}/licenses/actlize-LICENSE.txt"] = (
        actlize_source / "LICENSE.txt"
    ).read_bytes()
    files[f"{dist}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: deep-gemm\nVersion: {VERSION}\n"
        "Summary: Validated PPU exact4 native with compatible DeepGEMM Python runtime\n"
        "Requires-Python: >=3.12\nLicense: MIT\n\n"
    ).encode()
    files[f"{dist}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: rtp-llm-exact4-runtime\n"
        "Root-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n"
    ).encode()
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow((name, f"sha256={digest}", len(data)))
    writer.writerow((f"{dist}/RECORD", "", ""))
    files[f"{dist}/RECORD"] = records.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"deep_gemm-{VERSION}-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    print(f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Validated deep_gemm package directory"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--actlize-source",
        type=Path,
        required=True,
        help="PPU actlize source tree from the matching exact4 native source",
    )
    args = parser.parse_args()
    package(args.source, args.output, args.actlize_source)
