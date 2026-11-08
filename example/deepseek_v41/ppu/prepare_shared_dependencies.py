"""Prepare one isolated PPU DeepGEMM host re-export for shared V4.1 imports.

Run by L0 before the existing GPU-lock/Bazel smoke, never at model import.
The original package, metadata, native extension and package-copy cache are
read-only inputs. Only a fresh output directory may be materialized.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
import stat
from email.parser import BytesParser
from pathlib import Path

SCHEMA = "v41-ppu-shared-deepgemm-host-alias-v1"
SOURCE_COMMIT = "07674e672082c2e06ea9f1ec1993d79684444a1c"
VERSION = "0.0.0"
NATIVE = "deep_gemm_cpp.cpython-312-x86_64-linux-gnu.so"
NATIVE_SHA = "816ff22272d7636c8ee57b960189ada869910af74016f35d53638248fcdc3942"
LAYOUT_BEFORE_SHA = "3ea3e40fe44d8d302feef1091142344af68de616b8c2d758f384e96b535f1441"
LAYOUT_AFTER_SHA = "9211d06752fd1e5a915198227f9c1651d678894b311b3a48a45cf91e48f23381"
HEADER_PREFIX = "801e73833647717c"
VARIANT = "pip_v41_ppu_shared_deep_gemm_" + LAYOUT_AFTER_SHA[:16]
CACHE_KEY = f"deep_gemm_python-{VERSION}__{VARIANT}__{HEADER_PREFIX}"
MANIFEST = "rtp_v41_ppu_shared_adapter.json"
APPEND = (
    b"\n# PPU shared-model host import: preserve the existing unsupported native API.\n"
    b"from ..deep_gemm_cpp import get_mn_major_tma_aligned_packed_ue8m0_tensor\n"
)
PINNED_FILES = {
    "__init__.py": (
        "8d0883a94786920ec2588aa73fd25a947506daa424b92e85c8daa80123abfb97",
        7659, 0o644,
    ),
    "utils/__init__.py": (
        "bf31f6e9fc30482ef83a73d63fdb6ed4d46deec847b35c0f9d39816d9b923bb4",
        116, 0o644,
    ),
    "utils/layout.py": (LAYOUT_BEFORE_SHA, 514, 0o644),
    NATIVE: (NATIVE_SHA, 2741136, 0o755),
}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(path):
    """Record package bytes/modes and preserve symlinks without writing targets."""
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    result = {}
    for item in files:
        name = "." if item == path else item.relative_to(path).as_posix()
        mode = stat.S_IMODE(item.lstat().st_mode)
        if item.is_symlink():
            result[name] = {"link": str(item.readlink()), "mode": mode}
        elif item.is_file():
            result[name] = {"sha256": _sha(item), "bytes": item.stat().st_size, "mode": mode}
    return result


def _headers(package):
    # Exact input/ordering of the existing gpu_lock _cache_key implementation.
    root = package / "include/deep_gemm"
    if not root.is_dir():
        raise ValueError("qualified DeepGEMM include/deep_gemm is missing")
    digest = hashlib.sha256()
    for header in sorted(root.rglob("*")):
        if header.is_file():
            digest.update(header.relative_to(root).as_posix().encode() + b"\0")
            digest.update(header.read_bytes() + b"\0")
    return digest.hexdigest()


def _distribution(base):
    # Copy actual metadata unchanged, including its original wheel provenance.
    candidates = []
    for entry in sorted(base.iterdir()):
        if entry.name.endswith(".dist-info"):
            info = entry / "METADATA"
        elif entry.name.endswith(".egg-info"):
            info = entry / "PKG-INFO" if entry.is_dir() else entry
        else:
            continue
        if not info.is_file():
            continue
        parsed = BytesParser().parsebytes(info.read_bytes())
        name = re.sub(r"[-_.]+", "-", parsed.get("Name", "")).lower()
        if name == "deep-gemm":
            if entry.is_symlink() or parsed.get("Version") != VERSION:
                raise ValueError("unexpected DeepGEMM metadata identity")
            candidates.append(entry)
    if len(candidates) != 1:
        raise ValueError("require exactly one original DeepGEMM distribution metadata")
    selected = [
        dist for dist in metadata.distributions(path=[str(base)])
        if re.sub(r"[-_.]+", "-", dist.metadata.get("Name", "")).lower() == "deep-gemm"
    ]
    if len(selected) != 1 or selected[0].version != VERSION:
        raise ValueError("importlib.metadata cannot uniquely resolve the pinned distribution")
    if Path(selected[0].locate_file("deep_gemm")).resolve() != (base / "deep_gemm").resolve():
        raise ValueError("distribution metadata points outside the selected package")
    return candidates[0]


def _verify_cache(cache_root, package_files, header_sha, adjacent_libraries):
    target = cache_root / CACHE_KEY
    if not target.exists() and not target.is_symlink():
        return "ABSENT"
    package = target / "site-packages/deep_gemm"
    if target.is_symlink() or package.is_symlink():
        raise ValueError("new cache key is a symlink; leave it untouched")
    marker = target / ".copy_complete"
    if not marker.is_file() or marker.read_text() != f"deep_gemm=={VERSION}\n":
        raise ValueError("new cache key is incomplete; leave it untouched")
    receipt = json.loads((package / MANIFEST).read_text())
    if (
        receipt["schema"] != SCHEMA
        or receipt["cache_key"] != CACHE_KEY
        or receipt["header_sha256"] != header_sha
        or receipt["native_sha256"] != NATIVE_SHA
        or receipt["layout_after_sha256"] != LAYOUT_AFTER_SHA
        or receipt["package_files"] != package_files
        or receipt["adjacent_libraries"] != adjacent_libraries
    ):
        raise ValueError("new cache key has different provenance; leave it untouched")
    actual = _tree(package)
    actual.pop(MANIFEST, None)
    if actual != package_files:
        raise ValueError("new cache key content differs; leave it untouched")
    for name, files in adjacent_libraries.items():
        if _tree(target / "site-packages" / name) != files:
            raise ValueError("new cache adjacent library content differs")
    return "MATCHING_READONLY"


def prepare_shared_dependencies(base_site_packages, output_root, package_cache_root):
    base = Path(base_site_packages).resolve(strict=True)
    output = Path(output_root)
    cache_root = Path(package_cache_root)
    if not output.is_absolute() or not cache_root.is_absolute():
        raise ValueError("output and package-cache roots must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh output root required; nothing will be overwritten")
    output = output.parent.resolve(strict=True) / output.name
    if output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("input and output trees must be disjoint")
    resolved_cache = cache_root.resolve()
    if output.is_relative_to(resolved_cache) or resolved_cache.is_relative_to(output):
        raise ValueError("package cache is a read-only input, not an output location")
    if any(part.startswith("pip_") for part in output.parts):
        raise ValueError("output ancestors must not override the planned pip_* variant")
    package = base / "deep_gemm"
    if not package.is_dir() or package.is_symlink() or (package / MANIFEST).exists():
        raise ValueError("require the original, unadapted DeepGEMM package")
    for name, (digest, size, mode) in PINNED_FILES.items():
        path = package / name
        if (
            path.is_symlink() or not path.is_file() or _sha(path) != digest
            or path.stat().st_size != size or stat.S_IMODE(path.stat().st_mode) != mode
        ):
            raise ValueError(f"qualified original identity mismatch: {name}")
    header_sha = _headers(package)
    if not header_sha.startswith(HEADER_PREFIX):
        raise ValueError("original DeepGEMM headers differ from the b01 cache identity")
    info = _distribution(base)
    before = _tree(package)
    metadata_before = _tree(info)
    adjacent_libraries = {}
    # Match the existing package copier's two optional native-library siblings.
    for name in ("deep_gemm.libs", "deep_gemm_libs"):
        path = base / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise ValueError("unexpected DeepGEMM adjacent-library layout")
            adjacent_libraries[name] = _tree(path)
    after = dict(before)
    after["utils/layout.py"] = {"sha256": LAYOUT_AFTER_SHA, "bytes": 666, "mode": 0o644}
    cache_state = _verify_cache(cache_root, after, header_sha, adjacent_libraries)
    layout = (package / "utils/layout.py").read_bytes() + APPEND
    if hashlib.sha256(layout).hexdigest() != LAYOUT_AFTER_SHA:
        raise ValueError("host re-export does not match the approved exact bytes")
    # All validation precedes creation. Failure preserves any partial new output;
    # this preparer never removes, repairs or retries an existing directory.
    output.mkdir()
    site = output / VARIANT / "site-packages"
    site.mkdir(parents=True)
    copied = site / "deep_gemm"
    shutil.copytree(package, copied, symlinks=True)
    if info.is_dir():
        shutil.copytree(info, site / info.name, symlinks=True)
    else:
        shutil.copy2(info, site / info.name)
    for name in adjacent_libraries:
        shutil.copytree(base / name, site / name, symlinks=True)
    (copied / "utils/layout.py").write_bytes(layout)
    if (
        _tree(copied) != after or _tree(site / info.name) != metadata_before
        or _tree(package) != before or _tree(info) != metadata_before
        or _headers(copied) != header_sha or _distribution(site).name != info.name
    ):
        raise ValueError("copy verification failed; preserve output for diagnosis")
    for name, files in adjacent_libraries.items():
        if _tree(site / name) != files or _tree(base / name) != files:
            raise ValueError("adjacent-library copy changed original bytes")
    receipt = {
        "schema": SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "source_package": str(package),
        "source_metadata": str(info),
        "version": VERSION,
        "metadata_basename": info.name,
        "metadata_files": metadata_before,
        "adjacent_libraries": adjacent_libraries,
        "layout_before_sha256": LAYOUT_BEFORE_SHA,
        "layout_after_sha256": LAYOUT_AFTER_SHA,
        "native_sha256": NATIVE_SHA,
        "header_sha256": header_sha,
        "variant": VARIANT,
        "cache_key": CACHE_KEY,
        "package_cache_root": str(cache_root),
        "cache_state_before": cache_state,
        "site_packages": str(site),
        "package_files": after,
        "delta": ["utils/layout.py", MANIFEST],
    }
    manifest = copied / MANIFEST
    manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {
        "classification": "SOURCE_CHECK_PREPARED_ARTIFACT_ONLY",
        "site_packages": str(site),
        "manifest": str(manifest),
        "manifest_sha256": _sha(manifest),
        "cache_key": CACHE_KEY,
        "cache_state_before": cache_state,
        "native_sha256": _sha(copied / NATIVE),
        "original_package_and_metadata_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-site-packages", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--package-cache-root", default=str(Path.home() / ".cache"))
    args = parser.parse_args()
    result = prepare_shared_dependencies(
        args.base_site_packages, args.output_root, args.package_cache_root
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
