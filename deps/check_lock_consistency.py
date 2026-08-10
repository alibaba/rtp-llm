#!/usr/bin/env python3
"""Check consistency between pip requirements sources and their lockfiles.

For each platform (cuda12_9 / cuda12_arm / rocm) this script verifies that:
  1. Every explicit `name==version` pin in the source requirements file
     (including pins pulled in via `-r requirements_base.txt`) appears in the
     corresponding lockfile with the exact same version (PEP 440 zero-padding
     tolerated, e.g. `4.25` == `4.25.0`).
  2. The lockfile contains no unexpected direct URLs (`@ https`). Only the
     ROCm lockfile is allowed exactly one: amdsmi ships as a tar (not a
     wheel), so it cannot be served through the PEP 503 simple index and must
     stay a direct URL (see arch_config/arch_select.bzl whl_deps()).

Standard library only, no network access. Exit code is non-zero if any
platform fails. Run from anywhere: paths are resolved relative to this file.
"""

import os
import re
import sys
from urllib.parse import unquote

DEPS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DEPS_DIR)

# (platform name, source requirements, lockfile, platform_machine of the lock)
PLATFORMS = [
    (
        "cuda12_9",
        "requirements_torch_gpu_cuda12_9.txt",
        "requirements_lock_torch_gpu_cuda12_9.txt",
        "x86_64",
    ),
    (
        "cuda12_arm",
        "requirements_cuda12_arm.txt",
        "requirements_lock_cuda12_arm.txt",
        "aarch64",
    ),
    (
        "rocm",
        "requirements_rocm.txt",
        "requirements_lock_rocm.txt",
        "x86_64",
    ),
]

# Allowed `@ https` direct-URL count per lockfile. rocm is exactly 1 because
# amdsmi is distributed as a tar archive which a PEP 503 index cannot serve;
# everything else must resolve through the configured indexes.
ALLOWED_DIRECT_URLS = {
    "cuda12_9": 0,
    "cuda12_arm": 0,
    "rocm": 1,
}

EXPECTED_INDEX_LINES = {
    "cuda12_9": [
        "--index-url https://mirrors.aliyun.com/pypi/simple/",
        "--extra-index-url https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/",
        "--extra-index-url https://download.pytorch.org/whl/cu129/",
    ],
    "cuda12_arm": [
        "--index-url https://mirrors.aliyun.com/pypi/simple/",
        "--extra-index-url https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/",
        "--extra-index-url https://download.pytorch.org/whl/cu129/",
    ],
    "rocm": [
        "--index-url https://mirrors.aliyun.com/pypi/simple/",
        "--extra-index-url https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/",
    ],
}

# These pins are intentionally an allowlist. With uv's unsafe-best-match index
# strategy, changing a source pin can change which index supplies an artifact.
# Requiring a simultaneous checker update makes custom-wheel upgrades explicit
# and reviewable instead of accepting a relock-only provenance change.
EXPECTED_CRITICAL_PINS = {
    "cuda12_9": {
        "deep-ep": "1.2.1.10+d7d7b48",
        "deep-gemm": "2.1.1+local",
        "fast-safetensors": "0.7.3+torch2.1.2.cu121",
        "flash-attn": "2.8.3+cu12torch2.8cxx11abiTRUE",
        "flash-mla": "1.0.0+ca58fed",
        "flashinfer-cubin": "0.6.9",
        "flashinfer-jit-cache": "0.6.9",
        "flashinfer-python": "0.6.9",
        "nvidia-cutlass-dsl": "4.4.2",
        "rtp-kernel": "0.1.0+125c29e5.20260422154605",
        "torch": "2.8.0+cu129",
        "torchvision": "0.23.0+cu129",
    },
    "cuda12_arm": {
        "deep-ep": "1.2.1.11+unknown.pai",
        "deep-gemm": "2.1.1+local",
        "fast-safetensors": "0.7.3+torch2.1.2.cu121",
        "flash-attn": "2.8.3",
        "flash-mla": "1.0.0+47c35a7",
        "flashinfer-cubin": "0.6.6",
        "flashinfer-jit-cache": "0.6.6",
        "flashinfer-python": "0.6.6",
        "nvidia-cutlass-dsl": "4.4.2",
        "rtp-kernel": "0.1.0+125c29e5.20260422155252",
        "torch": "2.9.0+cu129",
        "torchvision": "0.24.0+cu129",
    },
    "rocm": {
        "aiter": "0.1.17.dev79+g2570b35f9.d20260623",
        "fast-safetensors": "0.7.3+torch2.1.2.rocm",
        "torch": "2.9.1+git7e1940d",
        "torchvision": "0.24.0+rocm7.2.0.gitb919bd0c",
        "triton": "3.7.0+amd.rocm7.2.0.gitd0d77a509",
        "triton-kernels": "1.0.0+amd.rocm7.2.0.gitd0d77a509",
    },
}

# CUDA packages in this set are shared artifacts or have matching per-arch
# wheels and therefore must keep the exact same version on x86_64 and aarch64.
# Packages blocked on unpublished ARM artifacts are intentionally not listed;
# their source requirements carry the concrete publication TODO.
CUDA_EXACT_VERSION_PARITY = {
    "deep-gemm",
    "fast-hadamard-transform",
    "fast-safetensors",
    "fastsafetensors",
    "flash-attn-3",
    "nvidia-cutlass-dsl",
}

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([^\]]+)\])?\s*==\s*(\S+)$")
DIRECT_URL_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?\s*@\s*https?://"
)
MACHINE_RE = re.compile(r"platform_machine\s*(==|!=)\s*['\"]([^'\"]+)['\"]")


def normalize_name(name):
    """PEP 503 name normalization: lowercase, -/_/. are equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


def canon_version(version):
    """Canonicalize a version for comparison.

    - strips trailing `.0` release segments (PEP 440: 4.25 == 4.25.0)
    - lowercases pre/post/dev suffixes and the local tag (case-insensitive
      per PEP 440)
    """
    if "+" in version:
        public, local = version.split("+", 1)
        local = "+" + local.lower()
    else:
        public, local = version, ""
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", public)
    if not match:
        return version.lower()
    release, suffix = match.groups()
    parts = release.split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts) + suffix.lower() + local


def marker_applies(marker, machine):
    """Lightweight platform_machine marker evaluation.

    Only `platform_machine ==/!= "..."` clauses are interpreted (the only
    marker kind used in deps/requirements*.txt). Unknown markers are treated
    as applicable so a mismatch is reported rather than silently skipped.
    """
    for op, value in MACHINE_RE.findall(marker):
        if op == "==" and value != machine:
            return False
        if op == "!=" and value == machine:
            return False
    return True


def strip_comment(line):
    """Drop full-line and trailing ` # ...` comments."""
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line).strip()


def requirement_include(line):
    """Return an included requirements path, or None for a non-include line."""
    if line in ("-r", "--requirement"):
        raise ValueError("requirement include is missing a path")
    if line.startswith("-r"):
        ref = line[2:].strip()
        if not ref:
            raise ValueError("requirement include is missing a path")
        return ref
    if line.startswith("--requirement="):
        ref = line.split("=", 1)[1].strip()
        if not ref:
            raise ValueError("requirement include is missing a path")
        return ref
    if line.startswith("--requirement "):
        ref = line.split(None, 1)[1].strip()
        if not ref:
            raise ValueError("requirement include is missing a path")
        return ref
    return None


def parse_source_pins(path, machine, seen=None):
    """Return {normalized_name: (raw_name, version)} of explicit == pins."""
    if seen is None:
        seen = set()
    real = os.path.realpath(path)
    if real in seen:
        return {}
    seen.add(real)

    pins = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = strip_comment(raw)
            if not line:
                continue
            try:
                ref = requirement_include(line)
            except ValueError as exc:
                raise ValueError("%s:%d: %s" % (path, lineno, exc)) from exc
            if ref is not None:
                ref_path = os.path.join(os.path.dirname(path), ref)
                pins.update(parse_source_pins(ref_path, machine, seen))
                continue
            if line.startswith("-"):  # other pip options (-c, --hash, ...)
                continue
            req, _, marker = line.partition(";")
            req = req.strip()
            if marker and not marker_applies(marker, machine):
                continue
            if " @ " in req or "@ http" in req:
                continue  # direct URLs handled by the @ https count check
            match = PIN_RE.match(req)
            if match:
                name, _, version = match.groups()
                pins[normalize_name(name)] = (name, version)
            elif "==" in req:
                raise ValueError(
                    "%s:%d: malformed exact pin is not understood: %s"
                    % (path, lineno, req)
                )
    return pins


def parse_lock_versions(path):
    """Return (versions, direct URL count, hashes) for a lockfile."""
    versions = {}
    hashes = {}
    direct_urls = 0
    current_name = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            clean = strip_comment(raw)
            if re.search(r"@\s*https?://", clean):
                direct_urls += 1
            line = raw.rstrip("\n")
            if not line or line[0] in "# \t-":
                hash_match = re.search(r"--hash=sha256:([0-9a-f]{64})", line)
                if hash_match and current_name:
                    hashes.setdefault(current_name, set()).add(hash_match.group(1))
                continue  # comments, hash/via continuations, index options
            line = line.rstrip("\\").strip()
            req = line.partition(";")[0].strip()
            match = PIN_RE.match(req)
            if match:
                name, _, version = match.groups()
                current_name = normalize_name(name)
                versions[current_name] = version
                continue
            direct_match = DIRECT_URL_RE.match(req)
            current_name = (
                normalize_name(direct_match.group(1)) if direct_match else None
            )
    return versions, direct_urls, hashes


def check_lock_metadata(platform, lock_path):
    errors = []
    with open(lock_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if not text.startswith(
        "# This file was autogenerated by uv via the following command:\n"
    ):
        errors.append(
            "%s: lockfile must retain its uv-generated header"
            % os.path.basename(lock_path)
        )
    command_line = text.splitlines()[1] if len(text.splitlines()) > 1 else ""
    for required in (
        "--generate-hashes",
        "--emit-index-url",
        "--index-strategy unsafe-best-match",
    ):
        if required not in command_line:
            errors.append(
                "%s: uv command header is missing %s"
                % (os.path.basename(lock_path), required)
            )
    actual_indexes = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("--index-url ") or line.startswith("--extra-index-url ")
    ]
    if actual_indexes != EXPECTED_INDEX_LINES[platform]:
        errors.append(
            "%s: index header mismatch: got %r, expected %r"
            % (
                os.path.basename(lock_path),
                actual_indexes,
                EXPECTED_INDEX_LINES[platform],
            )
        )
    if platform == "cuda12_arm" and "--only-binary flash-attn" not in text:
        errors.append(
            "%s: ARM lock must enforce --only-binary flash-attn to prevent an sdist fallback"
            % os.path.basename(lock_path)
        )
    if platform == "rocm" and (
        "amd_smi.tar#sha256=8a350c562cf6c63d562eef27b2511a8ea67adc056662c6cde85ab1312ecb22ff"
        not in text
    ):
        errors.append(
            "%s: amdsmi direct URL is missing its sha256 fragment"
            % os.path.basename(lock_path)
        )
    return errors


def check_platform(platform, source, lock, machine):
    errors = []
    source_path = os.path.join(DEPS_DIR, source)
    lock_path = os.path.join(DEPS_DIR, lock)

    try:
        pins = parse_source_pins(source_path, machine)
    except (OSError, ValueError) as exc:
        return 0, [str(exc)]
    lock_versions, direct_urls, _ = parse_lock_versions(lock_path)

    for norm_name in sorted(pins):
        raw_name, want = pins[norm_name]
        if norm_name not in lock_versions:
            errors.append(
                "%s: pinned %s==%s in %s but missing from %s"
                % (raw_name, raw_name, want, source, lock)
            )
            continue
        got = lock_versions[norm_name]
        if canon_version(want) != canon_version(got):
            errors.append(
                "%s: source pins ==%s but lockfile has ==%s" % (raw_name, want, got)
            )

    allowed = ALLOWED_DIRECT_URLS[platform]
    if direct_urls != allowed:
        errors.append(
            "%s: found %d direct '@ http(s)' URL(s), expected exactly %d"
            % (lock, direct_urls, allowed)
        )

    errors.extend(check_lock_metadata(platform, lock_path))

    for name, expected in EXPECTED_CRITICAL_PINS[platform].items():
        source_pin = pins.get(name)
        if source_pin is None:
            errors.append(
                "%s: critical package %s is not explicitly pinned" % (source, name)
            )
        elif canon_version(source_pin[1]) != canon_version(expected):
            errors.append(
                "%s: critical pin %s changed from allowlisted %s to %s; update the allowlist deliberately"
                % (source, name, expected, source_pin[1])
            )

    return len(pins), errors


def extract_select_requirements(text, key):
    match = re.search(r'"%s"\s*:\s*\[(.*?)\]\s*,?' % re.escape(key), text, re.DOTALL)
    if not match:
        raise ValueError("missing select branch %s" % key)
    return re.findall(r'"([^"\n]+)"', match.group(1))


def check_whl_deps(platform_pins):
    errors = []
    path = os.path.join(ROOT_DIR, "arch_config", "arch_select.bzl")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    whl_section = text.split("def whl_deps():", 1)[1].split("def platform_deps():", 1)[
        0
    ]
    branches = {
        "cuda12_9": "//conditions:default",
        "cuda12_arm": "@rtp_llm//:using_cuda12_arm",
        "rocm": "@rtp_llm//:using_rocm",
    }
    expected_names = {
        "cuda12_9": {"torch", "torchvision"},
        "cuda12_arm": {"torch", "torchvision"},
        "rocm": {
            "aiter",
            "fast-safetensors",
            "fastsafetensors",
            "pyrsmi",
            "torch",
            "torchvision",
            "triton",
            "triton-kernels",
        },
    }
    for platform, key in branches.items():
        try:
            requirements = extract_select_requirements(whl_section, key)
        except ValueError as exc:
            errors.append("arch_config/arch_select.bzl: %s" % exc)
            continue
        seen_names = set()
        for requirement in requirements:
            match = PIN_RE.match(requirement)
            if not match:
                continue
            name, _, version = match.groups()
            norm_name = normalize_name(name)
            seen_names.add(norm_name)
            lock_version = platform_pins[platform].get(norm_name)
            if lock_version is None or canon_version(version) != canon_version(
                lock_version
            ):
                errors.append(
                    "arch_select whl_deps %s %s==%s does not match its lockfile (%s)"
                    % (platform, name, version, lock_version or "missing")
                )
        missing = expected_names[platform] - seen_names
        if missing:
            errors.append(
                "arch_select whl_deps %s is missing %s" % (platform, sorted(missing))
            )

    amdsmi_hash = "8a350c562cf6c63d562eef27b2511a8ea67adc056662c6cde85ab1312ecb22ff"
    compact = re.sub(r"\s+", "", whl_section)
    if "amdsmi@https://" not in compact or "#sha256=" + amdsmi_hash not in compact:
        errors.append(
            "arch_select whl_deps amdsmi URL must carry its allowlisted sha256 fragment"
        )
    return errors


def check_http_torch_archives(platform_pins, platform_hashes):
    errors = []
    path = os.path.join(DEPS_DIR, "http.bzl")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    archives = {
        "torch_2.8_py310_cuda": "cuda12_9",
        "torch_2.9_py310_cuda_arm": "cuda12_arm",
        "torch_rocm": "rocm",
    }
    for archive, platform in archives.items():
        match = re.search(
            r'http_archive\(\s*name\s*=\s*"%s",(.*?)\n\s*\)' % re.escape(archive),
            text,
            re.DOTALL,
        )
        if not match:
            errors.append("deps/http.bzl: missing %s" % archive)
            continue
        body = match.group(1)
        sha_match = re.search(r'sha256\s*=\s*"([0-9a-f]{64})"', body)
        url_match = re.search(r'"([^"\n]*torch-[^"\n]+-cp310[^"\n]+\.whl)"', body)
        if not sha_match or not url_match:
            errors.append("deps/http.bzl: cannot parse %s torch URL/sha256" % archive)
            continue
        filename = unquote(url_match.group(1).rsplit("/", 1)[-1])
        version_match = re.match(r"torch-(.+?)-cp310", filename)
        version = version_match.group(1) if version_match else None
        expected_version = platform_pins[platform].get("torch")
        if version is None or canon_version(version) != canon_version(expected_version):
            errors.append(
                "deps/http.bzl %s version %s does not match %s lock torch==%s"
                % (archive, version, platform, expected_version)
            )
        if sha_match.group(1) not in platform_hashes[platform].get("torch", set()):
            errors.append(
                "deps/http.bzl %s sha256 is absent from the %s lock"
                % (archive, platform)
            )
    return errors


def check_whl_reqs(base_pins):
    errors = []
    path = os.path.join(ROOT_DIR, "rtp_llm", "BUILD")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r"whl_reqs\s*=\s*\[(.*?)\]\s*\+\s*whl_deps", text, re.DOTALL)
    if not match:
        return ["rtp_llm/BUILD: cannot parse whl_reqs"]
    for value in re.findall(r'"([^"\n]+)"', match.group(1)):
        pin = PIN_RE.match(value)
        if not pin:
            continue
        name, _, version = pin.groups()
        source = base_pins.get(normalize_name(name))
        if source and canon_version(version) != canon_version(source[1]):
            errors.append(
                "rtp_llm/BUILD whl_reqs %s==%s differs from requirements_base.txt ==%s"
                % (name, version, source[1])
            )
    return errors


def check_cross_platform_invariants(platform_pins):
    errors = []
    for name in sorted(CUDA_EXACT_VERSION_PARITY):
        versions = [platform_pins[p].get(name) for p in ("cuda12_9", "cuda12_arm")]
        if None in versions or canon_version(versions[0]) != canon_version(versions[1]):
            errors.append(
                "%s x86/arm pins must have the same version: %r" % (name, versions)
            )

    versions = [platform_pins[p].get("rtp-kernel") for p in ("cuda12_9", "cuda12_arm")]
    commits = []
    for version in versions:
        match = re.search(r"\+([0-9a-f]{7,40})\.", version or "")
        commits.append(match.group(1) if match else None)
    if None in commits or commits[0] != commits[1]:
        errors.append(
            "rtp-kernel x86/arm pins must carry the same git commit prefix: %r"
            % versions
        )
    return errors


def check_static_metadata():
    errors = []
    platform_pins = {}
    platform_hashes = {}
    for platform, source, lock, machine in PLATFORMS:
        source_path = os.path.join(DEPS_DIR, source)
        lock_path = os.path.join(DEPS_DIR, lock)
        source_entries = parse_source_pins(source_path, machine)
        platform_pins[platform] = {
            name: value[1] for name, value in source_entries.items()
        }
        _, _, platform_hashes[platform] = parse_lock_versions(lock_path)
    base_pins = parse_source_pins(
        os.path.join(DEPS_DIR, "requirements_base.txt"), "x86_64"
    )
    errors.extend(check_whl_deps(platform_pins))
    errors.extend(check_http_torch_archives(platform_pins, platform_hashes))
    errors.extend(check_whl_reqs(base_pins))
    errors.extend(check_cross_platform_invariants(platform_pins))
    expected_arm_flash_attn_hashes = {
        "9587c44a8adb6af4f83553f3147bdd7daa3c28a00a9be0ef97a945659f2885b5"
    }
    if not expected_arm_flash_attn_hashes.issubset(
        platform_hashes["cuda12_arm"].get("flash-attn", set())
    ):
        errors.append(
            "ARM flash-attn lock hashes are missing the allowlisted aarch64 wheel"
        )
    pip_bzl = os.path.join(DEPS_DIR, "pip.bzl")
    with open(pip_bzl, "r", encoding="utf-8") as fh:
        pip_text = fh.read()
    for required in (
        '"--platform=manylinux2014_aarch64"',
        '"--platform=manylinux_2_17_aarch64"',
        '"--platform=manylinux_2_25_aarch64"',
        '"--platform=manylinux_2_27_aarch64"',
        '"--platform=manylinux_2_28_aarch64"',
        '"--platform=linux_aarch64"',
        '"--python-version=3.10"',
        '"--only-binary=:all:"',
        "download_only = True",
        "extra_pip_args = PIP_CUDA_ARM_EXTRA_ARGS",
    ):
        if required not in pip_text:
            errors.append(
                "deps/pip.bzl: ARM cross-platform resolution is missing %s" % required
            )
    return errors


def main():
    failed = False
    for platform, source, lock, machine in PLATFORMS:
        pin_count, errors = check_platform(platform, source, lock, machine)
        if errors:
            failed = True
            print("[FAIL] %s (%d pins checked)" % (platform, pin_count))
            for err in errors:
                print("       - %s" % err)
        else:
            print("[PASS] %s (%d pins checked)" % (platform, pin_count))
    try:
        metadata_errors = check_static_metadata()
    except (OSError, ValueError) as exc:
        metadata_errors = [str(exc)]
    if metadata_errors:
        failed = True
        print("[FAIL] build metadata consistency")
        for err in metadata_errors:
            print("       - %s" % err)
    else:
        print("[PASS] build metadata consistency")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
