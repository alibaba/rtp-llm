#!/usr/bin/env python3
"""Static consistency check for the duplicated CUDA 13 wheel declarations.

The CUDA 13 wheel direct links and version pins are, by current design, kept in
several places until a generated manifest can serve them all (see the comments
in arch_config/arch_select.bzl and bazel/cuda13_packages.bzl):

  * whl_deps() in arch_config/arch_select.bzl  -> wheel metadata embedded in the
    produced wheel (a *subset*: only ABI-bound wheels).
  * deps/requirements_torch_gpu_cuda13.txt     -> x86 pip resolver input.
  * deps/requirements_cuda13_arm.txt           -> ARM pip resolver input.
  * CUDA13_EXPECTED_DEPENDENCY_VERSIONS in bazel/cuda13_packages.bzl -> versions
    the runtime smoke test asserts against.

These sets overlap but are intentionally *not* equal, so this check compares the
*intersection*: for any package that appears in two sources, the URL / pinned
version must agree. It does NOT require the sets to be identical.

Generated locks (requirements_lock_*cuda13*.txt) are derived from the source
requirements and validated by their own compile_pip_requirements `_test`, so
they are deliberately out of scope here.

Exit code 0 on success, 1 on any mismatch (with a human-readable diff).
Pure standard library so it can run on any CI lane without a GPU or Bazel.
"""

import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARCH_SELECT = os.path.join(_REPO_ROOT, "arch_config", "arch_select.bzl")
_CUDA13_PACKAGES = os.path.join(_REPO_ROOT, "bazel", "cuda13_packages.bzl")
_REQ_X86 = os.path.join(_REPO_ROOT, "deps", "requirements_torch_gpu_cuda13.txt")
_REQ_ARM = os.path.join(_REPO_ROOT, "deps", "requirements_cuda13_arm.txt")


def _normalize(name):
    # PEP 503-ish normalization plus unifying '_' and '-' so that wheel file
    # names (deep_gemm, flash_mla) match distribution names (deep-gemm, ...).
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _dist_from_wheel_url(url):
    # ".../deep_gemm-2.5.0%2B8a4dfba-cp310-...-linux_x86_64.whl" -> "deep_gemm"
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    return filename.split("-", 1)[0]


def _spec_from_token(token):
    """Return (normalized_name, kind, value) for a whl_deps token.

    Tokens are either "name@url" or "name==version".
    """
    token = token.strip()
    if "@" in token and "==" not in token.split("@", 1)[0]:
        name, url = token.split("@", 1)
        return _normalize(name), "url", url.strip()
    if "==" in token:
        name, ver = token.split("==", 1)
        return _normalize(name), "version", ver.strip()
    # Bare name with no constraint: nothing to compare.
    return _normalize(token), "bare", None


def _parse_requirements(path):
    """Parse a pip requirements file into {normalized_name: (kind, value)}."""
    specs = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                # Comment, blank, or a flag line (-r/-c/--index...).
                continue
            if line.startswith("http://") or line.startswith("https://"):
                specs[_normalize(_dist_from_wheel_url(line))] = ("url", line)
            elif "@" in line and "==" not in line.split("@", 1)[0]:
                name, url = line.split("@", 1)
                specs[_normalize(name)] = ("url", url.strip())
            elif "==" in line:
                name, ver = line.split("==", 1)
                # Strip any trailing environment markers / extras.
                ver = ver.split(";", 1)[0].strip()
                specs[_normalize(name)] = ("version", ver)
    return specs


def _parse_whl_deps_branch(text, config_setting):
    """Extract the list of tokens under a select() branch in arch_select.bzl."""
    marker = '"@rtp_llm//:%s":' % config_setting
    start = text.find(marker, text.find("def whl_deps"))
    if start == -1:
        raise RuntimeError("could not locate whl_deps branch for %s" % config_setting)
    open_bracket = text.find("[", start)
    close_bracket = text.find("]", open_bracket)
    body = text[open_bracket + 1 : close_bracket]
    specs = {}
    for token in re.findall(r'"([^"]+)"', body):
        name, kind, value = _spec_from_token(token)
        if kind != "bare":
            specs[name] = (kind, value)
    return specs


def _parse_expected_versions(text):
    """Parse CUDA13_EXPECTED_DEPENDENCY_VERSIONS = {"arch": {"pkg": "ver"}}."""
    result = {}
    for arch in ("x86", "arm"):
        block = re.search(r'"%s"\s*:\s*\{(.*?)\}' % arch, text, re.DOTALL)
        if not block:
            continue
        pins = {}
        for name, ver in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', block.group(1)):
            pins[_normalize(name)] = ver
        result[arch] = pins
    return result


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _compare_intersection(errors, label_a, a, label_b, b):
    for name in sorted(set(a) & set(b)):
        kind_a, val_a = a[name]
        kind_b, val_b = b[name]
        if kind_a != kind_b or val_a != val_b:
            errors.append(
                "  %s:\n      %s -> %s=%s\n      %s -> %s=%s"
                % (name, label_a, kind_a, val_a, label_b, kind_b, val_b)
            )


def main(argv):
    # Bazel passes the four files as $(location ...) args (paths relative to the
    # runfiles root). Run standalone with no args to use the in-tree defaults.
    if len(argv) == 4:
        arch_select, cuda13_packages, req_x86_path, req_arm_path = argv
    elif not argv:
        arch_select, cuda13_packages, req_x86_path, req_arm_path = (
            _ARCH_SELECT,
            _CUDA13_PACKAGES,
            _REQ_X86,
            _REQ_ARM,
        )
    else:
        sys.stderr.write(
            "usage: check_cuda13_wheel_consistency.py "
            "[arch_select.bzl cuda13_packages.bzl "
            "requirements_torch_gpu_cuda13.txt requirements_cuda13_arm.txt]\n"
        )
        return 2

    arch_text = _read(arch_select)
    pkgs_text = _read(cuda13_packages)

    whl_x86 = _parse_whl_deps_branch(arch_text, "using_cuda13_x86")
    whl_arm = _parse_whl_deps_branch(arch_text, "using_cuda13_arm")
    req_x86 = _parse_requirements(req_x86_path)
    req_arm = _parse_requirements(req_arm_path)
    expected = _parse_expected_versions(pkgs_text)

    errors = []

    # (1) whl_deps() direct links / pins must match the source requirements for
    #     every package that appears in both.
    _compare_intersection(
        errors, "whl_deps[x86]", whl_x86, "requirements_torch_gpu_cuda13", req_x86
    )
    _compare_intersection(
        errors, "whl_deps[arm]", whl_arm, "requirements_cuda13_arm", req_arm
    )

    # (2) CUDA13_EXPECTED_DEPENDENCY_VERSIONS pins must match the source
    #     requirements pins (nvidia-cutlass-dsl / apache-tvm-ffi).
    for arch, req in (("x86", req_x86), ("arm", req_arm)):
        pkgs = expected.get(arch, {})
        as_specs = {name: ("version", ver) for name, ver in pkgs.items()}
        _compare_intersection(
            errors,
            "cuda13_packages[%s]" % arch,
            as_specs,
            "requirements_cuda13_%s" % arch,
            req,
        )

    if errors:
        sys.stderr.write(
            "CUDA 13 wheel declarations are out of sync across sources.\n"
            "Update all copies together (see arch_config/arch_select.bzl "
            "whl_deps(), deps/requirements_*cuda13*.txt, and "
            "bazel/cuda13_packages.bzl):\n" + "\n".join(errors) + "\n"
        )
        return 1

    print("CUDA 13 wheel declarations are consistent across all sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
