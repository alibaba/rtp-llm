#!/usr/bin/env python3
import argparse
import importlib.metadata
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

PACKAGE_ROOTS = (
    "rtp_llm",
    "torch",
    "deep_ep",
    "deep_gemm",
    "flashinfer",
    "flashinfer_jit_cache",
    "flashinfer_cubin",
    "flash_attn",
    "flash_attn_3",
    "flash_mla",
    "fast_hadamard_transform",
    "fast_safetensors",
    "fastsafetensors",
    "rtp_kernel",
)
TOP_LEVEL_PATTERNS = (
    "flash_attn_2_cuda*.so*",
    "fast_hadamard_transform_cuda*.so*",
)
HOST_DRIVER_LIBRARIES = {"libcuda.so.1", "libnvidia-ml.so.1"}
FORBIDDEN_NEEDED = {"visibility=hidden"}
DYNAMIC_TAGS = ("NEEDED", "AUXILIARY", "FILTER")
SEARCH_PATH_TAGS = ("RPATH", "RUNPATH")
BRACKET_VALUE_RE = re.compile(r"\[([^]]*)\]")
CUDA_LIBRARY_RE = re.compile(
    r"^lib(?:cudart|cupti|cublas(?:Lt)?|cudnn(?:_[^.]+)?|nccl|nvrtc|"
    r"nvJitLink|cusolver|cusparse|curand|cufft|nvshmem(?:_[^.]+)?)\.so"
)
UNAMBIGUOUS_CUDA12_RE = re.compile(r"^lib(?:cudart|cupti)\.so\.12(?:\.|$)")
CUDA12_PATH_RE = re.compile(
    r"(?:^|[/+_.-])cu(?:da)?[-_.]?12(?:[0-9]*|[/+_.-]|$)", re.IGNORECASE
)
CUDA12_DIST_RE = re.compile(r"(?:^|[-_.+])cu12(?:[0-9]*|[-_.+]|$)", re.IGNORECASE)


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def elf_identity(path):
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    endian = {1: "<", 2: ">"}.get(header[5])
    if endian is None or struct.unpack(f"{endian}H", header[16:18])[0] not in (2, 3):
        return None
    return header[4], struct.unpack(f"{endian}H", header[18:20])[0]


def iter_files(root, seen_directories, seen_files):
    try:
        root_stat = root.stat()
    except OSError as error:
        fail(f"cannot inspect runtime path {root}: {error}")
    if root.is_file():
        identity = (root_stat.st_dev, root_stat.st_ino)
        if identity not in seen_files:
            seen_files.add(identity)
            yield root.resolve()
        return
    for directory, child_directories, files in os.walk(root, followlinks=True):
        directory_path = Path(directory)
        try:
            stat_result = directory_path.stat()
        except OSError as error:
            fail(f"cannot inspect runtime directory {directory_path}: {error}")
        identity = (stat_result.st_dev, stat_result.st_ino)
        if identity in seen_directories:
            child_directories[:] = []
            continue
        seen_directories.add(identity)
        retained = []
        for child in child_directories:
            child_path = directory_path / child
            try:
                child_stat = child_path.stat()
            except OSError as error:
                fail(f"cannot inspect runtime directory {child_path}: {error}")
            child_identity = (child_stat.st_dev, child_stat.st_ino)
            if child_identity not in seen_directories:
                retained.append(child)
        child_directories[:] = retained
        for name in files:
            path = directory_path / name
            try:
                file_stat = path.stat()
            except OSError as error:
                fail(f"cannot inspect runtime file {path}: {error}")
            identity = (file_stat.st_dev, file_stat.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            yield path.resolve()


def bad_distribution_files(platlib):
    bad_files = {}
    for distribution in importlib.metadata.distributions(path=[str(platlib)]):
        name = str(distribution.metadata.get("Name") or "")
        version = str(distribution.version or "")
        if not CUDA12_DIST_RE.search(name) and not CUDA12_DIST_RE.search(version):
            continue
        owner = f"{name}=={version}"
        for relative_path in distribution.files or ():
            path = Path(distribution.locate_file(relative_path))
            try:
                bad_files[path.resolve()] = owner
            except OSError:
                continue
    return bad_files


def under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def find_readelf():
    candidates = (
        "/usr/bin/readelf",
        "/bin/readelf",
        "/usr/bin/eu-readelf",
        "/usr/local/PPU_SDK/bin/llvm-readelf",
    )
    for candidate in candidates:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    resolved = shutil.which("readelf")
    if not resolved:
        fail("required ELF inspection tool is missing: readelf")
    return resolved


def dynamic_values(output, tags):
    values = {tag: [] for tag in tags}
    for line in output.splitlines():
        upper = line.upper()
        value = BRACKET_VALUE_RE.search(line)
        if value is None:
            continue
        for tag in tags:
            if f"({tag})" in upper or re.search(rf"(^|\s){tag}(\s|$)", upper):
                values[tag].append(value.group(1))
                break
    return values


def dynamic_metadata(readelf, inspection_environment, path, platform_name):
    result = subprocess.run(
        [readelf, "-d", str(path)],
        env=inspection_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip())
    needed_by_tag = dynamic_values(result.stdout, DYNAMIC_TAGS)
    search_values = dynamic_values(result.stdout, SEARCH_PATH_TAGS)
    search_paths = {tag: [] for tag in SEARCH_PATH_TAGS}
    for kind, values in search_values.items():
        for value in values:
            for item in value.split(":"):
                item = item.strip().strip("'")
                item = item.replace("${ORIGIN}", str(path.parent))
                item = item.replace("$ORIGIN", str(path.parent))
                item = item.replace("${LIB}", "lib64").replace("$LIB", "lib64")
                item = item.replace("${PLATFORM}", platform_name)
                item = item.replace("$PLATFORM", platform_name)
                if item:
                    search_paths[kind].append(Path(item))
    dependencies = []
    for tag in DYNAMIC_TAGS:
        dependencies.extend(needed_by_tag[tag])
    return dependencies, search_paths


def resolve_library(name, search_paths, expected_identity):
    incompatible = []
    candidates = [Path(name)] if "/" in name else [path / name for path in search_paths]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        provider = candidate.resolve()
        provider_identity = elf_identity(provider)
        if provider_identity == expected_identity:
            return provider, incompatible
        incompatible.append((provider, provider_identity))
    return None, incompatible


def enqueue_provider(pending, inspected_states, provider, inherited_rpath):
    state = (provider, inherited_rpath)
    if state not in inspected_states:
        pending.append(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda-major", type=int, required=True)
    parser.add_argument(
        "--platlib", type=Path, default=Path(sysconfig.get_path("platlib"))
    )
    parser.add_argument("--root", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.expected_cuda_major != 13:
        fail(f"this validator only supports CUDA 13, got {args.expected_cuda_major}")

    readelf = find_readelf()
    inspection_environment = os.environ.copy()
    inspection_environment.pop("LD_LIBRARY_PATH", None)
    platlib = args.platlib.resolve()
    roots = [path.resolve() for path in args.root]
    if not roots:
        roots = [
            (platlib / name).resolve()
            for name in PACKAGE_ROOTS
            if (platlib / name).exists()
        ]
        for pattern in TOP_LEVEL_PATTERNS:
            roots.extend(path.resolve() for path in platlib.glob(pattern))
    if not roots:
        fail("no runtime package roots found")

    environment_library_directories = [
        Path(path) for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
    ]
    default_library_directories = [
        platlib / "rtp_llm/libs",
        platlib / "torch/lib",
        platlib / "nvidia/cu13/lib",
        platlib / "nvidia/nvshmem/lib",
        Path("/opt/conda310/lib"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/targets/aarch64-linux/lib"),
        Path("/usr/local/cuda/targets/sbsa-linux/lib"),
        Path("/usr/local/PPU_SDK/lib"),
        Path("/usr/local/PPU_SDK/CUDA_SDK/lib64"),
        Path("/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib"),
        Path("/usr/local/PPU_SDK/sailSHMEM/lib"),
        Path("/usr/local/lib64"),
        Path("/usr/local/lib"),
        Path("/lib64"),
        Path("/usr/lib64"),
        Path("/lib"),
        Path("/usr/lib"),
    ]
    default_library_directories.extend(platlib.glob("nvidia/*/lib"))
    for base in (Path("/lib"), Path("/usr/lib")):
        if base.is_dir():
            default_library_directories.extend(base.glob("*-linux-gnu"))
    environment_library_directories = [
        path for path in environment_library_directories if path.is_dir()
    ]
    default_library_directories = [
        path for path in default_library_directories if path.is_dir()
    ]

    certified_roots = [platlib / "nvidia", platlib / "torch/lib"]
    for path in (Path("/usr/local/cuda"), Path("/usr/local/PPU_SDK")):
        if path.exists():
            certified_roots.append(path.resolve())
    certified_roots.extend(
        path.resolve() for path in Path("/usr/local").glob("cuda-13*")
    )
    bad_owned_files = bad_distribution_files(platlib)

    seen_directories = set()
    seen_files = set()
    seed_elfs = []
    for root in roots:
        seed_elfs.extend(iter_files(root, seen_directories, seen_files))
    seed_elfs = sorted(path for path in seed_elfs if elf_identity(path) is not None)
    if not seed_elfs:
        fail("runtime package roots contain no dynamic ELF files")

    failures = []
    inspected_states = set()
    inspected_files = set()
    pending = [(path, ()) for path in seed_elfs]
    while pending:
        elf, inherited_rpath = pending.pop()
        state = (elf, inherited_rpath)
        if state in inspected_states:
            continue
        inspected_states.add(state)
        inspected_files.add(elf)
        identity = elf_identity(elf)
        if identity is None:
            failures.append(f"{elf}: resolved provider is not a dynamic ELF")
            continue
        platform_name = {62: "x86_64", 183: "aarch64"}.get(
            identity[1], str(identity[1])
        )
        try:
            needed, object_search_paths = dynamic_metadata(
                readelf, inspection_environment, elf, platform_name
            )
        except RuntimeError as error:
            failures.append(f"{elf}: readelf failed: {error}")
            continue
        current_rpath = tuple(
            dict.fromkeys(
                path for path in object_search_paths["RPATH"] if path.is_dir()
            )
        )
        inherited_rpath = tuple(path for path in inherited_rpath if path.is_dir())
        effective_rpath = tuple(dict.fromkeys(current_rpath + inherited_rpath))
        runpath = [path for path in object_search_paths["RUNPATH"] if path.is_dir()]
        if runpath:
            search_paths = (
                list(inherited_rpath)
                + environment_library_directories
                + runpath
                + default_library_directories
            )
        else:
            search_paths = (
                list(effective_rpath)
                + environment_library_directories
                + default_library_directories
            )
        search_paths = list(dict.fromkeys(search_paths))
        for library_name in needed:
            if library_name in FORBIDDEN_NEEDED:
                failures.append(f"{elf}: forbidden dependency {library_name}")
                continue
            provider, incompatible = resolve_library(
                library_name, search_paths, identity
            )
            if provider is None:
                if incompatible:
                    details = ", ".join(
                        f"{path} ({provider_identity})"
                        for path, provider_identity in incompatible
                    )
                    failures.append(
                        f"{elf}: dependency {library_name} has incompatible providers: {details}"
                    )
                elif library_name not in HOST_DRIVER_LIBRARIES:
                    failures.append(f"{elf}: unresolved dependency {library_name}")
                continue
            owner = bad_owned_files.get(provider)
            if (
                UNAMBIGUOUS_CUDA12_RE.match(library_name)
                or CUDA12_PATH_RE.search(str(provider))
                or owner is not None
            ):
                owner_suffix = f" ({owner})" if owner else ""
                failures.append(
                    f"{elf}: CUDA 12 dependency {library_name} => {provider}{owner_suffix}"
                )
                continue
            if CUDA_LIBRARY_RE.match(library_name) and not any(
                under(provider, root.resolve()) for root in certified_roots
            ):
                failures.append(
                    f"{elf}: CUDA provider is outside certified CUDA 13 roots: "
                    f"{library_name} => {provider}"
                )
            enqueue_provider(pending, inspected_states, provider, effective_rpath)

    if failures:
        print("ERROR: runtime ELF closure validation failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"validated runtime ELF closure for {len(inspected_files)} dynamic object(s) "
        f"from {len(seed_elfs)} seed(s)"
    )


if __name__ == "__main__":
    main()
