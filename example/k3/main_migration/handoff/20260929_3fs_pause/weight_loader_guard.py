#!/usr/bin/env python3
"""Read-only RTP-LLM checkpoint and FastSafetensors guard."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat as stat_module
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


NETWORK_FS_TYPES = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.ceph",
    "fuse.glusterfs",
    "fuse.sshfs",
    "glusterfs",
    "lustre",
    "nfs",
    "nfs4",
    "smb",
    "smb2",
    "smb3",
    "smbfs",
    "sshfs",
}
DEFAULT_DATA_ROOT_RE = re.compile(r"^/data(?:[0-9]+)?(?:/|$)")
NAS_PATH_RE = re.compile(r"^/mnt/nas(?:[0-9A-Za-z_.-]*)?(?:/|$)", re.IGNORECASE)
SHARD_NAME_RE = re.compile(r"-(\d+)-of-(\d+)\.safetensors$")
MAX_HEADER_BYTES = 256 * 1024 * 1024
DEFAULT_COPY_RESERVE_BYTES = 10 * 1024 * 1024 * 1024
LOAD_METHOD_KEYS = {"load_method", "loadmethod"}


class GuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class MountInfo:
    source: str
    target: str
    fstype: str
    options: str
    stat_fs_type: str
    device: str
    inode: int
    kind: str
    size_bytes: int
    used_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class HeaderInfo:
    tensor_names: frozenset[str]
    header_bytes: int
    payload_bytes: int


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def passed(self, message: str) -> None:
        print(f"[PASS] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")


def run_readonly(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GuardError(f"required command not found: {command[0]}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise GuardError(f"{' '.join(command)} failed: {detail}")
    return result.stdout


def resolve_existing(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GuardError(f"cannot resolve existing path {path}: {exc}") from exc


def mount_info(path: Path) -> MountInfo:
    resolved = resolve_existing(path)
    raw_mount = run_readonly(
        ["findmnt", "-T", str(resolved), "-J", "-o", "SOURCE,TARGET,FSTYPE,OPTIONS"]
    )
    try:
        filesystems = json.loads(raw_mount)["filesystems"]
        mount = filesystems[0]
        source = str(mount["source"])
        target = str(mount["target"])
        fstype = str(mount["fstype"])
        options = str(mount.get("options", ""))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise GuardError(f"unexpected findmnt output for {resolved}") from exc

    stat_line = run_readonly(
        ["stat", "-L", "-c", "%d\t%i\t%F", str(resolved)]
    ).rstrip("\n")
    try:
        device, inode_text, kind = stat_line.split("\t", 2)
        inode = int(inode_text)
    except (ValueError, TypeError) as exc:
        raise GuardError(f"unexpected stat output for {resolved}: {stat_line!r}") from exc
    stat_fs_type = run_readonly(
        ["stat", "-f", "-L", "-c", "%T", str(resolved)]
    ).strip()

    df_lines = run_readonly(
        ["df", "-B1", "--output=size,used,avail", str(resolved)]
    ).strip().splitlines()
    try:
        size_bytes, used_bytes, available_bytes = map(int, df_lines[-1].split())
    except (IndexError, ValueError) as exc:
        raise GuardError(f"unexpected df output for {resolved}") from exc

    return MountInfo(
        source=source,
        target=target,
        fstype=fstype,
        options=options,
        stat_fs_type=stat_fs_type,
        device=device,
        inode=inode,
        kind=kind,
        size_bytes=size_bytes,
        used_bytes=used_bytes,
        available_bytes=available_bytes,
    )


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def default_local_data_path(path: Path) -> bool:
    return bool(DEFAULT_DATA_ROOT_RE.match(str(path)))


def mount_rejection_reasons(path: Path, info: MountInfo) -> list[str]:
    reasons: list[str] = []
    fstype = info.fstype.strip().lower()
    if fstype in NETWORK_FS_TYPES or fstype.startswith("nfs") or "sshfs" in fstype:
        reasons.append(f"network filesystem type {info.fstype!r}")
    if "nas" in info.source.lower():
        reasons.append(f"mount source contains 'nas': {info.source!r}")
    if NAS_PATH_RE.match(str(path)) or NAS_PATH_RE.match(info.target):
        reasons.append(f"NAS path or mount target: path={path}, target={info.target}")
    return reasons


def validate_local_path(
    path: Path,
    info: MountInfo,
    explicit_roots: tuple[Path, ...],
) -> list[str]:
    reasons = mount_rejection_reasons(path, info)
    if explicit_roots:
        if not any(path_is_under(path, root) for root in explicit_roots):
            rendered = ", ".join(map(str, explicit_roots))
            reasons.append(f"resolved path is outside explicit local data roots: {rendered}")
    elif not default_local_data_path(path):
        reasons.append("resolved path is outside the default /data or /data<N> roots")
    return reasons


def resolve_hf3fs_root(raw_root: str | None) -> tuple[Path | None, MountInfo | None]:
    if raw_root is None:
        return None, None
    root = resolve_existing(Path(raw_root))
    if not root.is_dir():
        raise GuardError(f"allowed 3FS root is not a directory: {root}")
    info = mount_info(root)
    if info.fstype.strip().lower() != "fuse.hf3fs":
        raise GuardError(
            f"allowed 3FS root must use fuse.hf3fs: {root}: {info.fstype}"
        )
    return root, info


def validate_checkpoint_path(
    path: Path,
    info: MountInfo,
    explicit_local_roots: tuple[Path, ...],
    allowed_hf3fs_root: Path | None,
    hf3fs_mount: MountInfo | None,
) -> list[str]:
    if allowed_hf3fs_root is not None and path_is_under(path, allowed_hf3fs_root):
        if (
            hf3fs_mount is None
            or info.fstype.strip().lower() != "fuse.hf3fs"
            or info.source != hf3fs_mount.source
            or info.target != hf3fs_mount.target
        ):
            return [f"3FS mount identity differs from allowed root: {path}"]
        return []
    return validate_local_path(path, info, explicit_local_roots)


def parse_safetensors_header(path: Path) -> HeaderInfo:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise GuardError(f"{path}: truncated 8-byte Safetensors prefix")
            (header_bytes,) = struct.unpack("<Q", prefix)
            if header_bytes <= 1:
                raise GuardError(f"{path}: invalid header length {header_bytes}")
            if header_bytes > MAX_HEADER_BYTES:
                raise GuardError(
                    f"{path}: header length {header_bytes} exceeds safety limit "
                    f"{MAX_HEADER_BYTES}"
                )
            if 8 + header_bytes > file_size:
                raise GuardError(
                    f"{path}: header length {header_bytes} exceeds file size {file_size}"
                )
            raw_header = handle.read(header_bytes)
            if len(raw_header) != header_bytes:
                raise GuardError(f"{path}: truncated Safetensors header")
    except OSError as exc:
        raise GuardError(f"cannot read Safetensors file {path}: {exc}") from exc

    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardError(f"{path}: invalid Safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise GuardError(f"{path}: Safetensors header must be a JSON object")

    payload_bytes = file_size - 8 - header_bytes
    tensor_names: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                raise GuardError(f"{path}: __metadata__ must be an object")
            continue
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise GuardError(f"{path}: invalid tensor descriptor for {name!r}")
        offsets = descriptor.get("data_offsets")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int for item in offsets)
        ):
            raise GuardError(f"{path}: tensor {name!r} has invalid data_offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_bytes:
            raise GuardError(
                f"{path}: tensor {name!r} range [{start}, {end}) exceeds "
                f"payload {payload_bytes}"
            )
        if not isinstance(dtype, str) or not dtype:
            raise GuardError(f"{path}: tensor {name!r} has invalid dtype")
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            raise GuardError(f"{path}: tensor {name!r} has invalid shape")
        tensor_names.add(name)
        ranges.append((start, end, name))

    if not tensor_names:
        raise GuardError(f"{path}: Safetensors header contains no tensors")
    previous_end = 0
    previous_name = "<payload start>"
    for start, end, name in sorted(ranges):
        if start < previous_end:
            raise GuardError(
                f"{path}: overlapping data ranges for {previous_name!r} and {name!r}"
            )
        previous_end = end
        previous_name = name
    return HeaderInfo(
        tensor_names=frozenset(tensor_names),
        header_bytes=header_bytes,
        payload_bytes=payload_bytes,
    )


def safe_shard_path(
    checkpoint: Path, shard_name: str, allowed_hf3fs_root: Path | None = None
) -> Path:
    candidate = Path(shard_name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise GuardError(f"unsafe shard path in index: {shard_name!r}")
    resolved = resolve_existing(checkpoint / candidate)
    if not path_is_under(resolved, checkpoint) and not (
        allowed_hf3fs_root and path_is_under(resolved, allowed_hf3fs_root)
    ):
        raise GuardError(
            f"shard resolves outside checkpoint directory: {shard_name!r} -> {resolved}"
        )
    if not resolved.is_file():
        raise GuardError(f"referenced shard is not a regular file: {resolved}")
    return resolved


def validate_shard_name_coverage(names: Iterable[str]) -> None:
    parsed: list[tuple[int, int, str]] = []
    unparsed: list[str] = []
    for name in names:
        match = SHARD_NAME_RE.search(name)
        if match:
            parsed.append((int(match.group(1)), int(match.group(2)), name))
        else:
            unparsed.append(name)
    if parsed and unparsed:
        raise GuardError(
            "index mixes numbered and unnumbered shard names: " + ", ".join(sorted(unparsed))
        )
    if not parsed:
        return
    totals = {total for _, total, _ in parsed}
    if len(totals) != 1:
        raise GuardError(f"shard names disagree on total count: {sorted(totals)}")
    total = next(iter(totals))
    actual = {number for number, _, _ in parsed}
    expected = set(range(1, total + 1))
    if actual != expected or len(parsed) != total:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise GuardError(
            f"numbered shard coverage is incomplete: expected={total}, "
            f"files={len(parsed)}, missing={missing}, extra={extra}"
        )


def discover_checkpoint(
    checkpoint: Path,
    allowed_hf3fs_root: Path | None = None,
) -> tuple[Path | None, dict[str, str], list[Path], int | None]:
    indexes = sorted(checkpoint.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise GuardError(
            "multiple Safetensors index files are ambiguous: "
            + ", ".join(path.name for path in indexes)
        )
    if indexes:
        index = resolve_existing(indexes[0])
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuardError(f"cannot parse index {index}: {exc}") from exc
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise GuardError(f"{index}: weight_map must be a nonempty object")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise GuardError(f"{index}: metadata must be an object when present")
        expected_payload_bytes = metadata.get("total_size")
        if expected_payload_bytes is not None and (
            type(expected_payload_bytes) is not int or expected_payload_bytes < 0
        ):
            raise GuardError(f"{index}: metadata.total_size must be a nonnegative integer")
        normalized: dict[str, str] = {}
        for tensor, shard in weight_map.items():
            if not isinstance(tensor, str) or not tensor:
                raise GuardError(f"{index}: invalid tensor name in weight_map")
            if not isinstance(shard, str) or not shard.endswith(".safetensors"):
                raise GuardError(
                    f"{index}: tensor {tensor!r} has invalid shard value {shard!r}"
                )
            normalized[tensor] = shard
        shard_names = sorted(set(normalized.values()))
        validate_shard_name_coverage(shard_names)
        shards = [
            safe_shard_path(checkpoint, name, allowed_hf3fs_root)
            for name in shard_names
        ]
        return index, normalized, shards, expected_payload_bytes

    shards = sorted(resolve_existing(path) for path in checkpoint.glob("*.safetensors"))
    if not shards:
        raise GuardError(f"{checkpoint}: no .safetensors files or index found")
    if len(shards) != 1:
        raise GuardError(
            f"{checkpoint}: found {len(shards)} Safetensors files but no index"
        )
    return None, {}, shards, None


def validate_checkpoint_files(
    checkpoint: Path,
    reporter: Reporter,
    explicit_roots: tuple[Path, ...],
    allowed_hf3fs_root: Path | None = None,
    hf3fs_mount: MountInfo | None = None,
) -> tuple[int, int, int, Path | None, list[Path]]:
    index, weight_map, shards, expected_payload_bytes = discover_checkpoint(
        checkpoint, allowed_hf3fs_root
    )
    files_to_check = ([index] if index else []) + shards
    for path in files_to_check:
        assert path is not None
        info = mount_info(path)
        reasons = validate_checkpoint_path(
            path, info, explicit_roots, allowed_hf3fs_root, hf3fs_mount
        )
        if reasons:
            for reason in reasons:
                reporter.fail(f"weight file {path}: {reason}")

    header_by_shard: dict[str, HeaderInfo] = {}
    tensor_owner: dict[str, str] = {}
    # discover_checkpoint resolves symlinked shards so storage checks use their
    # physical mount. Keep each index filename for the index-to-header check.
    shard_aliases = sorted(set(weight_map.values())) if weight_map else [p.name for p in shards]
    for indexed_name, shard in zip(shard_aliases, shards, strict=True):
        header = parse_safetensors_header(shard)
        header_by_shard[indexed_name] = header
        for tensor in header.tensor_names:
            previous = tensor_owner.setdefault(tensor, indexed_name)
            if previous != indexed_name:
                raise GuardError(
                    f"tensor {tensor!r} appears in both {previous!r} and {indexed_name!r}"
                )

    if weight_map:
        for tensor, indexed_name in sorted(weight_map.items()):
            header = header_by_shard.get(Path(indexed_name).name)
            if header is None or tensor not in header.tensor_names:
                raise GuardError(
                    f"index maps tensor {tensor!r} to {indexed_name!r}, but its header "
                    "does not contain that tensor"
                )
        unindexed = sorted(set(tensor_owner) - set(weight_map))
        if unindexed:
            reporter.warn(
                f"headers contain {len(unindexed)} tensor(s) absent from weight_map"
            )

    actual_payload_bytes = sum(header.payload_bytes for header in header_by_shard.values())
    if expected_payload_bytes is not None:
        if expected_payload_bytes != actual_payload_bytes:
            raise GuardError(
                f"index metadata.total_size={expected_payload_bytes} does not match "
                f"validated shard payload bytes={actual_payload_bytes}"
            )
        reporter.passed(
            f"index metadata.total_size matches shard payloads: {actual_payload_bytes} bytes"
        )
    elif index is not None:
        reporter.warn("index has no metadata.total_size to cross-check shard payload bytes")

    disk_bytes = sum(shard.stat().st_size for shard in shards)
    tensor_count = len(tensor_owner)
    reporter.passed(
        f"checkpoint structure: index={index.name if index else '<single-file>'}, "
        f"shards={len(shards)}, tensors={tensor_count}, bytes={disk_bytes} "
        f"({human_bytes(disk_bytes)})"
    )
    reporter.passed("all Safetensors headers and index-to-tensor mappings are structurally valid")
    return disk_bytes, len(shards), tensor_count, index, shards


def recursive_json_load_methods(value: Any, location: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z]", "", str(key).lower())
            child_location = f"{location}.{key}"
            if normalized_key in LOAD_METHOD_KEYS and isinstance(item, (str, int, float)):
                found.append((child_location, str(item)))
            found.extend(recursive_json_load_methods(item, child_location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(recursive_json_load_methods(item, f"{location}[{index}]"))
    return found


CONFIG_LINE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:LOAD_METHOD|load_method)\s*(?:=|:)\s*"
    r"(?P<value>[^\s#;,]+)",
    re.IGNORECASE,
)
JSON_LINE_RE = re.compile(
    r"^\s*[\"'](?:LOAD_METHOD|load_method)[\"']\s*:\s*"
    r"[\"']?(?P<value>[^\"'\s,}]+)",
    re.IGNORECASE,
)


def config_load_methods(path: Path) -> list[tuple[str, str]]:
    resolved = resolve_existing(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GuardError(f"cannot read config file {resolved}: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        return recursive_json_load_methods(parsed, str(resolved))

    found: list[tuple[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = CONFIG_LINE_RE.match(line) or JSON_LINE_RE.match(line)
        if match:
            found.append((f"{resolved}:{line_number}", match.group("value")))
    return found


def command_load_methods(command: str) -> list[tuple[str, str]]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise GuardError(f"cannot parse --launch-command: {exc}") from exc
    found: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        env_match = re.match(r"^(?:LOAD_METHOD|load_method)=(.*)$", token, re.IGNORECASE)
        flag_match = re.match(
            r"^--(?:load-method|load_method)=(.*)$", token, re.IGNORECASE
        )
        if env_match:
            found.append(("launch command environment", env_match.group(1)))
        elif flag_match:
            found.append(("launch command flag", flag_match.group(1)))
        elif token.lower() in {"--load-method", "--load_method"}:
            if index + 1 >= len(tokens):
                raise GuardError(f"{token} in --launch-command has no value")
            index += 1
            found.append(("launch command flag", tokens[index]))
        index += 1
    return found


def normalize_load_method(value: str) -> str:
    return value.strip().strip("\"'").lower()


def validate_load_method(args: argparse.Namespace, reporter: Reporter) -> None:
    candidates: list[tuple[str, str]] = []
    if args.load_method is not None:
        candidates.append(("--load-method", args.load_method))
    if args.launch_command is not None:
        candidates.extend(command_load_methods(args.launch_command))
    for config_file in args.config_file:
        matches = config_load_methods(Path(config_file))
        if not matches:
            reporter.fail(f"config file has no explicit LOAD_METHOD/load_method: {config_file}")
        candidates.extend(matches)
    if "LOAD_METHOD" in os.environ:
        candidates.append(("environment LOAD_METHOD", os.environ["LOAD_METHOD"]))

    if not candidates:
        reporter.fail(
            "no explicit load method found; set the actual launch configuration to "
            "fastsafetensors"
        )
        return
    for source, value in candidates:
        normalized = normalize_load_method(value)
        if normalized != "fastsafetensors":
            reporter.fail(f"{source} selects forbidden load method {value!r}")
        else:
            reporter.passed(f"explicit load method from {source}: fastsafetensors")


def directory_size(path: Path) -> int:
    total = 0
    for root, directories, filenames in os.walk(path, followlinks=False):
        directories.sort()
        filenames.sort()
        root_path = Path(root)
        for filename in filenames:
            candidate = root_path / filename
            try:
                metadata = candidate.stat()
            except OSError:
                continue
            if stat_module.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    return total


def validate_copy_root(
    raw_root: str,
    source: Path,
    required_bytes: int,
    reserve_bytes: int,
    reporter: Reporter,
    explicit_roots: tuple[Path, ...],
) -> None:
    try:
        root = resolve_existing(Path(raw_root))
        info = mount_info(root)
    except GuardError as exc:
        reporter.fail(f"copy root {raw_root!r} is unusable: {exc}")
        return
    reasons = validate_local_path(root, info, explicit_roots)
    if reasons:
        for reason in reasons:
            reporter.fail(f"copy root {root}: {reason}")
        return
    if not root.is_dir():
        reporter.fail(f"copy root is not a directory: {root}")
        return
    if not os.access(root, os.W_OK | os.X_OK):
        reporter.fail(f"copy root is not writable/searchable by the current user: {root}")
        return
    capacity_required = required_bytes + reserve_bytes
    if info.available_bytes < capacity_required:
        reporter.fail(
            f"copy root {root} has {human_bytes(info.available_bytes)} available, "
            f"but copy plus reserve requires {human_bytes(capacity_required)} "
            f"(tree={human_bytes(required_bytes)}, reserve={human_bytes(reserve_bytes)})"
        )
        return
    destination = root / source.name
    if destination == source:
        reporter.passed(
            f"copy root capacity: {root} has {human_bytes(info.available_bytes)} available "
            f"for tree={human_bytes(required_bytes)} plus "
            f"reserve={human_bytes(reserve_bytes)}"
        )
        reporter.info("checkpoint is already at the computed local destination; no copy suggested")
        return
    source_arg = shlex.quote(str(source) + "/")
    destination_arg = shlex.quote(str(destination) + "/")
    reporter.passed(
        f"copy root capacity: {root} has {human_bytes(info.available_bytes)} available "
        f"for tree={human_bytes(required_bytes)} plus "
        f"reserve={human_bytes(reserve_bytes)}"
    )
    reporter.info(
        "read-only copy suggestion (not executed): "
        f"mkdir -p {shlex.quote(str(destination))} && "
        f"rsync -aL --info=progress2 {source_arg} {destination_arg}"
    )
    reporter.info(
        "checksum verification after copy: "
        f"rsync -aL --checksum --dry-run --itemize-changes "
        f"{source_arg} {destination_arg}"
    )


def resolve_explicit_roots(values: list[str]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in values:
        root = resolve_existing(Path(value))
        if not root.is_dir():
            raise GuardError(f"local data root is not a directory: {root}")
        info = mount_info(root)
        reasons = mount_rejection_reasons(root, info)
        if reasons:
            raise GuardError(f"invalid local data root {root}: {'; '.join(reasons)}")
        roots.append(root)
    return tuple(sorted(set(roots), key=str))


def preflight(args: argparse.Namespace) -> int:
    reporter = Reporter()
    try:
        checkpoint = resolve_existing(Path(args.checkpoint))
        if not checkpoint.is_dir():
            raise GuardError(f"checkpoint must be a directory: {checkpoint}")
        explicit_roots = resolve_explicit_roots(args.local_data_root)
        allowed_hf3fs_root, hf3fs_mount = resolve_hf3fs_root(args.allow_hf3fs_root)
        info = mount_info(checkpoint)
        reporter.info(f"checkpoint requested={Path(args.checkpoint).expanduser()}")
        reporter.info(f"checkpoint realpath={checkpoint}")
        reporter.info(
            f"mount target={info.target} source={info.source} fstype={info.fstype} "
            f"stat_fstype={info.stat_fs_type} options={info.options}"
        )
        reporter.info(
            f"stat device={info.device} inode={info.inode} kind={info.kind}; "
            f"df size={info.size_bytes} used={info.used_bytes} "
            f"available={info.available_bytes} ({human_bytes(info.available_bytes)})"
        )
        location_reasons = validate_checkpoint_path(
            checkpoint, info, explicit_roots, allowed_hf3fs_root, hf3fs_mount
        )
        if location_reasons:
            for reason in location_reasons:
                reporter.fail(f"checkpoint location: {reason}")
        else:
            if allowed_hf3fs_root is not None and path_is_under(checkpoint, allowed_hf3fs_root):
                reporter.passed(f"checkpoint resolves under allowed 3FS root {allowed_hf3fs_root}")
            else:
                reporter.passed("checkpoint resolves to an allowed local data-disk mount")

        disk_bytes, _, _, _, _ = validate_checkpoint_files(
            checkpoint, reporter, explicit_roots, allowed_hf3fs_root, hf3fs_mount
        )
        validate_load_method(args, reporter)

        tree_bytes = directory_size(checkpoint)
        reporter.info(
            f"checkpoint tree bytes={tree_bytes} ({human_bytes(tree_bytes)}); "
            f"weight shard bytes={disk_bytes} ({human_bytes(disk_bytes)})"
        )
        if args.copy_root:
            validate_copy_root(
                args.copy_root,
                checkpoint,
                tree_bytes,
                args.copy_reserve_bytes,
                reporter,
                explicit_roots,
            )
        elif location_reasons:
            reporter.fail(
                "rejected source requires a local copy; rerun with --copy-root pointing "
                "to a writable local data disk for capacity and copy guidance"
            )
    except GuardError as exc:
        reporter.fail(str(exc))

    if reporter.failures:
        print(
            f"[RESULT] FAIL failures={len(reporter.failures)} "
            f"warnings={len(reporter.warnings)}"
        )
        return 2
    print(f"[RESULT] PASS warnings={len(reporter.warnings)}")
    return 0


POSITIVE_LOG_RE = re.compile(
    r"(?i)(?:load[_ ]?method|loader|loading|weights?|selected|using|initialized)"
    r".{0,160}\bfastsafetensors\b|\bfastsafetensors\b.{0,160}"
    r"(?:load[_ ]?method|loader|loading|weights?|selected|using|initialized)"
)
FALLBACK_LOADER_RE = re.compile(
    r"(?i)(?:\b(?:fallback|fall(?:ing)?\s+back)\b.{0,160}"
    r"(?:load[_ ]?method|loader|weights?|checkpoint|safetensors|scratch|torch|pytorch|"
    r"huggingface)|(?:load[_ ]?method|loader|weights?|checkpoint|safetensors|scratch|"
    r"torch|pytorch|huggingface).{0,160}\b(?:fallback|fall(?:ing)?\s+back)\b)"
)
SCRATCH_LOADER_RE = re.compile(
    r"(?i)\bscratch\s+(?:weight\s+)?loader\b|"
    r"\b(?:weight\s+)?loader\s*(?:=|:|is)?\s*scratch\b"
)
EXPLICIT_METHOD_RE = re.compile(
    r"(?i)\bload[_ ]?method\s*(?:=|:)\s*[\"']?([A-Za-z0-9_.+-]+)"
)
SELECTED_LOADER_RE = re.compile(
    r"(?i)\b(?:using|selected|selecting|initialized|initializing)\s+"
    r"(?:weight\s+)?loader\s*(?:=|:)?\s*[\"']?([A-Za-z0-9_.+-]+)"
)


def iter_log_lines(paths: list[str]) -> Iterable[tuple[str, int, str]]:
    stdin_used = False
    for raw_path in paths:
        if raw_path == "-":
            if stdin_used:
                raise GuardError("stdin log '-' may be specified only once")
            stdin_used = True
            for line_number, line in enumerate(sys.stdin, start=1):
                yield "<stdin>", line_number, line.rstrip("\r\n")
            continue
        path = resolve_existing(Path(raw_path))
        if not path.is_file():
            raise GuardError(f"log is not a regular file: {path}")
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    yield str(path), line_number, line.rstrip("\r\n")
        except OSError as exc:
            raise GuardError(f"cannot read log {path}: {exc}") from exc


def verify_log(args: argparse.Namespace) -> int:
    reporter = Reporter()
    positive: tuple[str, int, str] | None = None
    forbidden: list[tuple[str, int, str, str]] = []
    try:
        for source, line_number, line in iter_log_lines(args.log):
            if positive is None and POSITIVE_LOG_RE.search(line):
                positive = (source, line_number, line)
            if FALLBACK_LOADER_RE.search(line):
                forbidden.append((source, line_number, line, "weight-loader fallback evidence"))
            if SCRATCH_LOADER_RE.search(line):
                forbidden.append((source, line_number, line, "scratch-loader evidence"))
            for pattern, label in (
                (EXPLICIT_METHOD_RE, "explicit non-FastSafetensors load method"),
                (SELECTED_LOADER_RE, "explicit non-FastSafetensors loader"),
            ):
                match = pattern.search(line)
                if match and normalize_load_method(match.group(1)) != "fastsafetensors":
                    forbidden.append((source, line_number, line, label))
    except GuardError as exc:
        reporter.fail(str(exc))

    if positive is None:
        reporter.fail(
            "startup log has no loader-related positive evidence for fastsafetensors"
        )
    else:
        source, line_number, line = positive
        reporter.passed(
            f"FastSafetensors startup evidence at {source}:{line_number}: {line}"
        )
    for source, line_number, line, label in forbidden:
        reporter.fail(f"{label} at {source}:{line_number}: {line}")
    if not forbidden:
        reporter.passed(
            "no scratch-loader, weight-loader fallback, or explicit alternate-loader "
            "evidence found"
        )

    if reporter.failures:
        print(f"[RESULT] FAIL failures={len(reporter.failures)}")
        return 2
    print("[RESULT] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only gate for local RTP-LLM Safetensors checkpoints and explicit "
            "FastSafetensors startup evidence."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate checkpoint storage, structure, and launch configuration"
    )
    preflight_parser.add_argument("--checkpoint", required=True)
    preflight_parser.add_argument(
        "--allow-hf3fs-root",
        help="Allow this exact fuse.hf3fs subtree for checkpoint files and shard symlinks",
    )
    preflight_parser.add_argument(
        "--load-method",
        help="explicit value copied from the actual launch configuration",
    )
    preflight_parser.add_argument(
        "--launch-command",
        help="actual launch command to parse only; it is never executed",
    )
    preflight_parser.add_argument(
        "--config-file",
        action="append",
        default=[],
        help="env, shell, YAML, or JSON config containing LOAD_METHOD/load_method",
    )
    preflight_parser.add_argument(
        "--local-data-root",
        action="append",
        default=[],
        help=(
            "explicit allowed local root; repeat as needed (default: /data and "
            "/data<N> path families)"
        ),
    )
    preflight_parser.add_argument(
        "--copy-root",
        help="existing local directory used only for capacity checks and copy advice",
    )
    preflight_parser.add_argument(
        "--copy-reserve-bytes",
        type=nonnegative_int,
        default=DEFAULT_COPY_RESERVE_BYTES,
        help=(
            "free bytes to preserve after a full copy "
            f"(default: {DEFAULT_COPY_RESERVE_BYTES})"
        ),
    )
    preflight_parser.set_defaults(handler=preflight)

    log_parser = subparsers.add_parser(
        "verify-log", help="validate finite post-launch startup log files"
    )
    log_parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="startup log path; repeat for multiple files or use '-' for stdin",
    )
    log_parser.set_defaults(handler=verify_log)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
