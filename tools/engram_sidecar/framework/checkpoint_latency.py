#!/usr/bin/env python3
"""Paired real-checkpoint lookup latency; never a complete decode-step SLO.

Both paths call HostEngramEmbedding on the same GPU/IDs. The local control is
main's complete _PinnedSharedTable, populated from independent source shards;
the remote table is attached through the verified private bundle. Imports are
lazy so argument, evidence, and reporting contracts can be tested without CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.engram_sidecar.framework import checkpoint_lookup as lookup
from tools.engram_sidecar.latency import bench_consumer as bench

COMPARISONS = {"raw_remote_minus_raw_local": ("remote_raw", "local_raw")}
_FAILED_RESOURCES = []
SOURCE_TOUCH_BYTES = 8 << 20


def emit(status, **values):
    print(json.dumps(dict(status=status, scope="real_checkpoint_lookup_component",
                          decode_step_slo="NOT_MEASURED", **values),
                     sort_keys=True, allow_nan=False), flush=True)


def integer_list(value, *, allowed=None):
    if not re.fullmatch(r"\d+(,\d+)*", value):
        raise ValueError("expected comma-separated nonnegative integers")
    result = [int(item) for item in value.split(",")]
    if len(result) != len(set(result)) or (allowed is not None and not set(result) <= set(allowed)):
        raise ValueError("duplicate or unsupported list item")
    return result


def choice_list(value, allowed):
    result = value.split(",")
    if not result or len(result) != len(set(result)) or not set(result) <= set(allowed):
        raise ValueError("duplicate or unsupported choice")
    return result


def diagnostic_declaration(args):
    if not getattr(args, "diagnostic_nvtx", False):
        return {}
    return dict(measurement_status="PROFILE_ONLY",
                diagnostic_options=dict(nvtx=True, cuda_profiler_api=bool(getattr(args, "diagnostic_cuda_profiler", False))))


def check_options(args):
    args.allow_numa_cache_reclaim = getattr(args, "allow_numa_cache_reclaim", False)
    args.allow_source_hardlinks = getattr(args, "allow_source_hardlinks", False)
    if args.allow_source_hardlinks and not args.discard_private_source_cache:
        raise ValueError("hardlink cache advice requires explicit source cache discard")
    if args.allow_numa_cache_reclaim and args.source_cache_numa is not None:
        raise ValueError("cache-reclaim admission is only supported without source NUMA staging")
    args.batches = integer_list(args.batches, allowed=(1, 2, 4, 8, 16, 32, 64))
    args.layers = None if args.layers == "first" else integer_list(args.layers)
    args.modes = choice_list(args.modes, ("eager", "graph"))
    args.distributions = choice_list(args.distributions, ("hot", "random"))
    if getattr(args, "diagnostic_cuda_profiler", False):
        if not getattr(args, "diagnostic_nvtx", False):
            raise ValueError("CUDA profiler capture requires diagnostic NVTX markers")
        groups = 1 if args.layer_mode == "together" or args.layers is None else len(args.layers)
        if groups != 1 or len(args.batches) != 1 or len(args.modes) != 1 or len(args.distributions) != 1:
            raise ValueError("CUDA profiler capture requires exactly one case and one distribution")
    if not 1 <= args.samples <= 1_000_000 or args.warmup < 1 or args.device < 0 or args.cpu_slot < 0:
        raise ValueError("invalid samples, warmup, device or CPU slot")
    if args.seed < 0 or args.calibration_samples < 0 or not math.isfinite(args.threshold_us) or args.threshold_us <= 0:
        raise ValueError("invalid seed, calibration count or threshold")
    if args.layer_mode not in ("single", "together") or args.memory_headroom_gib < 1:
        raise ValueError("valid layer mode and positive memory headroom required")
    if args.source_cache_numa is not None:
        if type(args.source_cache_numa) is not int or args.source_cache_numa < 0:
            raise ValueError("source cache NUMA must be a nonnegative integer")
        if not args.discard_private_source_cache:
            raise ValueError("source cache NUMA requires explicit private-source cache advice")
    for name in ("checkpoint", "source_checkpoint", "bundle", "private_inputs", "output"):
        if not getattr(args, name).is_absolute():
            raise ValueError("all input/output paths must be absolute")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("use a new output directory; prior samples must not be overwritten")
    return args


def layer_groups(available, selected, mode):
    """Single preserves first-layer default; together defaults to all real layers."""
    if mode not in ("single", "together") or not available:
        raise ValueError("valid mode and model layers required")
    selected = list(selected) if selected is not None else list(available if mode == "together" else available[:1])
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(available):
        raise ValueError("selected layer absent or duplicated")
    if mode == "together":
        if len(selected) < 2:
            raise ValueError("together requires at least two real layers")
        # Preserve actual model order even when the CLI lists IDs out of order.
        return [tuple(layer for layer in available if layer in selected)]
    return [(layer,) for layer in selected]


def invoke_lookup_chain(embeddings, layers, ids_by_layer, device):
    """Same-stream serial submissions, independent inputs; no inter-layer model work."""
    if len(layers) == 1:
        return embeddings[layers[0]](ids_by_layer[layers[0]], device)
    return tuple(embeddings[layer](ids_by_layer[layer], device) for layer in layers)


def split_layer_outputs(outputs, layers):
    if len(layers) == 1:
        return {layers[0]: dict(outputs)}
    if any(not isinstance(value, tuple) or len(value) != len(layers) for value in outputs.values()):
        raise AssertionError("joint replay must return exactly one actual output per selected layer")
    return {layer: {name: values[position] for name, values in outputs.items()}
            for position, layer in enumerate(layers)}


def resolve_cgroup_directory(cgroup_text, mountinfo_text):
    members = [line.split("::", 1)[1] for line in cgroup_text.splitlines() if line.startswith("0::")]
    if len(members) != 1:
        raise RuntimeError("a unique cgroup-v2 membership is required")
    member = Path(members[0])
    candidates = []
    for line in mountinfo_text.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator or right.split()[0] != "cgroup2":
            continue
        fields = left.split()
        if len(fields) < 5 or "\\" in fields[3] or "\\" in fields[4]:
            raise RuntimeError("unsupported escaped cgroup mount path")
        root, target = Path(fields[3]), Path(fields[4])
        try:
            suffix = member.relative_to(root)
        except ValueError:
            continue
        candidates.append((target / suffix, target, root == Path("/")))
    if len(candidates) != 1:
        raise RuntimeError("cannot uniquely resolve this process cgroup-v2 mount")
    return candidates[0]


def parse_kib_field(text, field):
    pattern = re.compile(r"(?:Node\s+\d+\s+)?" + re.escape(field) + r":\s+(\d+)\s+kB")
    matches = [match.group(1) for line in text.splitlines() if (match := pattern.fullmatch(line.strip()))]
    if len(matches) != 1:
        raise RuntimeError("required memory counter unavailable")
    return int(matches[0]) * 1024


def read_cgroup_memory(current, mount, hierarchy_root):
    groups = []
    while True:
        limit_path = current / "memory.max"
        # Only the actual hierarchy-root mount may omit its limit interface.
        # Missing counters or limits at any non-root ancestor remain fatal.
        root_without_limit = current == mount and hierarchy_root and not limit_path.exists()
        limit = "max" if root_without_limit else limit_path.read_text().strip()
        groups.append(dict(path=str(current), current_bytes=int((current / "memory.current").read_text()),
                           max_bytes=None if limit == "max" else int(limit),
                           limit_source="root_no_limit_interface" if root_without_limit else "memory.max"))
        if current == mount:
            break
        current = current.parent
    return groups


def memory_snapshot(node, source_node=None):
    current, mount, hierarchy_root = resolve_cgroup_directory(Path("/proc/self/cgroup").read_text(),
                                                             Path("/proc/self/mountinfo").read_text())
    groups = read_cgroup_memory(current, mount, hierarchy_root)
    numa_text = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
    # An estimate, never a reservation: exclude dirty/writeback pages.
    # Shmem is not Inactive(file); the admission check adds its headroom margin.
    clean_inactive = max(0, parse_kib_field(numa_text, "Inactive(file)")
                        - parse_kib_field(numa_text, "Dirty")
                        - parse_kib_field(numa_text, "Writeback"))
    result = dict(cgroups=groups, shm_free_bytes=shutil.disk_usage("/dev/shm").free,
                host_available_bytes=parse_kib_field(Path("/proc/meminfo").read_text(), "MemAvailable"),
                numa_free_bytes=parse_kib_field(numa_text, "MemFree"),
                numa_cache_reclaim_estimate_bytes=clean_inactive)
    if source_node is not None:
        result["source_numa_free_bytes"] = parse_kib_field(
            Path(f"/sys/devices/system/node/node{source_node}/meminfo").read_text(), "MemFree")
    return result


def check_source_cache_budget(snapshot, *, table_bytes, source_bytes, headroom_bytes, phase):
    """Count all source files together; only verified resident pages remove a future charge."""
    if phase not in ("before_prefault", "prefaulted", "after_load"):
        raise ValueError("invalid source cache admission phase")
    if min(table_bytes, source_bytes) < 0 or headroom_bytes <= 0:
        raise ValueError("nonnegative allocation sizes and positive headroom required")
    table = table_bytes if phase != "after_load" else 0
    source = source_bytes if phase == "before_prefault" else 0
    margin = headroom_bytes + SOURCE_TOUCH_BYTES
    required = table + source + margin
    for group in snapshot["cgroups"]:
        if group["max_bytes"] is not None and group["max_bytes"] - group["current_bytes"] < required:
            raise RuntimeError("insufficient measured cgroup headroom for source NUMA staging")
    if snapshot["host_available_bytes"] < required:
        raise RuntimeError("insufficient host headroom for source NUMA staging")
    if snapshot["numa_free_bytes"] < table + margin:
        raise RuntimeError("insufficient GPU-local NUMA headroom for pinned baseline")
    if snapshot["source_numa_free_bytes"] < source + margin:
        raise RuntimeError("insufficient source NUMA headroom for complete source files")
    if snapshot["shm_free_bytes"] < table:
        raise RuntimeError("insufficient /dev/shm for complete pinned baseline")
    return dict(phase=phase, future_table_bytes=table, future_source_bytes=source,
                transient_touch_bytes=SOURCE_TOUCH_BYTES, reserved_headroom_bytes=headroom_bytes,
                snapshot=snapshot)


def check_memory_budget(snapshot, *, table_bytes, source_bytes, headroom_bytes, before,
                        allow_numa_cache_reclaim=False):
    """Conservative marginal budget; fadvise does not count as guaranteed reclaim."""
    if min(table_bytes, source_bytes, headroom_bytes) < 0 or headroom_bytes == 0:
        raise ValueError("nonnegative sizes and positive headroom required")
    additional = table_bytes + source_bytes if before else 0
    required = additional + headroom_bytes
    for group in snapshot["cgroups"]:
        if group["max_bytes"] is not None and group["max_bytes"] - group["current_bytes"] < required:
            raise RuntimeError("insufficient measured cgroup memory headroom for complete-table baseline")
    reclaim = (snapshot.get("numa_cache_reclaim_estimate_bytes", 0)
               if before and allow_numa_cache_reclaim else 0)
    if snapshot["host_available_bytes"] < required or snapshot["numa_free_bytes"] + reclaim < required:
        raise RuntimeError("insufficient host or GPU-local NUMA headroom")
    if snapshot["shm_free_bytes"] < (table_bytes if before else 0):
        raise RuntimeError("insufficient /dev/shm for complete pinned table")
    return dict(required_marginal_bytes=additional, reserved_headroom_bytes=headroom_bytes,
                phase="before_load" if before else "after_load", snapshot=snapshot,
                numa_cache_reclaim_estimate_used_bytes=reclaim,
                memory_reservation_guaranteed=False)


def source_paths(source, index, layer):
    from tools.engram_sidecar.framework.stage_checkpoint import _basename
    return sorted({source / _basename(index[f"layers.{layer}.engram.embed.{role}"])
                   for role in ("weight", "scale")})


def private_source_cache_advice(source, paths, *, allow_hardlinks=False):
    """Best-effort file-cache advice on an explicitly opted-in private source copy.

    Files must be current-UID, owner-readable and non-writable;
    source root must be mode0700. This cannot prove no same-UID process holds an
    open mapping. The operator promises immutable, task-owned copies. No global
    cache controls or writes to checkpoint bytes are used.
    Hardlinks require separate opt-in: advice affects all aliases of the inode;
    the operator must verify those aliases also belong to this experiment.
    """
    from tools.engram_sidecar.framework.stage_checkpoint import _canonical, _snapshot, _unchanged
    source = _canonical(source)
    directory = source.stat()
    if directory.st_uid != os.getuid() or stat.S_IMODE(directory.st_mode) != 0o700:
        raise ValueError("cache advice requires a current-UID mode0700 task-private source directory")
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("per-file POSIX_FADV_DONTNEED unavailable")
    advised = []
    for path in paths:
        if path.parent != source:
            raise ValueError("cache advice path must be a direct private-source child")
        expected = _snapshot(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or (info.st_nlink != 1 and not allow_hardlinks)
                    or stat.S_IMODE(info.st_mode) != 0o400
                    or (info.st_dev, info.st_ino) != (expected["device"], expected["inode"])):
                raise ValueError("cache advice requires private single-link immutable mode0400 source files")
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            _unchanged(path, expected)
            advised.append(dict(file=path.name, bytes=info.st_size, hardlinks=info.st_nlink,
                                advice="DONTNEED", reclaimed_bytes="NOT_ASSUMED"))
        finally:
            os.close(fd)
    return advised


def rounded_mapping_bytes(size, page_size):
    if type(size) is not int or size <= 0 or size > sys.maxsize:
        raise ValueError("source mmap size must fit positive signed pointer size")
    if type(page_size) is not int or page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("invalid host page size")
    rounded = ((size + page_size - 1) // page_size) * page_size
    if rounded > sys.maxsize:
        raise ValueError("page-rounded source mmap size overflow")
    return rounded


def parse_complete_numa_pages(text, address, node, size, page_size, *, marker="file="):
    """Require every page, including the partial final page, on exactly one node."""
    matches = [line for line in text.splitlines() if line.split()
               and int(line.split()[0], 16) == address and marker in line]
    if len(matches) != 1:
        raise RuntimeError("cannot uniquely identify complete source/pinned mapping")
    pages = {int(n): int(count) for n, count in re.findall(r"\bN(\d+)=(\d+)\b", matches[0]) if int(count)}
    sizes = re.findall(r"\bkernelpagesize_kB=(\d+)\b", matches[0])
    expected = rounded_mapping_bytes(size, page_size) // page_size
    if len(sizes) != 1 or int(sizes[0]) * 1024 != page_size:
        raise RuntimeError("source/pinned mapping page-size accounting is unsupported")
    if set(pages) != {node} or sum(pages.values()) != expected:
        raise RuntimeError("source/pinned mapping is incomplete or has pages on another NUMA node")
    return dict(node_pages={str(n): count for n, count in pages.items()}, expected_node=node,
                expected_pages=expected, resident_pages=sum(pages.values()), page_bytes=page_size,
                resident_bytes=expected * page_size)


def cpu_on_node(node, allowed):
    selected = set()
    for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        low, _, high = part.partition("-")
        selected.update(range(int(low), int(high or low) + 1))
    choices = sorted(selected & set(allowed))
    if not choices:
        raise RuntimeError("source NUMA has no CPU in the original process affinity")
    return choices[0]


class ReadOnlySourceMapping:
    """A private file's real PROT_READ/MAP_SHARED mapping; no Python buffer borrowing."""

    def __init__(self, source, path):
        from tools.engram_sidecar.framework.stage_checkpoint import _canonical, _snapshot
        source = _canonical(source)
        directory = source.stat()
        if directory.st_uid != os.getuid() or stat.S_IMODE(directory.st_mode) != 0o700 or path.parent != source:
            raise ValueError("source prefault requires a current-UID mode0700 private directory")
        self.path, self.expected = path, _snapshot(path)
        self.fd, self.address = None, None
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.munmap.restype = ctypes.c_int
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(self.fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != 0o400
                    or (info.st_dev, info.st_ino) != (self.expected["device"], self.expected["inode"])):
                raise ValueError("source prefault requires private single-link immutable mode0400 files")
            self.size = info.st_size
            self.resident_bytes = rounded_mapping_bytes(self.size, self.page_size)
            # Linux/Unix PROT_READ=1, MAP_SHARED=1. No MAP_POPULATE: first
            # touches must happen only after binding the source-node CPU.
            address = self.libc.mmap(None, self.size, 1, 1, self.fd, 0)
            if address is None or address == ctypes.c_void_p(-1).value:
                code = ctypes.get_errno()
                raise OSError(code, "private read-only source mmap failed")
            self.address = address
        except BaseException:
            self.close()
            raise

    def touch(self):
        if self.address is None:
            raise RuntimeError("source mapping is closed")
        touched = 0
        for offset in range(0, self.size, SOURCE_TOUCH_BYTES):
            length = min(SOURCE_TOUCH_BYTES, self.size - offset)
            block = ctypes.string_at(self.address + offset, length)
            if len(block) != length:
                raise RuntimeError("source prefault did not read complete chunk")
            touched += len(block)
            del block  # At most one 8MiB transient copy, including between iterations.
        return touched

    def verify(self, text, node):
        from tools.engram_sidecar.framework.stage_checkpoint import _unchanged
        _unchanged(self.path, self.expected)
        # Permission changes can share the same coarse filesystem timestamp.
        # Recheck the admitted immutable-file contract, not just stat identity.
        info = os.fstat(self.fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o400
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise RuntimeError("source mapping no longer has immutable private-file permissions")
        evidence = parse_complete_numa_pages(text, self.address, node, self.size, self.page_size)
        return dict(file=self.path.name, file_bytes=self.size, **evidence)

    def close(self):
        try:
            if self.address is not None:
                if self.libc.munmap(self.address, self.size):
                    raise OSError(ctypes.get_errno(), "private read-only source munmap failed")
                self.address = None
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


class SourceCacheBatch:
    """Keep every verified source mapping alive until main's copy has finished."""

    def __init__(self, source, paths, node, allowed_cpus):
        self.node, self.allowed_cpus, self.mappings = node, set(allowed_cpus), []
        try:
            for path in paths:
                self.mappings.append(ReadOnlySourceMapping(source, path))
        except BaseException:
            self.close()
            raise

    @property
    def resident_bytes(self):
        return sum(item.resident_bytes for item in self.mappings)

    def prefault(self):
        previous = os.sched_getaffinity(0)
        cpu = cpu_on_node(self.node, self.allowed_cpus)
        try:
            os.sched_setaffinity(0, {cpu})
            for item in self.mappings:
                if item.touch() != item.size:
                    raise RuntimeError("source prefault did not touch complete file")
        finally:
            os.sched_setaffinity(0, previous)
        return cpu

    def verify(self):
        text = Path("/proc/self/numa_maps").read_text()
        return [item.verify(text, self.node) for item in self.mappings]

    def close(self):
        errors = []
        for item in reversed(self.mappings):
            try:
                item.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            retain_failed(self, *errors)
            raise RuntimeError("private source mapping cleanup remains pending") from errors[0]


def construct_with_source_cache(*, source, files, source_node, gpu_node, allowed_cpus,
                                table_bytes, headroom_bytes, construct):
    """Admission/setup only. The callback is unchanged main _PinnedSharedTable construction."""
    if source_node == gpu_node:
        raise ValueError("source-cache NUMA must differ from GPU-local HOST_NUMA")
    cache, control, evidence = None, None, {}
    try:
        cache = SourceCacheBatch(source, files, source_node, allowed_cpus)
        source_bytes = cache.resident_bytes
        rounded_table = rounded_mapping_bytes(table_bytes, os.sysconf("SC_PAGE_SIZE"))
        def budget(phase):
            snapshot = memory_snapshot(gpu_node, source_node)
            return check_source_cache_budget(snapshot, table_bytes=rounded_table, source_bytes=source_bytes,
                                             headroom_bytes=headroom_bytes, phase=phase)
        evidence["before_prefault"] = budget("before_prefault")
        evidence["source_cpu"] = cache.prefault()
        evidence["source_before_construct"] = cache.verify()
        evidence["prefaulted"] = budget("prefaulted")
        # Source mappings remain live; main still creates and registers its own
        # complete tmpfs-backed table, on the restored GPU-local CPU affinity.
        control = construct()
        evidence["source_after_construct"] = cache.verify()
        evidence["pinned_complete"] = parse_complete_numa_pages(
            Path("/proc/self/numa_maps").read_text(), control.storage.data_ptr(), gpu_node,
            table_bytes, os.sysconf("SC_PAGE_SIZE"), marker="rtp_llm_engram_")
        evidence["after_load"] = budget("after_load")
        evidence["eviction_prevented"] = False  # mmap is not mlock; verification is fail-closed.
        return control, evidence
    except BaseException:
        if control is not None:
            retain_failed(control)
        raise
    finally:
        if cache is not None:
            try:
                cache.close()
            except BaseException:
                if control is not None:
                    retain_failed(control)
                raise


def parse_numa_pages(text, address, node):
    matches = [line for line in text.splitlines() if line.split()
               and int(line.split()[0], 16) == address and "rtp_llm_engram_" in line]
    if len(matches) != 1:
        raise RuntimeError("cannot uniquely locate pinned table in numa_maps")
    pages = {int(n): int(count) for n, count in re.findall(r"\bN(\d+)=(\d+)\b", matches[0]) if int(count)}
    if not pages or set(pages) != {node}:
        raise RuntimeError("baseline pages are not exclusively on GPU-local HOST_NUMA")
    return {"node_pages": {str(n): count for n, count in pages.items()}, "expected_gpu_host_numa": node}


def bind_gpu_local_cpu(device, slot):
    from cuda.bindings import driver
    status, = driver.cuInit(0)
    if int(status):
        raise RuntimeError("cuInit failed")
    status, gpu = driver.cuDeviceGet(device)
    if int(status):
        raise RuntimeError("cuDeviceGet failed")
    status, node = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, gpu)
    if int(status) or int(node) < 0:
        raise RuntimeError("GPU-local host NUMA identity unavailable")
    node = int(node)
    selected = set()
    for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        low, _, high = part.partition("-")
        selected.update(range(int(low), int(high or low) + 1))
    choices = sorted(selected & os.sched_getaffinity(0))
    if slot >= len(choices):
        raise RuntimeError("no authorized CPU at requested GPU-local NUMA slot")
    os.sched_setaffinity(0, {choices[slot]})
    return node, choices[slot]


def chunk_selected_ids(ids, maximum=lookup.MAX_SELECTED_ROWS):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("positive chunk size required")
    if any(type(row) is not int or row < -1 for row in ids):
        raise ValueError("invalid lookup row")
    rows = sorted(set(ids) - {-1})
    return [rows[start:start + maximum] for start in range(0, len(rows), maximum)]


def source_rows(source, index, layout, layer, matrix):
    flat = [row for item in matrix for row in item]
    decoded = {-1: [0.0] * layout.head_dim}
    for chunk in chunk_selected_ids(flat):
        selected = {str(lid): set(chunk) if lid == layer else set() for lid in layout.layer_ids}
        records, _ = lookup.read_selected_rows(source, index, layout, selected)
        for record in records[str(layer)]:
            decoded[record["row_id"]] = lookup.decode_row(bytes.fromhex(record["fp8_hex"]),
                                                         bytes.fromhex(record["e8m0_hex"]))
    return [[decoded[row] for row in item] for item in matrix]


def source_sample_check(source, index, layout, parsed):
    selected = {layer: set(rows) for layer, rows in parsed.items()}
    records, headers = lookup.read_selected_rows(source, index, layout, selected)
    for layer, values in records.items():
        for row in values:
            actual = bytes.fromhex(row["fp8_hex"]), bytes.fromhex(row["e8m0_hex"])
            if actual != parsed[layer][row["row_id"]]:
                raise ValueError("independent baseline source differs from verified oracle")
    return headers


def make_local_table(source, index, layout, layer, device):
    """Use main's exact tmpfs/pinned baseline, with mmap source and bounded copy.

    safe_open maps source tensors; _PinnedSharedTable copies in 64MiB chunks.
    Source file pages can still be resident during loading. Budget tmpfs + file
    cache headroom; this is not a claim of constant-RSS checkpoint staging.
    """
    from safetensors import safe_open
    from rtp_llm.models_py.modules.dsv41.engram import _PinnedSharedTable
    from tools.engram_sidecar.framework.stage_checkpoint import _basename, _snapshot
    names = [f"layers.{layer}.engram.embed.{role}" for role in ("weight", "scale")]
    paths = {name: source / _basename(index[name]) for name in names}
    before = {str(path): _snapshot(path) for path in set(paths.values())}
    with ExitStack() as stack:
        files = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                 for path in set(paths.values())}
        q, s = (files[paths[name]].get_tensor(name) for name in names)
        rows = dict(zip(layout.layer_ids, layout.num_embeddings))[layer]
        if tuple(q.shape) != (rows, layout.head_dim) or tuple(s.shape) != (rows, layout.head_dim // 32):
            raise ValueError("baseline source geometry differs from model")
        identity = json.dumps([before, layer, uuid4().hex], sort_keys=True)
        control = _PinnedSharedTable(q, s, identity, device)
        del q, s
    after = {str(path): _snapshot(path) for path in set(paths.values())}
    if before != after:
        retain_failed(control)
        raise RuntimeError("source changed while constructing baseline")
    return control


def make_local_embedding(control, embedding_type, torch):
    """Untimed proof that main's two CUDA tensor views alias the registered host table."""
    before = int(torch.cuda.memory_allocated(control.device))
    embedding = embedding_type(control.weight, control.scales, pinned=control)
    delta = int(torch.cuda.memory_allocated(control.device)) - before
    views = embedding._uva
    if not isinstance(views, tuple) or len(views) != 2:
        raise RuntimeError("baseline requires both registered host CUDA views")
    aliases = []
    with torch.cuda.device(control.device):
        for tensor, view in zip((control.weight, control.scales), views):
            status, pointer = control.runtime.cudaHostGetDevicePointer(tensor.data_ptr(), 0)
            if int(status):
                raise RuntimeError("baseline alias verification could not query registered host view")
            aliases.append(view.data_ptr() == int(pointer))
    if not all(aliases) or delta != 0:
        raise RuntimeError("baseline view wrapping must alias both host tensors without CUDA allocation")
    return embedding, dict(weight_alias=aliases[0], scale_alias=aliases[1],
                           cuda_allocated_delta_bytes=delta)


def retain_failed(*resources):
    for resource in resources:
        for name in ("_registration_finalizer", "_lease_finalizer"):
            finalizer = getattr(resource, name, None)
            if finalizer is not None:
                finalizer.atexit = False
    _FAILED_RESOURCES.append(resources)


def close_local(control, torch):
    torch.cuda.synchronize(control.device)
    finalizer = control._registration_finalizer
    if not finalizer.alive:
        raise RuntimeError("baseline registration finalizer ran unexpectedly")
    with torch.cuda.device(control.device):
        status, = control.runtime.cudaHostUnregister(control.storage.data_ptr())
    if int(status):
        raise RuntimeError("baseline cudaHostUnregister failed")
    if finalizer.detach() is None:
        raise RuntimeError("cannot detach completed baseline unregister")
    control._lease_finalizer()


def write_new_json(path, document):
    bench._validate_metadata(document)
    # Exclusive create prevents silent replacement of a prior run or result.
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def make_records(*, mode, distribution, pair_id, path, order, begin_ns,
                 submit_ns, complete_ns, gpu_us, before_switches, after_switches):
    if mode not in ("eager", "graph") or not math.isfinite(gpu_us) or gpu_us <= 0:
        raise ValueError("valid mode and positive finite GPU interval required")
    parts = bench._wall_parts(begin_ns, submit_ns, complete_ns, before_switches, after_switches)
    # Keep neutral naming for eager: this is a Python dispatch call, not Graph replay.
    parts["cpu_submit_call_us"] = parts.pop("cpu_graph_replay_call_us")
    base = dict(distribution=distribution, epoch=0, pair_id=pair_id, path=path,
                order_index=order.index(path), path_order=order,
                wall_begin_perf_counter_ns=begin_ns, wall_end_perf_counter_ns=complete_ns,
                mode=mode, **parts)
    return [dict(base, clock=f"gpu_{mode}_event_interval", duration_us=gpu_us),
            dict(base, clock="cpu_submit_to_event_complete", duration_us=(complete_ns - begin_ns) / 1000.0)]


def diagnostic_marker(metadata, distribution, pair_id, path, order_index):
    """Allowlisted geometry only: never interpolate paths, handles, pointers or IDs."""
    layers = metadata["layer_ids_measured"]
    if (not layers or any(type(layer) is not int or layer < 0 for layer in layers)
            or type(metadata["batch"]) is not int or metadata["batch"] not in (1, 2, 4, 8, 16, 32, 64)
            or metadata["mode"] not in ("eager", "graph") or distribution not in ("hot", "random")
            or type(pair_id) is not int or pair_id < 0 or path not in ("local_raw", "remote_raw")
            or type(order_index) is not int or order_index not in (0, 1)):
        raise ValueError("invalid diagnostic marker geometry")
    return ("engram_lookup.layers=" + ",".join(map(str, layers))
            + f".M={metadata['batch']}.mode={metadata['mode']}.ids={distribution}"
            + f".pair={pair_id}.path={path}.order={order_index}")


class DiagnosticTrace:
    """Opt-in API bookkeeping outside both clocks; no ownership of CUDA resources."""
    def __init__(self, nvtx, runtime=None):
        self.nvtx, self.runtime = nvtx, runtime
        self.marker_active = False
        self.pushes = self.pops = 0
        self.start_attempted = self.stop_attempted = False
        self.start_status = self.stop_status = "NOT_REQUESTED"
        self.cleanup_errors = []

    def _profiler_call(self, name):
        try:
            result = getattr(self.runtime, name)()
            if not isinstance(result, tuple) or len(result) != 1 or int(result[0]) != 0:
                raise RuntimeError("nonzero or invalid profiler API status")
        except Exception:
            raise RuntimeError("diagnostic profiler API failed: " + name) from None

    def start(self):
        if self.runtime is not None:
            if self.start_attempted:
                raise RuntimeError("diagnostic capture already attempted")
            self.start_attempted = True
            self.start_status = "FAILED"
            self._profiler_call("cudaProfilerStart")
            self.start_status = "SUCCESS"

    def push(self, label):
        if self.marker_active:
            raise RuntimeError("nested diagnostic sample markers are forbidden")
        self.nvtx.range_push(label)
        self.marker_active = True
        self.pushes += 1

    def pop(self, primary=None):
        if not self.marker_active:
            return
        self.marker_active = False
        try:
            self.nvtx.range_pop()
            self.pops += 1
        except BaseException:
            self.cleanup_errors.append("NVTX_POP_FAILED")
            if primary is None:
                raise RuntimeError("diagnostic NVTX pop failed") from None

    def stop(self, primary=None):
        if not self.start_attempted or self.stop_attempted:
            return
        self.stop_attempted = True
        self.stop_status = "FAILED"
        try:
            self._profiler_call("cudaProfilerStop")
            self.stop_status = "SUCCESS"
        except BaseException:
            self.cleanup_errors.append("PROFILER_STOP_FAILED")
            if primary is None:
                raise

    def close(self, primary=None):
        # A failed pop must not prevent Stop, and cleanup must preserve a primary
        # CUDA/oracle/I/O exception. Normal cleanup failures remain fatal.
        failure = primary
        try:
            self.pop(primary)
        except BaseException as error:
            failure = error
        self.stop(failure)
        if primary is None and failure is not None:
            raise failure

    def evidence(self):
        return dict(nvtx_pushes=self.pushes, nvtx_pops=self.pops,
                    profiler_start=self.start_status, profiler_stop=self.stop_status,
                    cleanup_errors=list(self.cleanup_errors),
                    trace_collection="EXTERNAL_NSYS_ARTIFACT_REQUIRED",
                    clock_alignment="NVTX label and CUDA launch correlation; no assumed shared clock origin")


def create_diagnostic_trace(args, torch):
    if not getattr(args, "diagnostic_nvtx", False):
        return None
    runtime = None
    if getattr(args, "diagnostic_cuda_profiler", False):
        from cuda.bindings import runtime
    return DiagnosticTrace(torch.cuda.nvtx, runtime)


def benchmark_case(*, paths, storage, update, verify, args, metadata, output):
    """One mode/case. Inputs, verification and calibration are outside timing.

    Both Graph paths use the SAME guarded replay wrapper/pinning overhead;
    local's guard is conservative (it does not actually read remote storage).
    The benchmark does not grant engine-owned CUDA Graph support.
    """
    import torch
    from rtp_llm.models_py.modules.dsv41.engram_storage import CapturedEngramGraph
    mode = metadata["mode"]
    graphs, events, outputs, records = {}, {}, {}, []
    stream = torch.cuda.Stream(device=args.device)
    stream.wait_stream(torch.cuda.current_stream(args.device))
    reader = bench._thread_switches_reader()
    l2 = bench._l2_bytes(torch, args.device)
    if args.l2_disturb and (l2 is None or 2 * l2 > 1 << 30):
        raise RuntimeError("runtime L2 size unavailable or scratch exceeds 1GiB")
    checks = []
    diagnostic = None
    raw = output / "samples.jsonl"
    try:
        with raw.open("x") as logfile, torch.inference_mode(), torch.cuda.stream(stream):
            os.chmod(raw, 0o600)
            diagnostic = create_diagnostic_trace(args, torch)
            scratch = torch.zeros(2 * l2, dtype=torch.uint8, device=args.device) if args.l2_disturb else None
            for name in paths:
                events[name] = (torch.cuda.Event(enable_timing=True, external=mode == "graph"),
                                torch.cuda.Event(enable_timing=True, external=mode == "graph"))
            for sample in range(args.warmup):
                update(sample, "random")
                for name in bench.balanced_order(paths, sample, args.seed):
                    outputs[name] = paths[name]()
            stream.synchronize()
            if mode == "graph":
                for name, path in paths.items():
                    def captured(path=path, name=name):
                        # CapturedEngramGraph uses its own capture stream.
                        events[name][0].record()
                        value = path()
                        events[name][1].record()
                        return value
                    graphs[name] = CapturedEngramGraph.capture(captured, [storage], warmup=1)
                del captured

            def invoke(name):
                if mode == "graph":
                    return graphs[name].replay()
                begin, end = events[name]
                begin.record(stream)
                value = paths[name]()
                end.record(stream)
                return value

            # Verify actual replay results on two different input buffers; no
            # eager overwrite may precede these checks.
            for index in (0, 1):
                update(index, "random")
                stream.synchronize()
                for name in paths:
                    outputs[name] = invoke(name)
                    events[name][1].synchronize()
                verify(index, "random", outputs)
                checks.append(dict(phase="before_measurement", trace_index=index, mode=mode))
            if diagnostic is not None:
                diagnostic.start()
            for distribution in args.distributions:
                for pair_id in range(args.samples):
                    update(pair_id, distribution)
                    order = bench.balanced_order(paths, pair_id, args.seed)
                    for name in order:
                        label = diagnostic_marker(metadata, distribution, pair_id, name, order.index(name)) if diagnostic is not None else None
                        if scratch is not None:
                            scratch.add_(1)
                        stream.synchronize()
                        if diagnostic is not None:
                            diagnostic.push(label)
                        try:
                            before = reader() if reader else None
                            begin_ns = time.perf_counter_ns()
                            outputs[name] = invoke(name)
                            submit_ns = time.perf_counter_ns()
                            events[name][1].synchronize()
                            complete_ns = time.perf_counter_ns()
                            after = reader() if reader else None
                        except BaseException as error:
                            if diagnostic is not None:
                                diagnostic.pop(error)
                            raise
                        else:
                            if diagnostic is not None:
                                diagnostic.pop()
                        rows = make_records(mode=mode, distribution=distribution, pair_id=pair_id,
                                            path=name, order=order, begin_ns=begin_ns, submit_ns=submit_ns,
                                            complete_ns=complete_ns,
                                            gpu_us=events[name][0].elapsed_time(events[name][1]) * 1000,
                                            before_switches=before, after_switches=after)
                        if diagnostic is not None:
                            for row in rows:
                                row.update(measurement_status="PROFILE_ONLY", diagnostic_nvtx_range=label)
                        records.extend(rows)
                        for row in rows:
                            logfile.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    if (pair_id + 1) % 128 == 0:
                        logfile.flush()
                        os.fsync(logfile.fileno())
                if diagnostic is not None:
                    diagnostic.stop()
                verify(args.samples - 1, distribution, outputs)
                checks.append(dict(phase="after_distribution", distribution=distribution,
                                   trace_index=0 if distribution == "hot" else args.samples - 1, mode=mode))
                logfile.flush()
                os.fsync(logfile.fileno())
            if diagnostic is not None:
                diagnostic.close()
            del invoke
        stream.synchronize()
        for graph in graphs.values():
            graph.close()
        graphs.clear()
        outputs.clear()
        events.clear()
    except BaseException as error:
        if diagnostic is not None:
            diagnostic.close(error)
        # A failed close/capture may still have Graph references in traceback.
        # Preserve the entire chain; outer run also retains its pinned baseline.
        retain_failed(graphs, outputs, paths, storage, error)
        raise
    summary = bench.summarize_records(records, COMPARISONS, threshold_us=args.threshold_us,
                                      bootstrap_iterations=0, seed=args.seed)
    summary.update(case_metadata=metadata, correctness=checks,
                   protocol=dict(samples_per_distribution=args.samples, independent_runs=1,
                                 graph=mode == "graph", seed=args.seed, warmup=args.warmup,
                                 input_preparation_timed=False, per_request_sync=True,
                                 cache_claim="scratch_disturbed_not_guaranteed_cold" if args.l2_disturb else "uncontrolled",
                                 runtime_l2_bytes=l2, scratch_bytes=2 * l2 if args.l2_disturb else 0,
                                 common_graph_guard=mode == "graph", instrumentation_subtracted=False,
                                 eager_gpu_interval="includes CPU dispatch gaps between event records",
                                 graph_gpu_interval="captured external nodes; not exclusive kernel execution time",
                                 wall_partition="submit call + completion synchronization; sync is not exclusive GPU wait",
                                 context_switch_scope="current thread; correlation is not causal attribution",
                                 screening=args.samples < 10000))
    if diagnostic is not None:
        summary.update(measurement_status="PROFILE_ONLY", diagnostics=diagnostic.evidence())
        for comparison in summary["paired_deltas"]:
            comparison["threshold_observation"] = "PROFILE_ONLY"
    summary["instrumentation_control"] = (bench.calibrate_timing(args.calibration_samples)
        if args.calibration_samples and mode == "graph" else {"status": "NOT_RUN", "reason": "Graph-only control or disabled"})
    write_new_json(output / "summary.json", summary)
    return summary


def run(args):
    # CPU identity and source checks precede allocating either complete baseline.
    config, metadata, _ = lookup.checkpoint_config(args.checkpoint, require_no_tables=True)
    source_config, source_metadata, index = lookup.checkpoint_config(args.source_checkpoint)
    if source_metadata != metadata:
        raise ValueError("baseline source and consumer config/tokenizer differ")
    package = lookup.load_package(args.shard_json, args.checkpoint)
    oracle = lookup.private_read(args.private_inputs)
    identity = lookup.package_identity(package)
    import torch
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError("selected authorized CUDA device unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    allowed_cpus = set(os.sched_getaffinity(0))
    node, cpu = bind_gpu_local_cpu(args.device, args.cpu_slot)
    if args.source_cache_numa == node:
        raise ValueError("source-cache NUMA must differ from GPU-local HOST_NUMA")
    torch.set_num_threads(1)
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState, HostEngramEmbedding
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import attach_from_package, hash_state_digest
    layout = EngramLayout(config)
    if lookup.layout_metadata(EngramLayout(source_config)) != lookup.layout_metadata(layout):
        raise ValueError("source layout mismatch")
    state = NgramHashState(layout, str(args.checkpoint))
    if hash_state_digest(state) != identity["hash_abi"]:
        raise ValueError("checkpoint hash ABI differs from publication")
    parsed = lookup.validate_oracle(oracle, layout=layout, metadata=metadata, identity=identity,
                                    token_count=len(state.token_map))
    lookup.verify_package_samples(parsed, package, layout)
    headers = source_sample_check(args.source_checkpoint, index, layout, parsed)
    groups = layer_groups(layout.layer_ids, args.layers, args.layer_mode)
    layers = [layer for group in groups for layer in group]
    if (args.samples + args.warmup + 2) * max(args.batches) * layout.n_hash_cols * 8 * max(map(len, groups)) > 512 << 20:
        raise ValueError("preallocated trace exceeds 512MiB")
    spec = importlib.util.find_spec("engram_native_dev_492ecf45")
    if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise RuntimeError("prebuilt genuine native bridge required")
    native = importlib.import_module("engram_native_dev_492ecf45")
    if any(native.resource_counters().values()):
        raise RuntimeError("benchmark must start with zero native resources")
    args.output.mkdir(mode=0o700)
    source_files = [Path(__file__), Path(lookup.__file__), Path(bench.__file__)]
    source_files += [REPO / "rtp_llm/models_py/modules/dsv41" / name
                     for name in ("engram.py", "_engram_triton.py", "engram_storage.py", "engram_bootstrap.py")]
    write_new_json(args.output / "run.json", dict(schema_version=1, identity=identity,
                   source_headers=headers, checkpoint_metadata=metadata, layers=layers,
                   layer_mode=args.layer_mode, layer_groups=groups,
                   batches=args.batches, modes=args.modes, samples=args.samples, seed=args.seed,
                   memory_headroom_bytes=args.memory_headroom_gib << 30,
                   discard_private_source_cache=args.discard_private_source_cache,
                   allow_numa_cache_reclaim=args.allow_numa_cache_reclaim,
                   allow_source_hardlinks=args.allow_source_hardlinks,
                   source_cache_numa=args.source_cache_numa,
                   source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source_files},
                   local_gpu_host_numa=node, local_cpu=cpu,
                   baseline="independent source mmap -> complete _PinnedSharedTable; extra copy is experiment control",
                   scope="lookup components only; serial independent layers; excludes hash/WKV/gate/model/HTTP",
                   **diagnostic_declaration(args)))
    storage, control = None, None
    local, remote, controls = {}, {}, {}
    paths = {}
    completed = []
    try:
        storage = attach_from_package(package, layout=layout, hash_state=state, device=device)
        rows_by_layer = dict(zip(layout.layer_ids, layout.num_embeddings))
        for group in groups:
            numa_by_layer = {}
            planned_bytes = sum(rows_by_layer[layer] * (layout.head_dim + layout.head_dim // 32) for layer in group)
            if shutil.disk_usage("/dev/shm").free < planned_bytes:
                raise RuntimeError("insufficient /dev/shm for all selected layers together")
            for layer in group:
                table_bytes = rows_by_layer[layer] * (layout.head_dim + layout.head_dim // 32)
                files = source_paths(args.source_checkpoint, index, layer)
                source_bytes = sum(path.stat().st_size for path in files)
                if args.discard_private_source_cache:
                    advice = private_source_cache_advice(args.source_checkpoint, files,
                                                        allow_hardlinks=args.allow_source_hardlinks)
                    emit("PRIVATE_SOURCE_CACHE_ADVICE", layer=layer, phase="before_load", files=advice)
                snapshot = memory_snapshot(node)
                write_new_json(args.output / f"memory-layer{layer}-before.json", snapshot)
                if args.source_cache_numa is None:
                    check_memory_budget(snapshot, table_bytes=table_bytes, source_bytes=source_bytes,
                                        headroom_bytes=args.memory_headroom_gib << 30, before=True,
                                        allow_numa_cache_reclaim=args.allow_numa_cache_reclaim)
                emit("BASELINE_LOADING", layer=layer, logical_bytes=table_bytes, source_file_bytes=source_bytes)
                if args.source_cache_numa is None:
                    control = make_local_table(args.source_checkpoint, index, layout, layer, device)
                else:
                    control, source_cache = construct_with_source_cache(
                        source=args.source_checkpoint, files=files, source_node=args.source_cache_numa,
                        gpu_node=node, allowed_cpus=allowed_cpus, table_bytes=table_bytes,
                        headroom_bytes=args.memory_headroom_gib << 30,
                        construct=lambda: make_local_table(args.source_checkpoint, index, layout, layer, device))
                controls[layer] = control
                if args.source_cache_numa is not None:
                    write_new_json(args.output / f"source-cache-layer{layer}.json", source_cache)
                    emit("SOURCE_CACHE_NUMA_VERIFIED", layer=layer, source_node=args.source_cache_numa,
                         gpu_host_numa=node, complete_source_verified=True, eviction_prevented=False)
                if args.discard_private_source_cache:
                    advice = private_source_cache_advice(args.source_checkpoint, files,
                                                        allow_hardlinks=args.allow_source_hardlinks)
                    emit("PRIVATE_SOURCE_CACHE_ADVICE", layer=layer, phase="after_load", files=advice)
                snapshot = memory_snapshot(node)
                write_new_json(args.output / f"memory-layer{layer}-after.json", snapshot)
                check_memory_budget(snapshot, table_bytes=table_bytes, source_bytes=source_bytes,
                                    headroom_bytes=args.memory_headroom_gib << 30, before=False)
                numa_by_layer[layer] = parse_numa_pages(Path("/proc/self/numa_maps").read_text(), control.storage.data_ptr(), node)
                local[layer], aliases = make_local_embedding(control, HostEngramEmbedding, torch)
                write_new_json(args.output / f"baseline-alias-layer{layer}.json", aliases)
                remote[layer] = HostEngramEmbedding.from_storage(storage, layer)
                emit("BASELINE_READY", layer=layer, numa=numa_by_layer[layer])
            for batch in args.batches:
                ids_by_layer, traces = {}, {}
                for layer in group:
                    ids_by_layer[layer] = torch.empty((batch, layout.n_hash_cols), dtype=torch.int64, device=device)
                    generator = torch.Generator().manual_seed(args.seed + layer * 100 + batch)
                    traces[layer] = torch.randint(rows_by_layer[layer], (args.samples + args.warmup + 2, batch, layout.n_hash_cols),
                                                 dtype=torch.int64, generator=generator).pin_memory()
                if len(group) == 1:
                    # Preserve single-layer's direct call; no tuple/chain wrapper
                    # is added to its measured eager dispatch.
                    paths = {"local_raw": lambda embedding=local[group[0]], ids=ids_by_layer[group[0]]: embedding(ids, device),
                             "remote_raw": lambda embedding=remote[group[0]], ids=ids_by_layer[group[0]]: embedding(ids, device)}
                else:
                    paths = {"local_raw": lambda: invoke_lookup_chain(local, group, ids_by_layer, device),
                             "remote_raw": lambda: invoke_lookup_chain(remote, group, ids_by_layer, device)}
                def update(sample, distribution):
                    for layer in group:
                        trace = traces[layer]
                        ids_by_layer[layer].copy_(trace[0 if distribution == "hot" else sample % len(trace)], non_blocking=True)
                def verify(sample, distribution, outputs):
                    per_layer = split_layer_outputs(outputs, group)
                    for layer in group:
                        trace = traces[layer]
                        matrix = trace[0 if distribution == "hot" else sample % len(trace)].tolist()
                        expected = torch.tensor(source_rows(args.source_checkpoint, index, layout, layer, matrix),
                                                dtype=torch.float32).to(torch.bfloat16)
                        for name, value in per_layer[layer].items():
                            actual = value.cpu()
                            if actual.dtype != torch.bfloat16 or not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                                raise AssertionError(f"{name} measured layer output differs from independent source BF16 oracle")
                for mode in args.modes:
                    label = f"layer{group[0]}" if len(group) == 1 else "layers" + "-".join(map(str, group))
                    output = args.output / f"{label}-M{batch}-{mode}"
                    output.mkdir(mode=0o700)
                    case = dict(layer_id=group[0] if len(group) == 1 else None,
                                layer_ids_measured=list(group), layer_mode=args.layer_mode,
                                actual_model_engram_layers=list(layout.layer_ids), layer_count_measured=len(group),
                                covers_all_model_engram_layers=set(group) == set(layout.layer_ids),
                                layer_dependency="independent inputs, same-stream serial lookup; no intervening model computation",
                                batch=batch, heads=layout.n_hash_cols, dim=layout.head_dim,
                                table_rows=rows_by_layer[group[0]] if len(group) == 1 else {str(layer): rows_by_layer[layer] for layer in group},
                                table_bytes=planned_bytes, mode=mode,
                                distribution="uniform rows over complete actual table; not a production token distribution",
                                trace_seed=args.seed + group[0] * 100 + batch if len(group) == 1 else
                                    {str(layer): args.seed + layer * 100 + batch for layer in group},
                                trace_sha256=hashlib.sha256(memoryview(traces[group[0]].numpy())).hexdigest() if len(group) == 1 else
                                    {str(layer): hashlib.sha256(memoryview(traces[layer].numpy())).hexdigest() for layer in group},
                                scope="IDs ready -> all selected lookup BF16 outputs ready; no hash/WKV/gate/model/HTTP",
                                baseline_numa=numa_by_layer[group[0]] if len(group) == 1 else
                                    {str(layer): numa_by_layer[layer] for layer in group},
                                torch=torch.__version__, cuda=torch.version.cuda,
                                gpu_name=torch.cuda.get_device_properties(device).name,
                                capability=list(torch.cuda.get_device_capability(device)), identity=identity)
                    emit("CASE_START", layers=list(group), batch=batch, mode=mode, samples=args.samples,
                         **diagnostic_declaration(args))
                    result = benchmark_case(paths=paths, storage=storage, update=update, verify=verify,
                                            args=args, metadata=case, output=output)
                    completed.append(output.name)
                    emit("CASE_COMPLETE", case=output.name, paired_deltas=result["paired_deltas"],
                         **diagnostic_declaration(args))
                paths.clear()
                del ids_by_layer, traces, update, verify
            torch.cuda.synchronize(device)
            for embedding in remote.values():
                embedding.release_external_views()
            remote.clear()
            local.clear()
            gc.collect()
            for control in controls.values():
                close_local(control, torch)
            controls.clear()
            control = None
            gc.collect()
        storage.close()
        counters = native.resource_counters()
        if any(counters.values()):
            raise RuntimeError("native resources remain after ordered cleanup")
        write_new_json(args.output / "complete.json", dict(cases=completed, counters=counters,
                       decode_step_slo="NOT_MEASURED", layer_mode=args.layer_mode,
                       scope="selected-layer serial lookup component; no inter-layer model computation",
                       **diagnostic_declaration(args)))
        emit("CLOSED", cases=completed, counters=counters, **diagnostic_declaration(args))
    except BaseException as error:
        retain_failed(control, *controls.values(), local, remote, storage, paths, error)
        emit("FAILED_RETAINED_UNTIL_PROCESS_EXIT", error_type=type(error).__name__, completed=completed)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source-checkpoint", "bundle", "private-inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batches", default="1")
    parser.add_argument("--layers", default="first")
    parser.add_argument("--layer-mode", choices=("single", "together"), default="single")
    parser.add_argument("--memory-headroom-gib", type=int, default=16)
    parser.add_argument("--allow-numa-cache-reclaim", action="store_true",
                        help="Admission may count clean inactive NUMA file cache; "
                             "not a reservation. Post-load free-memory and NUMA checks remain strict.")
    parser.add_argument("--discard-private-source-cache", action="store_true")
    parser.add_argument("--allow-source-hardlinks", action="store_true",
                        help="Operator confirms all source hardlinks are experiment-owned; "
                             "DONTNEED affects every alias. Requires source cache discard.")
    parser.add_argument("--source-cache-numa", type=int, default=None)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--distributions", default="hot,random")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4102101)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-slot", type=int, default=0)
    parser.add_argument("--l2-disturb", action="store_true")
    parser.add_argument("--threshold-us", type=float, default=100.0)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument("--diagnostic-nvtx", action="store_true")
    parser.add_argument("--diagnostic-cuda-profiler", action="store_true")
    try:
        args = check_options(parser.parse_args(argv))
        run(args)
        return 0
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        # pybind overload errors can include argument reprs, including opaque
        # Fabric capabilities. Never publish arbitrary exception text or codes.
        emit("FAILED", error_type=type(error).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
