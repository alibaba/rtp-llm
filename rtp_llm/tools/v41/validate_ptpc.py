"""Strict full-checkpoint comparison of the actual Flash1017 PPU loader.

Run with the ptpc_alignment_test Bazel target. Weights remain read-only.
Large unchanged tables are streamed; only one dense descriptor is retained.
Reference payloads are read independently from safetensors file offsets.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time

import numpy as np
import torch

from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.models.deepseek_v41 import DeepSeekV41, DeepSeekV41DSpark
from rtp_llm.models_py.modules.dsv41.ppu.dspark_weights import DeepSeekV41PPUDSparkWeight
from rtp_llm.models_py.modules.dsv41.ppu.model_weights import DeepSeekV41PPUWeight, PPUPTPCWeight
from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC, quant_mode
from rtp_llm.models_py.modules.dsv41.ppu.weight import _CheckpointSource, _load_dense
from rtp_llm.ops import HWKernelConfig, KVCacheConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase


class Checkpoint:
    def __init__(self, root, stack):
        self.root = Path(root).resolve(strict=True)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.files, self.headers, self.stats = {}, {}, {}
        for filename in sorted(set(self.index.values())):
            path = (self.root / filename).resolve(strict=True)
            if self.root not in path.parents:
                raise ValueError("checkpoint index escapes its directory")
            f = stack.enter_context(path.open("rb"))
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
            self.files[filename] = (f, 8 + length)
            self.headers.update({k: v for k, v in header.items() if k != "__metadata__"})
            self.stats[filename] = (path.stat().st_size, path.stat().st_mtime_ns)
        if set(self.index) - set(self.headers):
            raise ValueError("checkpoint index contains absent tensors")

    def read(self, name, offset, size):
        f, base = self.files[self.index[name]]
        f.seek(base + self.headers[name]["data_offsets"][0] + offset)
        value = f.read(size)
        if len(value) != size:
            raise ValueError(f"truncated checkpoint tensor {name}")
        return value

    def verify_unchanged(self):
        for filename, expected in self.stats.items():
            st = (self.root / filename).stat()
            if (st.st_size, st.st_mtime_ns) != expected:
                raise RuntimeError(f"checkpoint changed during comparison: {filename}")


class Comparison:
    def __init__(self, source, reference, stream):
        self.source, self.reference, self.stream = source, reference, stream
        self.seen, self.failures, self.bytes = set(), 0, 0

    def compare(self, name, dtype, shape, read_actual, size, origin):
        if name in self.seen:
            raise ValueError(f"tensor compared twice: {name}")
        header = self.reference.headers[name]
        if header["dtype"] != dtype or header["shape"] != list(shape):
            raise ValueError(f"logical dtype/shape mismatch: {name}: {dtype} {shape} vs {header}")
        if size != header["data_offsets"][1] - header["data_offsets"][0]:
            raise ValueError(f"logical byte length mismatch: {name}")
        digest, expected_digest = hashlib.sha256(), hashlib.sha256()
        mismatch, first = 0, None
        for offset in range(0, size, 8 * 1024 * 1024):
            length = min(8 * 1024 * 1024, size - offset)
            actual = read_actual(offset, length)
            expected = self.reference.read(name, offset, length)
            digest.update(actual)
            expected_digest.update(expected)
            if actual != expected:
                positions = np.flatnonzero(np.frombuffer(actual, dtype=np.uint8) != np.frombuffer(expected, dtype=np.uint8))
                mismatch += len(positions)
                if first is None:
                    i = int(positions[0])
                    first = {"byte_offset": offset + i, "actual": actual[i], "expected": expected[i]}
        record = dict(name=name, dtype=dtype, shape=list(shape), bytes=size, origin=origin,
                      actual_sha256=digest.hexdigest(), reference_sha256=expected_digest.hexdigest(),
                      mismatched_bytes=mismatch, first_mismatch=first)
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()
        self.seen.add(name)
        self.bytes += size
        self.failures += int(mismatch != 0)
        if mismatch or len(self.seen) % 1000 == 0:
            print(json.dumps(dict(compared=len(self.seen), failures=self.failures, last=record)), flush=True)

    def tensor(self, name, value, origin):
        raw = memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy()).cast("B")
        dtype = {torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32", torch.bfloat16: "BF16"}[value.dtype]
        self.compare(name, dtype, value.shape, lambda offset, size: raw[offset:offset + size].tobytes(), len(raw), origin)


class ObservedSource(DatabaseTensorSource):
    def __init__(self, database, comparison):
        super().__init__(database)
        self.comparison = comparison

    def observe_conversion(self, name, value, scale):
        self.comparison.tensor(name, value, "production_PPUPTPCWeight")
        self.comparison.tensor(name.removesuffix(".weight") + ".scale", scale, "production_PPUPTPCWeight")


def metadata_config(model_cls, checkpoint):
    """Use the explicit reference parser for this CPU-only checkpoint audit.

    Backend selection rejects PPU on CPU. Restore the requested PPU backend
    before creating the production weight descriptors; no model is executed.
    """
    previous = os.environ.get("RTP_V41_BACKEND")
    os.environ["RTP_V41_BACKEND"] = "reference"
    try:
        return model_cls._create_config(str(checkpoint))
    finally:
        if previous is None:
            os.environ.pop("RTP_V41_BACKEND", None)
        else:
            os.environ["RTP_V41_BACKEND"] = previous


def validate(source_path, reference_path, output):
    if quant_mode() != MXFP4_FP8_PTPC:
        raise ValueError("full alignment requires the explicit PTPC quantization mode")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="RUNNING", started=time.time(), source=str(source_path), reference=str(reference_path))
    snapshot_path = Path(__file__).resolve().parents[3] / ".v41_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    report["runtime_identity"] = {k: snapshot[k] for k in ("snapshot_id", "public_commit", "internal_commit", "source_sha256")}
    comparison = None
    try:
        with ExitStack() as stack:
            source, reference = Checkpoint(source_path, stack), Checkpoint(reference_path, stack)
            if source.root == reference.root:
                raise ValueError("source and reference must be different checkpoints")
            report["input_identity"] = {
                label: {name: hashlib.sha256((ckpt.root / name).read_bytes()).hexdigest()
                        for name in ("config.json", "model.safetensors.index.json")}
                for label, ckpt in (("source", source), ("reference", reference))
            }
            removed = {f"layers.{i}.engram.wkv.scale" for i in (1, 14)}
            if set(source.index) - set(reference.index) != removed or set(reference.index) - set(source.index):
                raise ValueError("unexpected source/reference key difference")
            stream = stack.enter_context((output / "tensors.jsonl").open("x"))
            comparison = Comparison(source, reference, stream)
            database, reference_db = CkptDatabase(str(source.root)), CkptDatabase(str(reference.root))
            observed = ObservedSource(database, comparison)
            dense_count = 0
            for model_cls, weight_cls in ((DeepSeekV41, DeepSeekV41PPUWeight),
                                          (DeepSeekV41DSpark, DeepSeekV41PPUDSparkWeight)):
                config = metadata_config(model_cls, source.root)
                config.ckpt_path = str(source.root)
                descriptor = weight_cls(config, ParallelismConfig(), HWKernelConfig(), KVCacheConfig())
                # Full checkpoint audit, independent of this process's serving role.
                descriptor.role_type = None
                descriptor.process_meta_from_ckpt(database.pretrain_file_list)
                info = descriptor.get_weight_info()
                load_config = descriptor.create_load_config(torch.bfloat16, database)
                entries = [(None, w) for w in info.weights]
                entries += [(i, w) for i, row in enumerate(info.layer_weights) for w in row]
                for layer_id, entry in entries:
                    if not isinstance(entry, PPUPTPCWeight):
                        continue
                    actual = entry.load(observed, layer_id, "cpu", load_config)
                    expected = entry.load(DatabaseTensorSource(reference_db), layer_id, "cpu", load_config)
                    for key, value in actual.items():
                        if not torch.equal(value.contiguous().view(torch.uint8), expected[key].contiguous().view(torch.uint8)):
                            raise ValueError(f"production logical merge mismatch: {layer_id} {key}")
                    dense_count += len(entry.weights)
                    del actual, expected
            if dense_count != 353:
                raise ValueError(f"expected 353 production dense projections, got {dense_count}")
            native_source = _CheckpointSource(str(source.root))
            for layer in (1, 14):
                prefix = f"layers.{layer}.engram.wkv"
                value = _load_dense(native_source, prefix).weight
                comparison.tensor(prefix + ".weight", value, "production_Engram_load_dense")
                del value
            for name in sorted(set(reference.index) - comparison.seen):
                a, b = source.headers[name], reference.headers[name]
                expert = ".ffn.experts." in name
                dtype = "U8" if expert and a["dtype"] in ("I8", "F8_E8M0", "U8") else a["dtype"]
                size = a["data_offsets"][1] - a["data_offsets"][0]
                comparison.compare(name, dtype, a["shape"],
                                   lambda offset, size, key=name: source.read(key, offset, size), size,
                                   "native_byte_reinterpret" if expert else "unchanged_checkpoint")
            source.verify_unchanged()
            reference.verify_unchanged()
            if comparison.seen != set(reference.index):
                raise ValueError("full tensor coverage is incomplete")
            report.update(tensors=len(comparison.seen), dense_projections=dense_count,
                          compared_bytes=comparison.bytes, failed_tensors=comparison.failures,
                          removed_source_keys=sorted(removed), status="FAIL" if comparison.failures else "PASS")
            if comparison.failures:
                raise ValueError(f"{comparison.failures} tensors differ byte-for-byte")
    except BaseException as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        if comparison is not None:
            report.update(tensors=len(comparison.seen), failed_tensors=comparison.failures,
                          compared_bytes=comparison.bytes)
        report["finished"] = time.time()
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


def check_audit(expected_report, audit_directory, output):
    expected_report, audit_directory, output = map(Path, (expected_report, audit_directory, output))
    baseline = json.loads((expected_report / "summary.json").read_text())
    if baseline.get("status") != "PASS" or baseline.get("tensors") != 96083:
        raise ValueError("audit reference must be a complete passing checkpoint comparison")
    expected, all_names = {}, set()
    for line in (expected_report / "tensors.jsonl").open():
        row = json.loads(line)
        if row["name"] in all_names:
            raise ValueError("duplicate tensor in the complete reference report")
        all_names.add(row["name"])
        if row["origin"].startswith("production_") or ".ffn.experts." in row["name"]:
            expected[row["name"]] = row
    if len(all_names) != 96083 or len(expected) != 95172:
        raise ValueError("reference report does not cover the full Flash1017 tensor contract")
    required = set()
    for name in expected:
        expert = re.search(r"\.ffn\.experts\.(\d+)\.", name)
        owners = (int(expert[1]) // (16 if name.startswith("mtp.") else 48),) if expert else range(8)
        required.update((rank, name) for rank in owners)
    seen, seen_records, ranks, failures = set(), set(), set(), []
    for path in sorted(audit_directory.glob("weights-rank*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            ref = expected.get(row["name"])
            key = (row["rank"], row["name"])
            if key in seen_records or key not in required or int(path.stem[-5:]) != row["rank"]:
                failures.append(dict(reason="duplicate or incorrect EP owner", record=row))
            seen_records.add(key)
            if row.get("source_sha256") != baseline["runtime_identity"]["source_sha256"] or ref is None or (row["dtype"], row["shape"], row["sha256"]) != (
                ref["dtype"], ref["shape"], ref["reference_sha256"]
            ):
                failures.append(row)
            seen.add(row["name"])
            ranks.add(row["rank"])
    missing = sorted(required - seen_records)
    report = dict(status="FAIL" if missing or failures or ranks != set(range(8)) else "PASS",
                  scope="actual EP8 loader converted dense/Engram and native expert buffers before packing",
                  expected_tensors=len(expected), actual_unique_tensors=len(seen), ranks=sorted(ranks),
                  expected_rank_records=len(required), actual_rank_records=len(seen_records),
                  missing=missing, mismatches=failures, baseline_identity=baseline["runtime_identity"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("missing", "mismatches")}), flush=True)
    if report["status"] != "PASS":
        raise ValueError(f"actual loader audit failed: missing={len(missing)}, mismatches={len(failures)}, ranks={ranks}")


def check_layouts(left, right, output):
    """Compare actual packed MoE and merged Dense consumer buffers, EP8 A/B."""
    inventories, identities = [], set()
    for directory in (Path(left), Path(right)):
        inventory = {}
        counts = {}
        for path in sorted(directory.glob("layouts-rank*.jsonl")):
            for line in path.open():
                row = json.loads(line)
                key = row["rank"], row["name"]
                if key in inventory or int(path.stem[-5:]) != row["rank"] or row["origin"] != "production_backend_layout":
                    raise ValueError("duplicate or invalid backend layout receipt")
                inventory[key] = row
                identities.add(row.get("source_sha256"))
                prefix = ":".join(row["name"].split(":")[:2])
                count_key = row["rank"], prefix
                counts[count_key] = counts.get(count_key, 0) + 1
        expected_counts = {(rank, prefix): count for rank in range(8) for prefix, count in (
            ("dense:40", 576), ("dense:3", 44), ("moe:384", 160), ("moe:128", 12),
        )}
        inventories.append((inventory, counts == expected_counts))
    a, b = (item[0] for item in inventories)
    missing = sorted(a.keys() ^ b.keys())
    differences = [key for key in a.keys() & b.keys() if any(a[key][field] != b[key][field] for field in ("dtype", "shape", "stride", "storage_offset", "sha256"))]
    valid = all(item[1] for item in inventories) and len(identities) == 1 and None not in identities
    report = dict(status="PASS" if valid and not missing and not differences else "FAIL",
                  scope="actual merged Dense FP8 and backend-packed MoE payload/scale buffers, main40+MTP3 EP8",
                  left_records=len(a), right_records=len(b), expected_records=6336,
                  complete_geometry=valid, missing=missing, mismatches=differences,
                  source_sha256=sorted(str(value) for value in identities))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "layout-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise ValueError("actual backend layout comparison failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--expected-report", type=Path)
    parser.add_argument("--compare-layout-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.compare_layout_dir:
        if args.audit_dir is None:
            parser.error("--compare-layout-dir requires --audit-dir")
        check_layouts(args.audit_dir, args.compare_layout_dir, args.output)
    elif args.audit_dir:
        if args.expected_report is None:
            parser.error("--audit-dir requires --expected-report")
        check_audit(args.expected_report, args.audit_dir, args.output)
    else:
        if args.source is None or args.reference is None:
            parser.error("checkpoint validation requires --source and --reference")
        validate(args.source, args.reference, args.output)


if __name__ == "__main__":
    main()
