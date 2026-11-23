"""Run a caller-supplied native metadata replay executable."""
import argparse
import json
import struct
import os
import shutil
import subprocess
from pathlib import Path

from prepare import digest
from bundles import legacy_view, unpack
from tempfile import TemporaryDirectory


def token_prefix(path, count):
    with path.open("rb") as f:
        magic, base, size, _, name_len = struct.unpack("<8sIQQI", f.read(32))
        if magic != b"BTCPFX01" or count > size:
            raise ValueError("invalid chunk header")
        f.seek(name_len, 1)
        total = 0
        for _ in range(count):
            _, length, rid_len = struct.unpack("<QII", f.read(16))
            total += length
            f.seek(rid_len + length // base * 8, 1)
        return total


def prefix(data, chunks, count):
    total = 0
    for chunk in chunks:
        if count == 0:
            break
        if count < chunk["requests"]:
            return total + token_prefix(data / chunk["file"], count)
        count -= chunk["requests"]
        total += chunk["input_tokens"]
    if count:
        raise ValueError("requested range exceeds manifest")
    return total


def main(a, scratch):
    data_source = a.data.resolve().name
    manifest = a.data / "manifest.json"
    trace = json.loads(manifest.read_bytes())
    if trace["format"] != "BTCPFX01":
        raise ValueError("unsupported trace format")
    if any(c["file"] != f"{i:06}.btcbin" for i, c in enumerate(trace["chunks"])):
        raise ValueError("unexpected chunk names")
    if any("bundle" in c for c in trace["chunks"]):
        unpack(a.data, trace["chunks"], scratch)
        a.data = scratch
    for i, c in enumerate(trace["chunks"]):
        if c["file"] != f"{i:06}.btcbin" or digest(a.data / c["file"]) != c["sha256"]:
            raise ValueError(f"chunk {i} failed identity check")
        with (a.data / c["file"]).open("rb") as stream:
            magic, base, count, tokens, name_len = struct.unpack("<8sIQQI", stream.read(32))
            if (magic, base, count, tokens) != (b"BTCPFX01", trace["base_tokens"], c["requests"], c["input_tokens"]):
                raise ValueError(f"chunk {i} header differs from manifest")
            if name_len > 1048576 or stream.read(name_len).decode() != trace["instance_id"]:
                raise ValueError(f"chunk {i} cache domain mismatch")
    if sum(c["requests"] for c in trace["chunks"]) != trace["requests"]:
        raise ValueError("trace request total mismatch")
    if a.warmup < 0 or a.measure < 1 or a.pause_after is not None and not 0 < a.pause_after < a.measure:
        raise ValueError("invalid replay range")
    if a.warmup + a.measure > trace["requests"]:
        raise ValueError("insufficient requests in the trace")
    end = a.measure
    expected = prefix(a.data, trace["chunks"], a.warmup + end) - prefix(a.data, trace["chunks"], a.warmup)
    binary = a.backend.resolve(strict=True)
    if trace["base_tokens"] != 256 or any(
        c["requests"] != 100000 for c in trace["chunks"][:-1]
    ) or trace["chunks"][-1]["requests"] > 100000:
        probe = subprocess.run([str(binary), "--btcpfx-reader-version"], capture_output=True, text=True)
        if probe.returncode or probe.stdout.strip() != "2":
            if trace["base_tokens"] != 256:
                raise ValueError("legacy backend supports only 256 tokens/block; rebuild the header-driven reader")
            view = scratch / "legacy-view"
            legacy_view(a.data, trace["chunks"], view, trace["base_tokens"], trace["instance_id"])
            a.data = view
    config = a.config.read_bytes()
    identity = dict(trace_sha256=digest(manifest), backend_sha256=digest(binary),
                    config_sha256=digest(a.config), revision=a.revision, warmup=a.warmup, measure=a.measure)
    a.out.mkdir(parents=True, exist_ok=False)
    ident = a.out / "identity.json"
    ident.write_text(json.dumps(identity, sort_keys=True) + "\n")
    cfg = a.out / "parameters.json"
    cfg.write_bytes(config)
    native = a.out / "native"
    cmd = [str(binary), str(cfg), str(a.data), str(native), "--identity", str(ident),
           "--requests", str(a.warmup + a.measure), "--warmup", str(a.warmup)]
    if a.pause_after is not None:
        cmd.extend(["--pause-after", str(a.warmup + a.pause_after)])
    subprocess.run(cmd, check=True)
    t = json.loads((native / "summary.json").read_bytes())["total"]
    if (t["requests"], t["input_tokens"]) != (end, expected):
        raise ValueError("backend request or input-token denominator mismatch")
    if not 0 <= t["joint_tokens"]["sum"] <= t["full_tokens"]["sum"] <= expected:
        raise ValueError("invalid aggregate match lengths")
    if (native / "match-lengths.u64").stat().st_size != 16 * end:
        raise ValueError("unexpected number of per-request match pairs")
    metadata = dict(source=data_source, instance=trace["instance_id"], base_tokens=trace["base_tokens"])
    metadata.update(trace.get("report_environment", {}))
    (a.out / "report-environment.json").write_text(json.dumps(metadata, ensure_ascii=False))
    arrival = native / "arrival-hits.u64"
    if arrival.exists():
        if arrival.stat().st_size != 48 * end:
            raise ValueError("unexpected number of arrival-time records")
        stages = a.out / "report-stages"
        stages.mkdir()
        dst = stages / "0000000000000000.u64"
        try:
            os.link(arrival, dst)
        except OSError:
            shutil.copyfile(arrival, dst)
    (a.out / "SUCCESS").touch()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("backend", "data", "config", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--measure", type=int, required=True)
    p.add_argument("--pause-after", type=int, help="pause after N measured requests, keeping this process alive")
    args = p.parse_args()
    with TemporaryDirectory() as tmp:
        main(args, Path(tmp))
