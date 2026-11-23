"""Stream one ordered cache domain into BTCPFX01 chunks."""
import argparse
import gzip
import hashlib
import json
import os
import re
import struct
import sys
import subprocess
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

CHUNK = 1_000_000
MASK = (1 << 64) - 1


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


@contextmanager
def lines(path, sample=False):
    if path.suffix == ".zst":
        with subprocess.Popen(["zstd", "-dc", "--", str(path)], stdout=subprocess.PIPE) as proc:
            try:
                yield proc.stdout
            finally:
                proc.stdout.close()
            status = proc.wait()
            if status and not (sample and status in (-13, 141)):
                raise ValueError(f"zstd failed for {path}")
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            yield stream


def bailian(row, base, mode):
    length = row["input_length"]
    hashes = row["input_block_hash_ids"]
    if type(length) is not int or not isinstance(hashes, list) or len(hashes) not in {
        length // base, (length + base - 1) // base
    }:
        raise ValueError("inconsistent input length and hash count")
    keys = []
    prefix = 0
    for value in hashes[:length // base]:
        if not isinstance(value, str) or not re.fullmatch("[0-9a-fA-F]{16}", value):
            raise ValueError("expected 16 hexadecimal digits per hash")
        key = int(value, 16)
        if mode == "raw":
            prefix ^= (key + 0x9e3779b97f4a7c15 + (prefix << 12) + (prefix >> 32)) & MASK
            key = prefix
        keys.append(key if key < 1 << 63 else key - (1 << 64))
    ts = Decimal(str(row["timestamp"])) * 1000000000
    if ts != ts.to_integral_value():
        raise ValueError("timestamp finer than nanoseconds")
    return int(ts), length, keys


def resolve_format(args):
    """A repeated key with distinct predecessor identities contradicts prefix hashing."""
    from itertools import islice
    mode = args.hash_mode
    with lines(args.inputs[0], sample=True) as stream:
        row = json.loads(next(stream), parse_float=Decimal)
    base = row.get("block_tokens")
    args.base_evidence = "log field block_tokens"
    if base is None and args.format == "bailian":
        # A producer may include or omit a key for the incomplete tail.
        low, high = 1, 2**31 - 1
        for path in args.inputs:
            with lines(path, sample=True) as stream:
                for line in islice(stream, 10000):
                    record = json.loads(line)
                    length = record["input_length"]
                    count = len(record["input_block_hash_ids"])
                    if type(length) is not int or length <= 0:
                        raise ValueError("invalid input_length")
                    low = max(low, length // (count + 1) + 1)
                    if count > 1:
                        high = min(high, (length - 1) // (count - 1))
                    explicit = record.get("block_tokens")
                    if explicit is not None:
                        if type(explicit) is not int:
                            raise ValueError("invalid block_tokens declaration")
                        low, high = max(low, explicit), min(high, explicit)
                    if low >= high:
                        break
            if low >= high:
                break
        if low != high:
            raise ValueError(f"cannot establish one block size from trace (bounds {low}..{high}); obtain producer declaration")
        base = low
        args.base_evidence = "unique size consistent with sampled log lengths, key counts and declarations"
    if type(base) is not int or not 0 < base < 2**31:
        raise ValueError("missing or invalid block size in log")
    if args.base_tokens is not None and args.base_tokens != base:
        raise ValueError(f"--base-tokens conflicts with log: expected {base}")
    if args.format == "normalized":
        return base, "prefix", "normalized schema"
    declared = row.get("key_form")
    if mode and declared and mode != declared:
        raise ValueError("hash mode conflicts with source declaration")
    if declared or mode:
        return base, declared or mode, "record declaration" if declared else "explicit producer setting"
    predecessors = {}
    with lines(args.inputs[0], sample=True) as stream:
        for line in islice(stream, 10000):
            row = json.loads(line)
            previous = None
            keys = row["input_block_hash_ids"]
            for key in keys[:row["input_length"] // base]:
                key = key.lower()
                if key in predecessors and predecessors[key] != previous:
                    return base, "raw", "observed identical key following different predecessors"
                if len(predecessors) < 65536:
                    predecessors[key] = previous
                previous = key
    raise ValueError("hash mode ambiguous: provide producer key_form or --hash-mode raw/prefix")


def prepare(args, dest):
    domain = args.domain.encode()
    if not 0 < len(domain) <= 1048576 or not 0 < args.base_tokens < 2**31:
        raise ValueError("invalid domain or block size")
    chunks = []
    previous = count = total = 0
    stream = None
    try:
        for path in args.inputs:
            with lines(path) as source:
                for number, line in enumerate(source, 1):
                    try:
                        row = json.loads(line, parse_float=Decimal)
                        if args.format == "bailian":
                            if row.get("key_form", args.hash_mode) != args.hash_mode or row.get("block_tokens", args.base_tokens) != args.base_tokens:
                                raise ValueError("conflicting hash mode or block granularity")
                            ts, length, keys = bailian(row, args.base_tokens, args.hash_mode)
                        else:
                            if (row["key_form"], row["instance_id"], row["block_tokens"]) != (
                                "prefix", args.domain, args.base_tokens
                            ):
                                raise ValueError("domain, granularity or key form mismatch")
                            ts, length, keys = row["timestamp_ns"], row["input_len"], row["keys"]
                        rid = row["request_id"].encode()
                        if not (type(ts) is int and previous <= ts < 2**64 and ts > 0
                                and type(length) is int and 0 < length < 2**31
                                and isinstance(keys, list) and len(keys) == length // args.base_tokens
                                and len(rid) <= 1048576 and all(type(k) is int for k in keys)):
                            raise ValueError("invalid record or timestamp order")
                        record = struct.pack("<QII", ts, length, len(rid)) + rid + struct.pack(f"<{len(keys)}q", *keys)
                    except (KeyError, ArithmeticError, ValueError, TypeError, AttributeError, struct.error) as e:
                        raise ValueError(f"{path}:{number}: {e}") from e
                    if stream is None:
                        chunk = dest / f"{len(chunks):06}.btcbin"
                        stream = chunk.open("w+b")
                        stream.write(struct.pack("<8sIQQI", b"BTCPFX01", args.base_tokens, 0, 0, len(domain)) + domain)
                        n = tokens = 0
                    stream.write(record)
                    n += 1
                    tokens += length
                    total += length
                    count += 1
                    previous = ts
                    if n == args.requests_per_file:
                        finish(stream, chunk, n, tokens, chunks)
                        stream = None
        if not count:
            raise ValueError("empty trace")
        if stream:
            finish(stream, chunk, n, tokens, chunks)
    finally:
        if stream and not stream.closed:
            stream.close()
    manifest = dict(format="BTCPFX01", base_tokens=args.base_tokens, instance_id=args.domain,
                    hash_mode=args.hash_mode or "prefix", hash_evidence=args.hash_evidence, requests=count, input_tokens=total, chunks=chunks)
    manifest["report_environment"] = {"source": ", ".join(path.name for path in args.inputs)}
    for key in ("cluster", "prefill_ip", "capture_time"):
        if getattr(args, key, None):
            manifest["report_environment"][key] = getattr(args, key)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def finish(stream, path, n, tokens, chunks):
    stream.seek(12)
    stream.write(struct.pack("<QQ", n, tokens))
    stream.close()
    chunks.append(dict(file=path.name, requests=n, input_tokens=tokens, sha256=digest(path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", choices=["bailian", "normalized"], required=True)
    p.add_argument("--hash-mode", choices=["raw", "prefix"])
    p.add_argument("--base-tokens", type=int)
    p.add_argument("--domain", required=True)
    p.add_argument("--cluster", help="source cluster configuration for the report")
    p.add_argument("--prefill-ip", help="prefill node recorded by the data collector")
    p.add_argument("--capture-time", help="documented capture interval, including timezone")
    p.add_argument("--requests-per-file", type=int, default=CHUNK)
    a = p.parse_args()
    if a.requests_per_file < 1:
        p.error("--requests-per-file must be positive")
    if a.format == "normalized" and a.hash_mode:
        p.error("specify --hash-mode only for Bailian input; normalized input requires prefix keys")
    print("预处理：检查来源粒度和 hash 类型…", file=sys.stderr, flush=True)
    try:
        a.base_tokens, a.hash_mode, a.hash_evidence = resolve_format(a)
    except (ValueError, StopIteration) as e:
        p.error(str(e) or "empty trace")
    print(f"判定：{a.base_tokens} tokens/block（{a.base_evidence}），hash={a.hash_mode}，依据={a.hash_evidence}", file=sys.stderr)
    if a.hash_mode not in ("raw", "prefix"):
        p.error("invalid hash mode")
    if a.out.exists():
        p.error("output already exists")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out.parent) as tmp:
        staged = Path(tmp) / "data"
        staged.mkdir()
        prepare(a, staged)
        os.rename(staged, a.out)
    receipt = json.loads((a.out / "manifest.json").read_text())
    print(f"完成：{receipt['requests']} 条请求；{a.base_tokens} tokens/block；输入 hash={a.hash_mode}；输出 hash=prefix；清单 {a.out / 'manifest.json'}")
