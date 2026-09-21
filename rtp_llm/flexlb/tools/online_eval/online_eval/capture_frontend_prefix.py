"""Bounded pod-local anonymized access-log extraction; no service changes."""

import argparse, array, collections, glob, gzip, hashlib, json, os, socket, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.start >= a.end:
        raise ValueError("capture start must precede end")
    stats = collections.Counter()
    coverage = []
    seen = set()
    started = time.time()
    out = gzip.open(a.out + ".jsonl.gz", "wt", compresslevel=1)
    try:
        for path in sorted(glob.glob("logs/dash_sc_grpc_access_r0_s*.log*")):
            if time.time() - started > 900:
                raise RuntimeError("extraction time budget exceeded")
            if not os.path.isfile(path) or path.endswith(".gz"):
                continue
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                if not size:
                    continue
                try:
                    first = json.loads(f.readline())
                except Exception:
                    stats["invalid_first"] += 1
                    continue
                f.seek(max(0, size - 16000000))
                lines = f.read(size - f.tell()).splitlines()
                last = None
                for line in reversed(lines):
                    try:
                        last = json.loads(line)
                        break
                    except Exception:
                        pass
                if last is None:
                    stats["invalid_last"] += 1
                    continue
                lo = int(first.get("ts_epoch_ms") or 0)
                hi = int(last.get("ts_epoch_ms") or 0)
                # Include completions up to 5 minutes later; preserve requested arrival window.
                if hi < a.start or lo > a.end + 300000:
                    continue
                coverage.append(
                    {
                        "path": path,
                        "inode": os.fstat(f.fileno()).st_ino,
                        "bytes": size,
                        "first_completion": lo,
                        "last_completion": hi,
                    }
                )
                f.seek(0)
                while f.tell() < size:
                    line = f.readline()
                    stats["scanned"] += 1
                    if stats["scanned"] % 1000 == 0 and time.time() - started > 900:
                        raise RuntimeError("extraction time budget exceeded")
                    try:
                        r = json.loads(line)
                    except Exception:
                        stats["invalid_json"] += 1
                        continue
                    ts = int(
                        r.get("request_enter_ts_epoch_ms") or r.get("ts_epoch_ms") or 0
                    )
                    if not a.start <= ts < a.end:
                        continue
                    rid = str(r.get("upstream_request_id") or r.get("request_id") or "")
                    # Log rotation overlap is deduplicated by local request identity + arrival.
                    ident = (str(r.get("request_id")), ts)
                    if ident in seen:
                        stats["duplicate"] += 1
                        continue
                    seen.add(ident)
                    ids = r.get("input_ids")
                    if not isinstance(ids, list) or not ids:
                        stats["missing_tokens"] += 1
                        continue
                    values = array.array("i", ids)
                    if __import__("sys").byteorder != "little":
                        values.byteswap()
                    raw = values.tobytes()
                    hasher = hashlib.sha256()
                    keys = []
                    for i in range(0, len(ids) // 512 * 2048, 2048):
                        hasher.update(raw[i : i + 2048])
                        keys.append(hasher.hexdigest()[:32])
                    cfg = r.get("generate_config") or {}
                    aux = r.get("aux_info") or {}
                    row = {
                        "ts": ts,
                        "il": len(ids),
                        "ol": r.get("output_token_len"),
                        "keys": keys,
                        "tail_hash": hashlib.sha256(raw).hexdigest()[:32],
                        "rid": hashlib.sha256(rid.encode()).hexdigest()[:32],
                        "status": r.get("status"),
                        "error": r.get("backend_error_code"),
                        "cached": r.get("prompt_cached_token_num"),
                        "aux_reuse": aux.get("reuse_len"),
                        "latency_ms": r.get("latency_total_ms"),
                        "priority": cfg.get("traffic_reject_priority"),
                        "max_new_tokens": cfg.get("max_new_tokens"),
                        "timeout_ms": cfg.get("timeout_ms"),
                    }
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    stats["records"] += 1
                    stats["tokens"] += len(ids)
                    stats["status:" + str(row["status"])] += 1
    finally:
        out.close()
    summary = {
        "hostname": socket.gethostname(),
        "start": a.start,
        "end": a.end,
        "stats": dict(stats),
        "files": coverage,
        "elapsed_s": time.time() - started,
        "output_bytes": os.path.getsize(a.out + ".jsonl.gz"),
        "sha256": hashlib.sha256(open(a.out + ".jsonl.gz", "rb").read()).hexdigest(),
        "hash": "SHA256 prefix, 128 bit, little-endian int32, 512 tokens",
    }
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
