"""本地 frontend 日志匿名采集；数据对象见 traffic.capture_model。

运行环境与部署定位由调用方负责。本模块只读取已可访问的日志文件。
"""

import argparse, array, collections, glob, gzip, hashlib, json, lzma, os, socket, time, sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from file_digest import sha256_file
from traffic.capture_model import CaptureMetadata, CaptureRecord
from traffic.lineage_model import DEFAULT_PREFIX_BLOCK_SIZE, prefix_block_size


class BudgetExceeded(Exception):
    pass


class CaptureInputError(ValueError):
    """只含固定诊断原因，可安全写入 summary。"""


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=("gzip", "xz"), default="gzip")
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--log-glob", default="dash_sc_grpc_access_r0_s*.log*")
    p.add_argument("--time-budget-s", type=float, default=900)
    p.add_argument("--block-size", type=int, default=DEFAULT_PREFIX_BLOCK_SIZE)
    p.add_argument("--on-budget", choices=("error", "truncate"), default="error")
    return p


def capture(a, *, clock=time.monotonic):
    """同一核心用于容器目录或下载目录；clock 可用于确定性护栏验证。"""
    if not 0 < a.time_budget_s < float("inf"):
        raise ValueError("invalid capture guard parameters")
    if a.start >= a.end:
        raise ValueError("capture start must precede end")
    metadata = CaptureMetadata(block_size=prefix_block_size(a.block_size))
    block_size = metadata.block_size
    stats = collections.Counter()
    coverage = []
    seen = set()
    started = clock()
    truncated = False
    failure = None
    errors = []
    output_path = a.out + (".jsonl.xz" if a.format == "xz" else ".jsonl.gz")
    out = lzma.open(output_path, "wt", preset=3) if a.format == "xz" else gzip.open(output_path, "wt", compresslevel=1)
    try:
        paths = [path for path in sorted(glob.glob(str(a.log_dir / a.log_glob)))
                 if os.path.isfile(path) and not path.endswith(".gz")]
        if not paths:
            errors.append("no supported log files matched")
            failure = ValueError("no supported log files matched")
        for path in paths:
            if clock() - started > a.time_budget_s:
                raise BudgetExceeded("extraction time budget exceeded")
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                if not size:
                    continue
                file_coverage = {
                    "path": path, "inode": os.fstat(f.fileno()).st_ino,
                    "bytes": size, "first_completion": None, "last_completion": None,
                }
                coverage.append(file_coverage)
                line_number = 0
                while f.tell() < size:
                    line = f.readline()
                    line_number += 1
                    stats["scanned"] += 1
                    if stats["scanned"] % 1000 == 0 and clock() - started > a.time_budget_s:
                        raise BudgetExceeded("extraction time budget exceeded")
                    try:
                        r = json.loads(line)
                    except Exception:
                        stats["invalid_json"] += 1
                        continue
                    if not isinstance(r, dict):
                        stats["invalid_json"] += 1
                        continue
                    completion = r.get("ts_epoch_ms")
                    if type(completion) is int:
                        if file_coverage["first_completion"] is None:
                            file_coverage["first_completion"] = completion
                        file_coverage["last_completion"] = completion
                    if not r.get("request_enter_ts_epoch_ms"):
                        stats["missing_arrival_timestamp"] += 1
                        continue
                    ts = int(r["request_enter_ts_epoch_ms"])
                    if not a.start <= ts < a.end:
                        continue
                    rid = str(r.get("request_id") or r.get("upstream_request_id") or "")
                    if not rid:
                        stats["missing_request_identity"] += 1
                        continue
                    # Log rotation overlap is deduplicated by local request identity + arrival.
                    ident = (rid, ts)
                    if ident in seen:
                        stats["duplicate"] += 1
                        continue
                    ids = r.get("input_ids")
                    if not isinstance(ids, list) or not ids:
                        stats["missing_tokens"] += 1
                        continue
                    if any(type(token) is not int or not -2147483648 <= token <= 2147483647 for token in ids):
                        stats["invalid_tokens"] += 1
                        raise CaptureInputError("invalid int32 input tokens")
                    values = array.array("i", ids)
                    if values.itemsize != 4:
                        raise RuntimeError("capture requires 4-byte int32")
                    if __import__("sys").byteorder != "little":
                        values.byteswap()
                    raw = values.tobytes()
                    hasher = hashlib.sha256()
                    keys = []
                    for i in range(0, len(ids) // block_size * (block_size * 4), block_size * 4):
                        hasher.update(raw[i : i + block_size * 4])
                        keys.append(hasher.hexdigest()[:32])
                    cfg = r.get("generate_config") or {}
                    aux = r.get("aux_info") or {}
                    try:
                        record = CaptureRecord(
                            schema_version=metadata.schema_version, block_size=metadata.block_size,
                            ts=ts, il=len(ids), ol=r.get("output_token_len"), keys=tuple(keys),
                            tail_hash=hashlib.sha256(raw).hexdigest()[:32],
                            rid=hashlib.sha256(rid.encode()).hexdigest()[:32],
                            status=r.get("status"), error=r.get("backend_error_code"),
                            cached=r.get("prompt_cached_token_num"), aux_reuse=aux.get("reuse_len"),
                            latency_ms=r.get("latency_total_ms"), priority=cfg.get("traffic_reject_priority"),
                            max_new_tokens=cfg.get("max_new_tokens"), timeout_ms=cfg.get("timeout_ms"))
                    except ValueError:
                        stats["contract_errors"] += 1
                        raise
                    seen.add(ident)
                    out.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
                    stats["records"] += 1
                    stats["tokens"] += len(ids)
                    stats["status:" + str(record.status)] += 1
    except BudgetExceeded as exc:
        truncated = True
        stats["budget_exceeded"] += 1
        errors.append(str(exc))
        if a.on_budget == "error":
            failure = exc
    except Exception as exc:
        # 不写日志原文或原始 token 到错误产物。
        stats["fatal_errors"] += 1
        safe_reason = isinstance(exc, CaptureInputError) or (
            isinstance(exc, ValueError) and "contract_errors" in stats
        )
        errors.append(str(exc) if safe_reason else type(exc).__name__)
        failure = exc
    finally:
        out.close()
    if stats["invalid_json"] and failure is None:
        errors.append("invalid JSON log records")
        failure = CaptureInputError("invalid JSON log records")
    summary = {
        **metadata.to_dict(),
        "complete": not truncated and failure is None,
        "truncated": truncated,
        "errors": errors,
        "parameters": dict(log_dir=str(a.log_dir), log_glob=a.log_glob,
                           time_budget_s=a.time_budget_s, block_size=block_size,
                           on_budget=a.on_budget),
        "hostname": socket.gethostname(),
        "start": a.start,
        "end": a.end,
        "stats": dict(stats),
        "files": coverage,
        "elapsed_s": clock() - started,
        "output_bytes": os.path.getsize(output_path),
        "sha256": sha256_file(output_path),
        "hash": f"SHA256 prefix, 128 bit, little-endian int32, {block_size} tokens",
    }
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)

    if failure:
        raise RuntimeError(f"capture incomplete; see {a.out}.summary.json") from None
    return summary


def main():
    capture(parser().parse_args())


if __name__ == "__main__":
    main()
