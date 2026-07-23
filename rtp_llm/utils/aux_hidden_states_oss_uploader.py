from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_ENDPOINT = "oss-cn-shanghai.aliyuncs.com"

AUX_READY_DIR_ENV = "AUX_HIDDEN_STATES_READY_DIR"
AUX_BASE_DIR_ENV = "AUX_HIDDEN_STATES_BASE_DIR"
AUX_UPLOAD_ENABLE_ENV = "AUX_HIDDEN_STATES_OSS_UPLOAD"
AUX_UPLOAD_WORK_DIR_ENV = "AUX_HIDDEN_STATES_UPLOAD_WORK_DIR"
AUX_OSS_PREFIX_ENV = "AUX_HIDDEN_STATES_OSS_PREFIX"
AUX_OSS_ENDPOINT_ENV = "AUX_HIDDEN_STATES_OSS_ENDPOINT"
AUX_OSSUTIL_BIN_ENV = "AUX_HIDDEN_STATES_OSSUTIL_BIN"
AUX_UPLOAD_BATCH_SIZE_ENV = "AUX_HIDDEN_STATES_UPLOAD_BATCH_SIZE"
AUX_UPLOAD_MAX_BATCH_BYTES_ENV = "AUX_HIDDEN_STATES_UPLOAD_MAX_BATCH_BYTES"
AUX_UPLOAD_INTERVAL_SEC_ENV = "AUX_HIDDEN_STATES_UPLOAD_INTERVAL_SEC"
AUX_UPLOAD_MIN_AGE_SEC_ENV = "AUX_HIDDEN_STATES_UPLOAD_MIN_AGE_SEC"
AUX_UPLOAD_POLL_SEC_ENV = "AUX_HIDDEN_STATES_UPLOAD_POLL_SEC"
AUX_UPLOAD_KEEP_UPLOADED_ENV = "AUX_HIDDEN_STATES_UPLOAD_KEEP_LOCAL"
AUX_OSS_APPEND_WORKER_ID_ENV = "AUX_HIDDEN_STATES_OSS_APPEND_WORKER_ID"

OSSUTIL_CONFIG_FILE_ENV = "OSSUTIL_CONFIG_FILE"
OSS_ACCESS_KEY_ID_ENV = "OSS_ACCESS_KEY_ID"
OSS_ACCESS_KEY_SECRET_ENV = "OSS_ACCESS_KEY_SECRET"
AUX_OSS_ACCESS_KEY_ID_ENV = "AUX_HIDDEN_STATES_OSS_ACCESS_KEY_ID"
AUX_OSS_ACCESS_KEY_SECRET_ENV = "AUX_HIDDEN_STATES_OSS_ACCESS_KEY_SECRET"


class UploadError(RuntimeError):
    pass


@dataclass
class UploaderConfig:
    ready_dir: Path
    work_dir: Path
    oss_prefix: str
    endpoint: str = DEFAULT_ENDPOINT
    ossutil_bin: Optional[str] = None
    ready_glob: str = "*.ready"
    batch_size: int = 128
    max_batch_bytes: int = 0
    batch_interval_sec: float = 10.0
    min_age_sec: float = 0.0
    poll_sec: float = 1.0
    keep_uploaded: bool = False
    once: bool = False


def log(message: str) -> None:
    logging.info("[aux_hidden_states_oss_uploader] %s", message)
    print(f"[aux_hidden_states_oss_uploader] {message}", flush=True)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "on", "yes", "y")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def is_upload_enabled() -> bool:
    return env_flag(AUX_UPLOAD_ENABLE_ENV, default=False)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def find_ossutil(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for candidate in ("ossutil_x86_64", "ossutil64", "ossutil"):
        path = shutil.which(candidate)
        if path:
            return path
    raise UploadError(
        "ossutil not found. Set AUX_HIDDEN_STATES_OSSUTIL_BIN/OSSUTIL_BIN "
        "or install ossutil/ossutil64."
    )


def build_ossutil_config(work_dir: Path, endpoint: str) -> List[str]:
    explicit_config = os.environ.get(OSSUTIL_CONFIG_FILE_ENV)
    if explicit_config:
        return ["-c", explicit_config]

    access_key_id = os.environ.get(AUX_OSS_ACCESS_KEY_ID_ENV) or os.environ.get(
        OSS_ACCESS_KEY_ID_ENV
    )
    access_key_secret = os.environ.get(AUX_OSS_ACCESS_KEY_SECRET_ENV) or os.environ.get(
        OSS_ACCESS_KEY_SECRET_ENV
    )
    if not access_key_id or not access_key_secret:
        return []

    config_path = work_dir / "ossutilconfig"
    content = (
        "[Credentials]\n"
        "language=EN\n"
        f"endpoint={endpoint}\n"
        f"accessKeyID={access_key_id}\n"
        f"accessKeySecret={access_key_secret}\n"
        "stsToken=\n"
    )
    old_umask = os.umask(0o177)
    try:
        write_text_atomic(config_path, content)
    finally:
        os.umask(old_umask)
    return ["-c", str(config_path)]


def join_oss(prefix: str, *parts: str) -> str:
    base = prefix.rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part)
    return f"{base}/{suffix}" if suffix else base


def sanitize_path_part(value: str) -> str:
    clean = []
    for char in value:
        if char.isalnum() or char in ("-", "_", ".", "="):
            clean.append(char)
        else:
            clean.append("_")
    return "".join(clean).strip("_")


def worker_identity_parts() -> List[str]:
    parts = []
    app = os.environ.get("HIPPO_APP", "").strip()
    role = (
        os.environ.get("HIPPO_ROLE_SHORT_NAME", "").strip()
        or os.environ.get("HIPPO_ROLE", "").strip()
    )
    host = os.environ.get("HIPPO_SLAVE_IP", "").strip() or socket.gethostname()
    rank = os.environ.get("WORLD_RANK", "").strip()
    for key, value in (("app", app), ("role", role), ("host", host), ("rank", rank)):
        if value:
            parts.append(sanitize_path_part(f"{key}={value}"))
    return parts


def maybe_append_worker_id(oss_prefix: str) -> str:
    if not env_flag(AUX_OSS_APPEND_WORKER_ID_ENV, default=False):
        return oss_prefix
    return join_oss(oss_prefix, *worker_identity_parts())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_base_dir() -> Path:
    explicit = os.environ.get(AUX_BASE_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit)
    hippo_workdir = os.environ.get("HIPPO_PROC_WORKDIR", "").strip()
    if hippo_workdir:
        return Path(hippo_workdir) / "aux_hidden_states"
    log_dir = os.environ.get("LOG_DIR", "").strip()
    if log_dir:
        return Path(log_dir).parent / "aux_hidden_states"
    return Path("aux_hidden_states")


def build_config_from_env() -> UploaderConfig:
    base_dir = default_base_dir()
    ready_dir = Path(os.environ.get(AUX_READY_DIR_ENV, "").strip() or base_dir / "ready")
    work_dir = Path(
        os.environ.get(AUX_UPLOAD_WORK_DIR_ENV, "").strip()
        or base_dir / "upload_work"
    )
    oss_prefix = (
        os.environ.get(AUX_OSS_PREFIX_ENV, "").strip()
        or os.environ.get("OSS_PREFIX", "").strip()
    )
    endpoint = (
        os.environ.get(AUX_OSS_ENDPOINT_ENV, "").strip()
        or os.environ.get("OSS_ENDPOINT", "").strip()
        or DEFAULT_ENDPOINT
    )
    ossutil_bin = (
        os.environ.get(AUX_OSSUTIL_BIN_ENV, "").strip()
        or os.environ.get("OSSUTIL_BIN", "").strip()
        or None
    )
    return UploaderConfig(
        ready_dir=ready_dir,
        work_dir=work_dir,
        oss_prefix=maybe_append_worker_id(oss_prefix),
        endpoint=endpoint,
        ossutil_bin=ossutil_bin,
        batch_size=env_int(AUX_UPLOAD_BATCH_SIZE_ENV, 128),
        max_batch_bytes=env_int(AUX_UPLOAD_MAX_BATCH_BYTES_ENV, 0),
        batch_interval_sec=env_float(AUX_UPLOAD_INTERVAL_SEC_ENV, 10.0),
        min_age_sec=env_float(AUX_UPLOAD_MIN_AGE_SEC_ENV, 0.0),
        poll_sec=env_float(AUX_UPLOAD_POLL_SEC_ENV, 1.0),
        keep_uploaded=env_flag(AUX_UPLOAD_KEEP_UPLOADED_ENV, default=False),
    )


def apply_service_env(config: UploaderConfig) -> None:
    os.environ[AUX_READY_DIR_ENV] = str(config.ready_dir)


def collect_ready_files(config: UploaderConfig) -> List[Path]:
    now = time.time()
    files: List[Path] = []
    total_bytes = 0
    for path in sorted(
        config.ready_dir.glob(config.ready_glob), key=lambda p: p.stat().st_mtime
    ):
        if not path.is_file():
            continue
        stat = path.stat()
        if config.min_age_sec > 0 and now - stat.st_mtime < config.min_age_sec:
            continue
        if (
            files
            and config.max_batch_bytes > 0
            and total_bytes + stat.st_size > config.max_batch_bytes
        ):
            break
        files.append(path)
        total_bytes += stat.st_size
        if len(files) >= config.batch_size:
            break
    return files


def should_flush(files: Sequence[Path], config: UploaderConfig) -> bool:
    if not files:
        return False
    if len(files) >= config.batch_size:
        return True
    oldest_mtime = min(path.stat().st_mtime for path in files)
    return time.time() - oldest_mtime >= config.batch_interval_sec


def object_name_for(path: Path, used: Dict[str, int]) -> str:
    name = path.name
    if name.endswith(".ready"):
        name = name[: -len(".ready")]
    count = used.get(name, 0)
    used[name] = count + 1
    if count == 0:
        return name
    return f"{count:04d}_{name}"


def move_into_batch(
    files: Sequence[Path],
    batch_files_dir: Path,
) -> List[Tuple[Path, Path, str]]:
    ensure_dir(batch_files_dir)
    used_object_names: Dict[str, int] = {}
    staged: List[Tuple[Path, Path, str]] = []
    for source in files:
        object_name = object_name_for(source, used_object_names)
        target = batch_files_dir / source.name
        try:
            source.rename(target)
        except OSError:
            shutil.move(str(source), str(target))
        staged.append((source, target, object_name))
    return staged


def build_direct_entries(
    staged: Sequence[Tuple[Path, Path, str]],
    oss_prefix: str,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for original, staged_path, object_name in staged:
        stat = staged_path.stat()
        entries.append(
            {
                "source_name": original.name,
                "object_name": object_name,
                "remote_object": join_oss(oss_prefix, "files", object_name),
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "sha256": sha256_file(staged_path),
            }
        )
    return entries


def requeue(staged: Sequence[Tuple[Path, Path, str]]) -> None:
    for original, staged_path, _ in staged:
        if not staged_path.exists():
            continue
        ensure_dir(original.parent)
        if original.exists():
            retry_name = f"{original.stem}.retry_{int(time.time())}{original.suffix}"
            original = original.with_name(retry_name)
        try:
            staged_path.rename(original)
        except OSError:
            shutil.move(str(staged_path), str(original))


def run_ossutil_cp(
    ossutil_bin: str,
    config_args: Sequence[str],
    local_path: Path,
    remote_path: str,
) -> None:
    cmd = [ossutil_bin, *config_args, "cp", str(local_path), remote_path, "-u"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise UploadError(
            f"ossutil cp failed for {local_path} -> {remote_path}: "
            f"stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}"
        )


def upload_batch(
    files: Sequence[Path],
    config: UploaderConfig,
    ossutil_bin: str,
    config_args: Sequence[str],
) -> bool:
    batch_id = (
        f"aux_hidden_{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    batch_dir = ensure_dir(config.work_dir / "uploading" / batch_id)
    batch_files_dir = batch_dir / "files"
    manifest_path = batch_dir / f"{batch_id}.json"
    staged: List[Tuple[Path, Path, str]] = []

    try:
        staged = move_into_batch(files, batch_files_dir)
        entries = build_direct_entries(staged, config.oss_prefix)
        manifest_object = join_oss(config.oss_prefix, "manifests", manifest_path.name)
        manifest = {
            "batch_id": batch_id,
            "created_at": int(time.time()),
            "host": socket.gethostname(),
            "worker_identity": worker_identity_parts(),
            "upload_mode": "direct",
            "file_count": len(entries),
            "files": entries,
        }
        write_text_atomic(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

        log(
            f"uploading direct batch {batch_id}: files={len(entries)} "
            f"remote_prefix={join_oss(config.oss_prefix, 'files')}"
        )
        for entry, (_, staged_path, _) in zip(entries, staged):
            run_ossutil_cp(
                ossutil_bin,
                config_args,
                staged_path,
                str(entry["remote_object"]),
            )
        run_ossutil_cp(ossutil_bin, config_args, manifest_path, manifest_object)

        uploaded_manifest_dir = ensure_dir(config.work_dir / "uploaded_manifests")
        shutil.copy2(manifest_path, uploaded_manifest_dir / manifest_path.name)
        if config.keep_uploaded:
            uploaded_dir = ensure_dir(config.work_dir / "uploaded")
            shutil.move(str(batch_dir), str(uploaded_dir / batch_id))
        else:
            shutil.rmtree(batch_dir, ignore_errors=True)
        log(f"uploaded batch {batch_id}: manifest={manifest_object}")
        return True
    except Exception as exc:
        log(f"batch {batch_id} failed: {exc}")
        requeue(staged)
        failed_dir = ensure_dir(config.work_dir / "failed" / batch_id)
        if manifest_path.exists():
            shutil.move(str(manifest_path), str(failed_dir / manifest_path.name))
        shutil.rmtree(batch_dir, ignore_errors=True)
        return False


def run_uploader(config: UploaderConfig) -> int:
    config.ready_dir = ensure_dir(config.ready_dir)
    config.work_dir = ensure_dir(config.work_dir)
    ensure_dir(config.work_dir / "uploading")
    ensure_dir(config.work_dir / "failed")
    if not config.oss_prefix.startswith("oss://"):
        raise UploadError(
            "invalid OSS prefix. Set AUX_HIDDEN_STATES_OSS_PREFIX or OSS_PREFIX "
            "to an oss:// bucket path."
        )

    ossutil_bin = find_ossutil(config.ossutil_bin)
    config_args = build_ossutil_config(config.work_dir, config.endpoint)
    if not config_args:
        log(
            "using default ossutil config; set OSSUTIL_CONFIG_FILE or "
            "OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET if needed"
        )

    log(
        f"watching ready_dir={config.ready_dir} glob={config.ready_glob} "
        f"oss_prefix={config.oss_prefix} batch_size={config.batch_size}"
    )

    had_failure = False
    while True:
        files = collect_ready_files(config)
        if config.once:
            if files:
                had_failure = (
                    not upload_batch(files, config, ossutil_bin, config_args)
                    or had_failure
                )
            return 1 if had_failure else 0

        if should_flush(files, config):
            had_failure = (
                not upload_batch(files, config, ossutil_bin, config_args)
                or had_failure
            )
        time.sleep(config.poll_sec)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> UploaderConfig:
    env_config = build_config_from_env()
    parser = argparse.ArgumentParser(
        description=(
            "Direct-upload completed aux/mtp hidden-state files to OSS. "
            "Only files matching --ready-glob are consumed; producers should "
            "write *.tmp first and atomically rename to *.ready."
        )
    )
    parser.add_argument("--ready-dir", type=Path, default=env_config.ready_dir)
    parser.add_argument("--work-dir", type=Path, default=env_config.work_dir)
    parser.add_argument("--ready-glob", default=env_config.ready_glob)
    parser.add_argument("--oss-prefix", default=env_config.oss_prefix)
    parser.add_argument("--endpoint", default=env_config.endpoint)
    parser.add_argument("--ossutil-bin", default=env_config.ossutil_bin)
    parser.add_argument("--batch-size", type=positive_int, default=env_config.batch_size)
    parser.add_argument(
        "--max-batch-bytes",
        type=non_negative_int,
        default=env_config.max_batch_bytes,
    )
    parser.add_argument(
        "--batch-interval-sec",
        type=float,
        default=env_config.batch_interval_sec,
    )
    parser.add_argument("--min-age-sec", type=float, default=env_config.min_age_sec)
    parser.add_argument("--poll-sec", type=float, default=env_config.poll_sec)
    parser.add_argument("--once", action="store_true", default=env_config.once)
    parser.add_argument(
        "--keep-uploaded",
        action="store_true",
        default=env_config.keep_uploaded,
    )
    args = parser.parse_args(argv)
    return UploaderConfig(
        ready_dir=args.ready_dir,
        work_dir=args.work_dir,
        ready_glob=args.ready_glob,
        oss_prefix=args.oss_prefix,
        endpoint=args.endpoint,
        ossutil_bin=args.ossutil_bin,
        batch_size=args.batch_size,
        max_batch_bytes=args.max_batch_bytes,
        batch_interval_sec=args.batch_interval_sec,
        min_age_sec=args.min_age_sec,
        poll_sec=args.poll_sec,
        once=args.once,
        keep_uploaded=args.keep_uploaded,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_uploader(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
