import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rtp_llm.test.perf_test.batch_decode_test import main
from rtp_llm.test.perf_test.dataset import extract_arg


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
#  Custom arg extraction (consumed before forwarding to engine)
# ---------------------------------------------------------------------------

def _extract_custom_args(argv: List[str]) -> Tuple[List[str], Optional[str], Optional[str], bool]:
    """Extract --perf_test_name, --test_env, --compare_test_result, --update_test_result from argv.

    Returns (cleaned_argv, compare_baseline_path, perf_test_name, update_test_result).
    Also sets --test_env K=V pairs as environment variables.
    """
    cleaned: List[str] = []
    compare_baseline: Optional[str] = None
    perf_test_name: Optional[str] = None
    update_test_result: bool = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--compare_test_result" and i + 1 < len(argv):
            compare_baseline = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--compare_test_result="):
            compare_baseline = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--update_test_result":
            update_test_result = True
            i += 1
            continue
        if arg == "--perf_test_name" and i + 1 < len(argv):
            perf_test_name = argv[i + 1]
            os.environ["PERF_TEST_NAME"] = perf_test_name
            i += 2
            continue
        if arg.startswith("--perf_test_name="):
            perf_test_name = arg.split("=", 1)[1]
            os.environ["PERF_TEST_NAME"] = perf_test_name
            i += 1
            continue
        if arg == "--test_env" and i + 1 < len(argv):
            kv = argv[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                os.environ[k] = v
            i += 2
            continue
        if arg.startswith("--test_env="):
            kv = arg.split("=", 1)[1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                os.environ[k] = v
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned, compare_baseline, perf_test_name, update_test_result


# ---------------------------------------------------------------------------
#  Model path conversion (removed — weight_convert_api deleted)
# ---------------------------------------------------------------------------

def _try_convert_model_path(argv: List[str]) -> List[str]:
    return argv


# ---------------------------------------------------------------------------
#  Baseline validation
# ---------------------------------------------------------------------------

REGRESSION_THRESHOLD = 0.10


def _load_baseline(baseline_path: str) -> Dict[str, Any]:
    if not os.path.exists(baseline_path):
        logging.warning(f"Baseline file not found: {baseline_path}")
        return {}
    with open(baseline_path) as f:
        return json.load(f)


def _collect_decode_times(result_dir: str) -> Dict[str, float]:
    """Extract composite-key -> avg_decode_time mapping from result JSONs.

    Grid mode:         key = "bs{batch_size}_seq{input_len}"  (cartesian product)
    Distribution mode: key = "{batch_size}"                   (batch_size is unique)
    """
    times: Dict[str, float] = {}
    if not os.path.isdir(result_dir):
        return times
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if data.get("mode") == "distribution":
            for tc in data.get("test_cases", []):
                bs = tc.get("batch_size")
                dt = tc.get("avg_decode_time_per_token")
                if bs is not None and dt is not None:
                    times[str(int(bs))] = float(dt)
        elif data.get("mode") == "grid":
            for m in data.get("metrics", []):
                bs = m.get("batch_size")
                seq = m.get("input_len")
                dt = m.get("avg_decode_time")
                success_rate = m.get("success_rate")
                if bs is not None and seq is not None and dt is not None and success_rate == 1.0:
                    times[f"bs{int(bs)}_seq{int(seq)}"] = float(dt)
    return times


def _sort_key(key: str):
    """Sort composite keys numerically: 'bs1_seq128' -> (1,128), '32' -> (32,0)."""
    nums = list(map(int, re.findall(r'\d+', key)))
    return tuple(nums) if nums else (0,)


def _format_comparison_table(
    baseline_times: Dict[str, float],
    current_times: Dict[str, float],
) -> str:
    header = f"{'key':>20} {'baseline(ms)':>14} {'current(ms)':>14} {'delta':>10} {'status':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for key in sorted(baseline_times, key=_sort_key):
        baseline_val = baseline_times[key]
        if key not in current_times:
            lines.append(f"{key:>20} {baseline_val:>14.2f} {'N/A':>14} {'N/A':>10} {'SKIP':>8}")
            continue
        current_val = current_times[key]
        delta_pct = (current_val / baseline_val - 1) * 100
        threshold = baseline_val * (1 + REGRESSION_THRESHOLD)
        status = "FAIL" if current_val > threshold else "OK"
        lines.append(
            f"{key:>20} {baseline_val:>14.2f} {current_val:>14.2f} {delta_pct:>+9.1f}% {status:>8}"
        )
    lines.append(sep)
    return "\n".join(lines)


def validate_against_baseline(result_dir: str, baseline_path: Optional[str]) -> bool:
    if not baseline_path:
        return True

    baseline = _load_baseline(baseline_path)
    if not baseline:
        return True

    baseline_times = baseline.get("decode_times", {})
    if not baseline_times:
        logging.info("Baseline has no decode_times, skip validation")
        return True

    current_times = _collect_decode_times(result_dir)
    if not current_times:
        logging.warning("No decode times in results, skip validation")
        return True

    table = _format_comparison_table(baseline_times, current_times)
    print(f"\n=== Perf Baseline Comparison (threshold: {REGRESSION_THRESHOLD * 100:.0f}%) ===")
    print(table)

    passed = True
    for key, baseline_val in baseline_times.items():
        if key not in current_times:
            continue
        current_val = current_times[key]
        threshold = baseline_val * (1 + REGRESSION_THRESHOLD)
        if current_val > threshold:
            passed = False

    if passed:
        print("Result: ALL PASSED")
    else:
        print("Result: REGRESSION DETECTED")
    print()

    return passed


# ---------------------------------------------------------------------------
#  OSS upload via ossutil
# ---------------------------------------------------------------------------

OSS_BUCKET = "oss://rtp-maga"
OSS_PREFIX = "perf_test"


def _oss_env() -> Dict[str, str]:
    return {
        "id": os.environ.get("OSS_ACCESS_KEY_ID", ""),
        "key": os.environ.get("OSS_ACCESS_KEY_SECRET", ""),
        "host": os.environ["OSS_ENDPOINT"],
    }


def _ossutil_cmd(local: str, remote: str) -> List[str]:
    cred = _oss_env()
    if not cred["id"] or not cred["key"]:
        raise EnvironmentError(
            "OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET not set, cannot upload to OSS"
        )
    return [
        "ossutil", "cp", local, remote,
        "-i", cred["id"],
        "-k", cred["key"],
        "-e", cred["host"],
        "-f",
    ]


def upload_results_to_oss(result_dir: str) -> str:
    cred = _oss_env()
    if not cred["id"] or not cred["key"]:
        logging.warning("OSS credentials not set, skip upload")
        return ""
    if not os.path.isdir(result_dir):
        logging.warning(f"result_dir {result_dir} does not exist, skip OSS upload")
        return ""

    test_name = os.environ.get("PERF_TEST_NAME", "unknown")
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    ts = now.strftime("%H%M%S")
    tar_name = f"{test_name}_{date_str}_{ts}.tar.gz"
    tar_local = os.path.join("/tmp", tar_name)

    with tarfile.open(tar_local, "w:gz") as tf:
        tf.add(result_dir, arcname=os.path.basename(result_dir))
    logging.info(f"Packed results to {tar_local} ({os.path.getsize(tar_local)} bytes)")

    oss_path = f"{OSS_BUCKET}/{OSS_PREFIX}/{date_str}/{test_name}/{tar_name}"
    cmd = _ossutil_cmd(tar_local, oss_path)
    logging.info(f"Uploading to {oss_path}")
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if ret.returncode != 0:
        logging.error(f"ossutil upload failed: {ret.stderr}")
        return ""
    logging.info(f"OSS upload success: {oss_path}")

    try:
        os.remove(tar_local)
    except OSError:
        pass
    return oss_path


# ---------------------------------------------------------------------------
#  ODPS: record write
# ---------------------------------------------------------------------------

ODPS_TABLE = "batch_decode_test_result"


def _get_odps_client():
    ak_id = os.environ.get("ODPS_ACCESS_ID", "")
    ak_secret = os.environ.get("ODPS_ACCESS_KEY", "")
    project = os.environ.get("ODPS_PROJECT", "")
    endpoint = os.environ.get("ODPS_ENDPOINT", "")
    if not ak_id or not ak_secret or not project or not endpoint:
        return None
    from odps import ODPS
    return ODPS(ak_id, ak_secret, project, endpoint=endpoint)


def _detect_mode(result_dir: str) -> str:
    if not os.path.isdir(result_dir):
        return "grid"
    for fname in os.listdir(result_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(result_dir, fname)) as f:
                if json.load(f).get("mode") == "distribution":
                    return "distribution"
        except (json.JSONDecodeError, IOError, KeyError):
            continue
    return "grid"


def _read_test_info(result_dir: str) -> Dict[str, Any]:
    info_path = os.path.join(result_dir, "test_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _parse_partial() -> int:
    for i, arg in enumerate(sys.argv):
        if arg == "--partial" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
    return 0


def write_summary_to_odps(
    result_dir: str,
    oss_path: str,
    duration_seconds: float,
):
    odps_client = _get_odps_client()
    if odps_client is None:
        logging.warning("ODPS credentials not set, skip ODPS write")
        return

    test_info = _read_test_info(result_dir)
    now = datetime.now()
    ds = now.strftime("%Y%m%d")
    partial = _parse_partial()
    mode = _detect_mode(result_dir)

    dataset_name = test_info.get("dataset_name") or ""
    if not dataset_name:
        dataset_path = test_info.get("dataset_path") or ""
        if dataset_path:
            parent = os.path.basename(os.path.dirname(dataset_path))
            dataset_name = parent if parent and parent != "." else os.path.basename(dataset_path)

    test_types = []
    if partial in (0, 1):
        test_types.append("decode")
    if partial in (0, 2):
        test_types.append("prefill")

    table = odps_client.get_table(ODPS_TABLE)
    partition_spec = f"ds='{ds}'"
    try:
        table.create_partition(partition_spec, if_not_exists=True)
    except Exception as e:
        logging.warning(f"Failed to create partition {partition_spec}: {e}")

    records = []
    for tt in test_types:
        records.append([
            os.environ.get("PERF_TEST_NAME", "unknown"),
            tt,
            mode,
            dataset_name,
            get_git_commit(),
            oss_path,
            now.isoformat(),
            round(duration_seconds, 2),
        ])

    for attempt in range(1, 6):
        try:
            with table.open_writer(partition=partition_spec) as writer:
                writer.write(records)
            logging.info(f"ODPS write success: {len(records)} record(s) (attempt {attempt})")
            return
        except Exception as e:
            logging.warning(f"ODPS write attempt {attempt} failed: {e}")
            if attempt >= 5:
                logging.error("ODPS write failed after 5 attempts, giving up")
            else:
                time.sleep(2 * attempt)


def _print_new_golden(result_dir: str, baseline_path: Optional[str]) -> None:
    """Collect current results and print them as new golden baseline JSON."""
    current_times = _collect_decode_times(result_dir)
    if not current_times:
        logging.warning("No decode times in results, cannot generate new golden")
        return

    golden = {}
    if baseline_path:
        golden = _load_baseline(baseline_path)
    golden["decode_times"] = {k: dt for k, dt in sorted(current_times.items(), key=lambda x: _sort_key(x[0]))}
    golden["updated"] = datetime.now().isoformat()

    golden_json = json.dumps(golden, indent=4)

    print("\n" + "=" * 60)
    print("  NEW GOLDEN BASELINE — copy the JSON below to update")
    print("=" * 60)
    print(golden_json)
    print("=" * 60 + "\n")
    logging.info(
        "This is the NEW golden result. "
        "Replace the contents of your baseline JSON file with the output above."
    )


def write_test_meta(result_dir: str):
    env_keys = [
        "DEVICE_NAME", "DEVICE_RESERVE_MEMORY_BYTES", "RESERVER_RUNTIME_MEM_MB",
        "SEQ_SIZE_PER_BLOCK", "INT8_MODE", "QUANTIZATION",
    ]
    env_args = {k: os.environ[k] for k in env_keys if k in os.environ}
    meta = {
        "test_name": os.environ.get("PERF_TEST_NAME", "unknown"),
        "env_args": env_args,
        "git_commit": get_git_commit(),
        "test_timestamp": datetime.now().isoformat(),
    }
    os.makedirs(result_dir, exist_ok=True)
    meta_path = os.path.join(result_dir, "test_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Wrote test meta to {meta_path}")


if __name__ == "__main__":
    argv, compare_baseline, _, update_test_result = _extract_custom_args(list(sys.argv))
    sys.argv = _try_convert_model_path(argv)

    start_time = time.time()
    result_dir = main()
    duration = time.time() - start_time

    write_test_meta(result_dir)
    _print_new_golden(result_dir, compare_baseline)
    # not upload oss
    if update_test_result:
        pass
    elif compare_baseline:
        baseline_ok = validate_against_baseline(result_dir, compare_baseline)
        if not baseline_ok:
            logging.error("PERF REGRESSION DETECTED — test marked as FAILED")
            sys.exit(1)
    else:
        oss_path = upload_results_to_oss(result_dir)
        write_summary_to_odps(result_dir, oss_path, duration)
